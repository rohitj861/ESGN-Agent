"""PySpark transform: raw article JSONL -> curated, deduplicated ESG dataset.

Reads the raw landing zone, applies enrichment, filters to the trailing N-day
window, collapses duplicates (same story syndicated across outlets) and writes
partitioned Parquet plus a few rollups.
"""

from __future__ import annotations

import logging
import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, BooleanType, DoubleType, StringType, StructField, StructType,
)

from .enrich import enrich

log = logging.getLogger(__name__)

RAW_SCHEMA = StructType([
    StructField("article_id", StringType()),
    StructField("title", StringType()),
    StructField("url", StringType()),
    StructField("canonical_url", StringType()),
    StructField("summary", StringType()),
    StructField("author", StringType()),
    StructField("published_at", StringType()),
    StructField("published_raw", StringType()),
    StructField("categories", ArrayType(StringType())),
    StructField("source_id", StringType()),
    StructField("source_name", StringType()),
    StructField("source_market", StringType()),
    StructField("source_type", StringType()),
    StructField("language", StringType()),
    StructField("feed_url", StringType()),
    StructField("fetched_at", StringType()),
])

ENRICH_SCHEMA = StructType([
    StructField("esg_pillars", ArrayType(StringType())),
    StructField("regulations", ArrayType(StringType())),
    StructField("topics", ArrayType(StringType())),
    StructField("markets", ArrayType(StringType())),
    StructField("is_eu", BooleanType()),
    StructField("is_us", BooleanType()),
    StructField("relevance_score", DoubleType()),
    StructField("has_publish_date", BooleanType()),
    StructField("title_key", StringType()),
    StructField("dedupe_key", StringType()),
])

_ENRICH_FIELDS = [f.name for f in ENRICH_SCHEMA.fields]


# Each local Python worker is a full interpreter process. Fanning out to every
# core (local[*] on a 12-core box) reliably crashed workers here with
# "Python worker exited unexpectedly"; 4 is stable and still faster than the
# driver alone. Override with --master when running on a real cluster.
DEFAULT_MASTER = "local[4]"


def get_spark(app_name="esgn-agent", master=DEFAULT_MASTER, shuffle_partitions=8):
    # Spark launches Python workers by invoking bare "python". On Windows that
    # hits the Microsoft Store App Execution Alias instead of the real
    # interpreter and every UDF task dies with "Python was not found". Pin both
    # ends to the interpreter actually running this process.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def _enrich_udf():
    def _apply(article_id, title, summary, categories, source_market, source_type):
        row = {
            "article_id": article_id,
            "title": title,
            "summary": summary,
            "categories": list(categories or []),
            "source_market": source_market,
            "source_type": source_type,
            "canonical_url": None,
            "published_at": None,
        }
        out = enrich(row)
        return tuple(out[f] for f in _ENRICH_FIELDS)

    return F.udf(_apply, ENRICH_SCHEMA)


def hadoop_native_available():
    """Whether Hadoop's native Windows filesystem layer is usable.

    Spark reaches the local filesystem through Hadoop, which on Windows needs
    winutils.exe and hadoop.dll. Without them every read/write dies with
    UnsatisfiedLinkError on NativeIO$Windows.access0. Everywhere else this is a
    non-issue, so only Windows is probed.
    """
    if os.name != "nt":
        return True

    import shutil
    from pathlib import Path

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and (Path(hadoop_home) / "bin" / "winutils.exe").exists():
        return True
    return shutil.which("winutils.exe") is not None


def read_raw(spark, raw_path, driver_io=None):
    """Read the raw JSONL landing zone.

    driver_io=True loads the JSONL in the driver and parallelises it, bypassing
    Hadoop's filesystem layer. Defaults to auto-detection.
    """
    if driver_io is None:
        driver_io = not hadoop_native_available()

    if not driver_io:
        return (
            spark.read
            .schema(RAW_SCHEMA)
            .option("recursiveFileLookup", "true")
            .json(str(raw_path))
        )

    from .local_job import read_raw as read_raw_local

    log.info("Hadoop native libs unavailable - reading via the driver")
    rows = read_raw_local(raw_path)
    if not rows:
        return spark.createDataFrame([], RAW_SCHEMA)

    fields = [f.name for f in RAW_SCHEMA.fields]
    tuples = [tuple(r.get(f) for f in fields) for r in rows]
    return spark.createDataFrame(tuples, RAW_SCHEMA)


def transform(df, days=7, min_relevance=0.15, now=None):
    """Enrich, filter to the window, score and deduplicate."""
    udf = _enrich_udf()

    df = df.filter(F.col("title").isNotNull() & F.col("url").isNotNull())

    df = (
        df
        .withColumn("published_ts", F.to_timestamp("published_at"))
        .withColumn("fetched_ts", F.to_timestamp("fetched_at"))
        .withColumn(
            "_e",
            udf(
                F.col("article_id"), F.col("title"), F.col("summary"),
                F.col("categories"), F.col("source_market"), F.col("source_type"),
            ),
        )
    )

    for field in _ENRICH_FIELDS:
        df = df.withColumn(field, F.col("_e").getField(field))
    df = df.drop("_e")

    # dedupe_key from the UDF has no URL context; rebuild it here.
    df = df.withColumn(
        "dedupe_key",
        F.coalesce(F.col("canonical_url"), F.col("title_key")),
    )

    # --- 7-day window ------------------------------------------------------
    now_col = F.lit(now).cast("timestamp") if now else F.current_timestamp()
    cutoff = F.expr(f"date_sub(current_date(), {days})").cast("timestamp") \
        if now is None else (now_col - F.expr(f"INTERVAL {days} DAYS"))

    df = df.filter(
        F.col("published_ts").isNull()  # keep undated regulator items
        | ((F.col("published_ts") >= cutoff)
           & (F.col("published_ts") <= now_col + F.expr("INTERVAL 12 HOURS")))
    )

    # --- ESG relevance gate ------------------------------------------------
    df = df.filter(
        ((F.col("source_type") == "regulator") & (F.size("regulations") > 0))
        | ((F.size("esg_pillars") > 0) & (F.col("relevance_score") >= min_relevance))
        | (F.col("relevance_score") >= min_relevance + 0.15)
    )

    # --- effective timestamp used for ordering / partitioning --------------
    df = df.withColumn(
        "event_ts", F.coalesce(F.col("published_ts"), F.col("fetched_ts"))
    ).withColumn("published_date", F.to_date("event_ts"))

    # --- deduplication -----------------------------------------------------
    # Prefer regulators, then higher relevance, then the earliest publisher.
    source_rank = F.when(F.col("source_type") == "regulator", 0).otherwise(1)

    by_url = Window.partitionBy("dedupe_key").orderBy(
        source_rank.asc(), F.col("relevance_score").desc(), F.col("event_ts").asc()
    )
    df = (
        df.withColumn("_rn", F.row_number().over(by_url))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Same headline syndicated to different URLs on the same day.
    by_title = Window.partitionBy("title_key", "published_date").orderBy(
        source_rank.asc(), F.col("relevance_score").desc(), F.col("event_ts").asc()
    )
    df = (
        df.withColumn("_rn", F.row_number().over(by_title))
        .withColumn(
            "duplicate_count",
            F.count("*").over(Window.partitionBy("title_key", "published_date")),
        )
        .filter((F.col("_rn") == 1) | (F.length("title_key") < 15))
        .drop("_rn")
    )

    df = df.withColumn(
        "primary_market",
        F.when(F.col("is_eu") & F.col("is_us"), F.lit("EU_US"))
        .when(F.col("is_eu"), F.lit("EU"))
        .when(F.col("is_us"), F.lit("US"))
        .otherwise(F.lit("GLOBAL")),
    ).withColumn("ingest_date", F.current_date())

    return df.select(
        "article_id", "title", "url", "canonical_url", "summary", "author",
        "source_id", "source_name", "source_type", "source_market",
        "language", "published_ts", "published_date", "fetched_ts",
        "has_publish_date", "categories", "esg_pillars", "topics",
        "regulations", "markets", "is_eu", "is_us", "primary_market",
        "relevance_score", "duplicate_count", "feed_url", "event_ts",
        "ingest_date",
    ).orderBy(F.col("relevance_score").desc(), F.col("event_ts").desc())


def _to_driver_rows(df):
    """Collect to plain dicts using the local engine's column names."""
    renamed = (
        df.withColumn("published_at", F.col("published_ts").cast("string"))
        .withColumn("fetched_at", F.col("fetched_ts").cast("string"))
        .withColumn("event_ts", F.col("event_ts").cast("string"))
        .withColumn("published_date", F.col("published_date").cast("string"))
        .withColumn("ingest_date", F.col("ingest_date").cast("string"))
    )
    return [row.asDict(recursive=True) for row in renamed.collect()]


def write_curated(df, out_path, mode="overwrite", driver_io=None):
    """Write partitioned Parquet, or fall back to driver-side writes."""
    if driver_io is None:
        driver_io = not hadoop_native_available()

    if not driver_io:
        (
            df.write
            .mode(mode)
            .partitionBy("ingest_date", "primary_market")
            .parquet(str(out_path))
        )
        return

    from .local_job import build_rollups as build_rollups_local
    from .local_job import write_curated as write_curated_local
    from .local_job import write_rollups as write_rollups_local

    log.info("Hadoop native libs unavailable - writing via the driver")
    rows = _to_driver_rows(df)
    write_curated_local(rows, out_path)
    write_rollups_local(build_rollups_local(rows), out_path)


def build_rollups(df):
    """Small aggregates for dashboards / digest emails."""
    by_market_day = (
        df.groupBy("primary_market", "published_date")
        .agg(
            F.count("*").alias("article_count"),
            F.round(F.avg("relevance_score"), 3).alias("avg_relevance"),
            F.countDistinct("source_id").alias("source_count"),
        )
        .orderBy("published_date", "primary_market")
    )

    by_regulation = (
        df.withColumn("regulation", F.explode("regulations"))
        .groupBy("regulation", "primary_market")
        .agg(F.count("*").alias("article_count"))
        .orderBy(F.col("article_count").desc())
    )

    by_topic = (
        df.withColumn("topic", F.explode("topics"))
        .groupBy("topic")
        .agg(
            F.count("*").alias("article_count"),
            F.round(F.avg("relevance_score"), 3).alias("avg_relevance"),
        )
        .orderBy(F.col("article_count").desc())
    )

    by_pillar = (
        df.withColumn("pillar", F.explode("esg_pillars"))
        .groupBy("pillar", "primary_market")
        .agg(F.count("*").alias("article_count"))
        .orderBy("pillar", "primary_market")
    )

    return {
        "by_market_day": by_market_day,
        "by_regulation": by_regulation,
        "by_topic": by_topic,
        "by_pillar": by_pillar,
    }


def run(raw_path, curated_path, days=7, min_relevance=0.15,
        master=DEFAULT_MASTER, write_rollups=True):
    """Full Spark leg. Returns (row_count, curated_path)."""
    spark = get_spark(master=master or DEFAULT_MASTER)
    spark.sparkContext.setLogLevel("WARN")
    driver_io = not hadoop_native_available()
    try:
        raw = read_raw(spark, raw_path, driver_io=driver_io)
        curated = transform(raw, days=days, min_relevance=min_relevance).cache()
        count = curated.count()
        log.info("curated rows: %d", count)

        write_curated(curated, curated_path, driver_io=driver_io)

        # The driver-side path already emits rollups.json alongside the data.
        if write_rollups and not driver_io:
            for name, agg in build_rollups(curated).items():
                agg.write.mode("overwrite").parquet(
                    str(curated_path).rstrip("/\\") + f"_rollup_{name}"
                )

        return count, str(curated_path)
    finally:
        spark.stop()
