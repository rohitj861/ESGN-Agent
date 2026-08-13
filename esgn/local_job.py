"""Pure-Python / pandas transform mirroring `spark_job.transform`.

Used when PySpark or a JVM is unavailable. Same enrichment, same window,
same dedupe precedence -- so the curated output is interchangeable.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .enrich import enrich, is_esg_relevant
from .fetch import parse_datetime

log = logging.getLogger(__name__)

try:  # optional
    import pandas as _pd
except ImportError:  # pragma: no cover
    _pd = None


def read_raw(raw_path):
    """Read every JSONL file under the raw landing zone."""
    rows = []
    for path in sorted(Path(raw_path).rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("bad JSON line in %s", path)
    return rows


def _event_ts(row):
    return (
        parse_datetime(row.get("published_at"))
        or parse_datetime(row.get("fetched_at"))
        or datetime.now(timezone.utc)
    )


def _primary_market(row):
    if row["is_eu"] and row["is_us"]:
        return "EU_US"
    if row["is_eu"]:
        return "EU"
    if row["is_us"]:
        return "US"
    return "GLOBAL"


def transform(rows, days=7, min_relevance=0.15, now=None):
    """Enrich -> window filter -> relevance gate -> dedupe. Returns list of dicts."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    horizon = now + timedelta(hours=12)

    enriched = []
    for row in rows:
        if not row.get("title") or not row.get("url"):
            continue

        item = enrich(row)
        published = parse_datetime(item.get("published_at"))
        if published is not None and not (cutoff <= published <= horizon):
            continue
        if not is_esg_relevant(item, threshold=min_relevance):
            continue

        ts = _event_ts(item)
        item["event_ts"] = ts.isoformat()
        item["published_date"] = ts.date().isoformat()
        item["primary_market"] = _primary_market(item)
        item["ingest_date"] = now.date().isoformat()
        enriched.append(item)

    # ---- dedupe: same precedence as the Spark job ------------------------
    def precedence(item):
        return (
            0 if item.get("source_type") == "regulator" else 1,
            -item.get("relevance_score", 0.0),
            item["event_ts"],
        )

    by_url = {}
    for item in sorted(enriched, key=precedence):
        by_url.setdefault(item["dedupe_key"], item)
    deduped = list(by_url.values())

    title_counts = Counter(
        (i["title_key"], i["published_date"]) for i in deduped
    )
    by_title = {}
    survivors = []
    for item in sorted(deduped, key=precedence):
        key = (item["title_key"], item["published_date"])
        item["duplicate_count"] = title_counts[key]
        if len(item["title_key"]) < 15 or key not in by_title:
            by_title[key] = item
            survivors.append(item)

    # Newest first within each relevance tier (stable sort, applied in reverse).
    survivors.sort(key=lambda i: i["event_ts"], reverse=True)
    survivors.sort(key=lambda i: -i.get("relevance_score", 0.0))
    return survivors


CURATED_COLUMNS = [
    "article_id", "title", "url", "canonical_url", "summary", "author",
    "source_id", "source_name", "source_type", "source_market", "language",
    "published_at", "published_date", "fetched_at", "has_publish_date",
    "categories", "esg_pillars", "topics", "regulations", "markets",
    "is_eu", "is_us", "primary_market", "relevance_score",
    "duplicate_count", "feed_url", "event_ts", "ingest_date",
]


def write_curated(rows, out_dir):
    """Write Parquet when pandas+pyarrow are present; always write JSONL + CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    jsonl_path = out_dir / "articles.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(
                {c: row.get(c) for c in CURATED_COLUMNS}, ensure_ascii=False
            ) + "\n")
    written.append(jsonl_path)

    if _pd is not None:
        df = _pd.DataFrame([{c: r.get(c) for c in CURATED_COLUMNS} for r in rows])
        csv_df = df.copy()
        for col in ("categories", "esg_pillars", "topics", "regulations", "markets"):
            csv_df[col] = csv_df[col].apply(
                lambda v: "|".join(v) if isinstance(v, list) else ""
            )
        csv_path = out_dir / "articles.csv"
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        written.append(csv_path)

        try:
            parquet_path = out_dir / "articles.parquet"
            df.to_parquet(parquet_path, index=False)
            written.append(parquet_path)
        except (ImportError, ValueError) as exc:
            log.info("parquet skipped (%s) -- install pyarrow to enable", exc)

    return written


def build_rollups(rows):
    """Same aggregates as the Spark job, as plain dicts."""
    by_market_day = defaultdict(lambda: {"article_count": 0, "score_sum": 0.0,
                                         "sources": set()})
    by_regulation = Counter()
    by_topic = defaultdict(lambda: {"article_count": 0, "score_sum": 0.0})
    by_pillar = Counter()
    by_source = Counter()

    for row in rows:
        key = (row["primary_market"], row["published_date"])
        bucket = by_market_day[key]
        bucket["article_count"] += 1
        bucket["score_sum"] += row.get("relevance_score", 0.0)
        bucket["sources"].add(row["source_id"])

        for reg in row.get("regulations") or []:
            by_regulation[(reg, row["primary_market"])] += 1
        for topic in row.get("topics") or []:
            by_topic[topic]["article_count"] += 1
            by_topic[topic]["score_sum"] += row.get("relevance_score", 0.0)
        for pillar in row.get("esg_pillars") or []:
            by_pillar[(pillar, row["primary_market"])] += 1
        by_source[row["source_name"]] += 1

    return {
        "by_market_day": [
            {
                "primary_market": m, "published_date": d,
                "article_count": v["article_count"],
                "avg_relevance": round(v["score_sum"] / v["article_count"], 3),
                "source_count": len(v["sources"]),
            }
            for (m, d), v in sorted(by_market_day.items(), key=lambda kv: kv[0][1])
        ],
        "by_regulation": [
            {"regulation": r, "primary_market": m, "article_count": c}
            for (r, m), c in by_regulation.most_common()
        ],
        "by_topic": [
            {"topic": t, "article_count": v["article_count"],
             "avg_relevance": round(v["score_sum"] / v["article_count"], 3)}
            for t, v in sorted(by_topic.items(),
                               key=lambda kv: -kv[1]["article_count"])
        ],
        "by_pillar": [
            {"pillar": p, "primary_market": m, "article_count": c}
            for (p, m), c in sorted(by_pillar.items())
        ],
        "by_source": [
            {"source_name": s, "article_count": c} for s, c in by_source.most_common()
        ],
    }


def write_rollups(rollups, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rollups.json"
    path.write_text(json.dumps(rollups, indent=2), encoding="utf-8")
    return path


def run(raw_path, curated_path, days=7, min_relevance=0.15):
    rows = read_raw(raw_path)
    curated = transform(rows, days=days, min_relevance=min_relevance)
    write_curated(curated, curated_path)
    write_rollups(build_rollups(curated), curated_path)
    return len(curated), str(curated_path)
