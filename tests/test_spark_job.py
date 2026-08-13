"""Spark transform tests.

Builds DataFrames in memory rather than reading files, so the transform logic
is exercised without Hadoop's native Windows filesystem layer (winutils.exe /
hadoop.dll). Skips cleanly when PySpark or a JVM is unavailable.

    python tests/test_spark_job.py
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esgn.pipeline import spark_available

if not spark_available():
    raise unittest.SkipTest("PySpark or JVM unavailable")

from pyspark.sql import functions as F  # noqa: E402

from esgn import spark_job  # noqa: E402

NOW = datetime.now(timezone.utc)


def article(**kw):
    base = {
        "article_id": "a1",
        "title": "Net zero carbon emissions plan under CSRD",
        "url": "https://a.example/1",
        "canonical_url": "https://a.example/1",
        "summary": "Climate and emission disclosure rules tighten.",
        "author": None,
        "published_at": (NOW - timedelta(days=1)).isoformat(),
        "published_raw": None,
        "categories": [],
        "source_id": "s1",
        "source_name": "S1",
        "source_market": "EU",
        "source_type": "media",
        "language": "en",
        "feed_url": "https://a.example/rss",
        "fetched_at": NOW.isoformat(),
    }
    base.update(kw)
    return base


class SparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = spark_job.get_spark(
            app_name="esgn-tests", master="local[2]", shuffle_partitions=2
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def df(self, rows):
        return self.spark.createDataFrame(rows, schema=spark_job.RAW_SCHEMA)

    # -- tests ------------------------------------------------------------
    def test_enrichment_columns_populated(self):
        out = spark_job.transform(self.df([article()]), days=7).collect()
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertIn("E", row["esg_pillars"])
        self.assertIn("CSRD", row["regulations"])
        self.assertIn("EU", row["markets"])
        self.assertEqual(row["primary_market"], "EU")
        self.assertGreater(row["relevance_score"], 0.0)
        self.assertLessEqual(row["relevance_score"], 1.0)

    def test_dedupe_by_canonical_url(self):
        rows = [article(source_id="s1"),
                article(article_id="a2", source_id="s2")]
        out = spark_job.transform(self.df(rows), days=7).collect()
        self.assertEqual(len(out), 1, "same canonical URL must collapse")

    def test_regulator_wins_dedupe(self):
        rows = [
            article(article_id="a1", source_id="media", source_type="media"),
            article(article_id="a2", source_id="reg", source_type="regulator"),
        ]
        out = spark_job.transform(self.df(rows), days=7).collect()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["source_id"], "reg",
                         "primary sources outrank trade press")

    def test_out_of_window_dropped(self):
        old = (NOW - timedelta(days=40)).isoformat()
        out = spark_job.transform(self.df([article(published_at=old)]),
                                  days=7).collect()
        self.assertEqual(out, [])

    def test_irrelevant_dropped(self):
        rows = [article(title="Quarterly earnings beat expectations",
                        summary="Revenue rose 4% on strong demand.")]
        out = spark_job.transform(self.df(rows), days=7).collect()
        self.assertEqual(out, [])

    def test_undated_item_retained(self):
        rows = [article(published_at=None)]
        out = spark_job.transform(self.df(rows), days=7).collect()
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["has_publish_date"])

    def test_dual_market_classification(self):
        rows = [article(
            title="European Commission and SEC diverge on climate disclosure",
            summary="Brussels and Washington take different paths on CSRD.",
            source_market="GLOBAL",
        )]
        out = spark_job.transform(self.df(rows), days=7).collect()
        self.assertEqual(out[0]["primary_market"], "EU_US")

    def test_acronym_case_sensitivity(self):
        """Lowercase 'us' must not be read as the United States."""
        rows = [article(
            title="Biodiversity study tells us more about deforestation",
            summary="Researchers tracked carbon emission trends in forests.",
            source_market="GLOBAL",
        )]
        out = spark_job.transform(self.df(rows), days=7).collect()
        self.assertNotIn("US", out[0]["markets"])

    def test_rollups(self):
        rows = [article(article_id=f"a{i}", canonical_url=f"https://a.example/{i}",
                        url=f"https://a.example/{i}",
                        title=f"Net zero carbon plan under CSRD number {i}")
                for i in range(5)]
        df = spark_job.transform(self.df(rows), days=7)
        rollups = spark_job.build_rollups(df)
        self.assertEqual(
            sum(r["article_count"] for r in rollups["by_market_day"].collect()), 5
        )
        regs = {r["regulation"] for r in rollups["by_regulation"].collect()}
        self.assertIn("CSRD", regs)
        self.assertTrue(rollups["by_topic"].count() > 0)
        self.assertTrue(rollups["by_pillar"].count() > 0)

    def test_parity_with_local_engine(self):
        """Spark and local engines must agree on which articles survive."""
        from esgn import local_job

        rows = [
            article(article_id="a1", url="https://a.example/1",
                    canonical_url="https://a.example/1"),
            article(article_id="a2", url="https://a.example/2",
                    canonical_url="https://a.example/2",
                    title="SEC climate rule challenged in Washington court",
                    source_market="US"),
            article(article_id="a3", url="https://a.example/3",
                    canonical_url="https://a.example/3",
                    title="Quarterly earnings beat expectations",
                    summary="Revenue rose 4%."),
            article(article_id="a4", url="https://a.example/1",
                    canonical_url="https://a.example/1"),  # dupe of a1
        ]

        spark_ids = {r["canonical_url"] for r
                     in spark_job.transform(self.df(rows), days=7).collect()}
        local_ids = {r["canonical_url"] for r
                     in local_job.transform([dict(r) for r in rows], days=7)}
        self.assertEqual(spark_ids, local_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
