"""Unit tests for the parsing / enrichment / dedupe layers.

    python -m pytest tests/ -q          (if pytest is installed)
    python tests/test_pipeline.py       (stdlib unittest, no deps)
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esgn import enrich as enrich_mod
from esgn import fetch, local_job

FEED = {
    "id": "test_feed", "name": "Test Feed", "url": "https://example.com/rss",
    "market": "EU", "source_type": "media", "language": "en",
}

RSS_2 = """<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Test</title>
    <item>
      <title>EU adopts CSRD simplification omnibus package</title>
      <link>https://example.com/a?utm_source=rss&amp;id=7</link>
      <description>&lt;p&gt;The European Commission agreed new &lt;b&gt;net zero&lt;/b&gt;
        reporting rules.&lt;/p&gt;</description>
      <pubDate>Wed, 06 Aug 2025 14:03:11 +0000</pubDate>
      <dc:creator>Jane Doe</dc:creator>
      <category>Regulation</category>
    </item>
    <item>
      <title>No link item</title>
      <description>should be skipped</description>
    </item>
  </channel>
</rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test</title>
  <entry>
    <title>SEC climate disclosure rule faces court challenge</title>
    <link rel="alternate" href="https://example.org/b/"/>
    <summary>Litigation over the SEC climate rule continues in Washington.</summary>
    <published>2025-08-06T14:03:11Z</published>
    <author><name>John Roe</name></author>
    <category term="Governance"/>
  </entry>
</feed>"""

RDF = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://purl.org/rss/1.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <item rdf:about="https://example.net/c">
    <title>Carbon market volumes rise</title>
    <link>https://example.net/c</link>
    <description>Carbon credit trading grew.</description>
    <dc:date>2025-08-05T09:00:00+02:00</dc:date>
  </item>
</rdf:RDF>"""

ESMA_STYLE = """<?xml version="1.0"?>
<rss version="2.0"><channel><item>
  <title>ESMA consults on sustainability disclosure</title>
  <link>https://esma.example/x</link>
  <description>&lt;time datetime="2026-08-03T10:56:46+02:00"&gt;03 August 2026&lt;/time&gt;
    Guidance on SFDR reporting.</description>
</item></channel></rss>"""


class TestParsing(unittest.TestCase):
    def test_rss2(self):
        items = fetch.parse_feed(RSS_2.encode(), FEED)
        self.assertEqual(len(items), 1, "item without a link must be dropped")
        item = items[0]
        self.assertEqual(item["author"], "Jane Doe")
        self.assertEqual(item["categories"], ["Regulation"])
        self.assertIn("net zero", item["summary"])
        self.assertNotIn("<b>", item["summary"], "HTML must be stripped")
        self.assertTrue(item["published_at"].startswith("2025-08-06T14:03:11"))

    def test_atom(self):
        items = fetch.parse_feed(ATOM.encode(), FEED)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.org/b/")
        self.assertEqual(items[0]["author"], "John Roe")
        self.assertEqual(items[0]["categories"], ["Governance"])

    def test_rdf_rss1(self):
        items = fetch.parse_feed(RDF.encode(), FEED)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["published_at"].startswith("2025-08-05T07:00"),
                        "dc:date +02:00 must normalise to UTC")

    def test_date_recovered_from_html_body(self):
        items = fetch.parse_feed(ESMA_STYLE.encode(), FEED)
        self.assertTrue(items[0]["published_at"].startswith("2026-08-03T08:56"))

    def test_malformed_xml_returns_empty(self):
        self.assertEqual(fetch.parse_feed(b"<rss><chan", FEED), [])

    def test_canonicalize_url(self):
        self.assertEqual(
            fetch.canonicalize_url(
                "HTTPS://WWW.Example.com/Path/?utm_source=rss&id=7&fbclid=z#frag"
            ),
            "https://example.com/Path?id=7",
        )
        self.assertEqual(fetch.canonicalize_url(None), None)

    def test_parse_datetime_formats(self):
        for value in ("Wed, 06 Aug 2025 14:03:11 +0000",
                      "2025-08-06T14:03:11Z",
                      "2025-08-06T14:03:11+00:00",
                      "2025-08-06"):
            self.assertIsNotNone(fetch.parse_datetime(value), value)
        self.assertIsNone(fetch.parse_datetime("not a date"))

    def test_window_filter(self):
        now = datetime(2025, 8, 10, tzinfo=timezone.utc)
        recent = {"published_at": (now - timedelta(days=3)).isoformat()}
        old = {"published_at": (now - timedelta(days=30)).isoformat()}
        undated = {"published_at": None}
        self.assertTrue(fetch.within_window(recent, 7, now))
        self.assertFalse(fetch.within_window(old, 7, now))
        self.assertTrue(fetch.within_window(undated, 7, now),
                        "undated items are kept and flagged downstream")


GOOGLE_NEWS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Google News</title>
  <item>
    <title>Drought squeezes livestock sector across the EU - euractiv.com</title>
    <link>https://news.google.com/rss/articles/CBMiXxyz</link>
    <description>Farmers face water restrictions.</description>
    <pubDate>Wed, 13 Aug 2026 15:19:42 GMT</pubDate>
    <source url="https://www.euractiv.com">euractiv.com</source>
  </item>
</channel></rss>"""


class TestGoogleNewsRelay(unittest.TestCase):
    FEED = dict(FEED, id="euractiv", name="Euractiv (via Google News)",
                via="google_news")

    def test_publisher_suffix_stripped_from_title(self):
        items = fetch.parse_feed(GOOGLE_NEWS.encode(), self.FEED)
        self.assertEqual(
            items[0]["title"],
            "Drought squeezes livestock sector across the EU",
            "Google News' ' - publisher' tail must not leak into title_key",
        )

    def test_real_publisher_recorded(self):
        items = fetch.parse_feed(GOOGLE_NEWS.encode(), self.FEED)
        self.assertEqual(items[0]["source_name"], "euractiv.com")
        self.assertEqual(items[0]["source_id"], "euractiv")

    def test_direct_feeds_keep_dashes_in_titles(self):
        """The suffix stripper must only run for google_news feeds."""
        rss = RSS_2.replace(
            "EU adopts CSRD simplification omnibus package",
            "Net-zero - what it really means for asset owners",
        )
        items = fetch.parse_feed(rss.encode(), FEED)
        self.assertEqual(items[0]["title"],
                         "Net-zero - what it really means for asset owners")

    def test_suffix_stripper_leaves_long_tails_alone(self):
        title = "Net-zero - what it really means for asset owners"
        self.assertEqual(fetch.strip_publisher_suffix(title), title)


class TestEnrichment(unittest.TestCase):
    def test_pillars_and_regulations(self):
        out = enrich_mod.enrich({
            "title": "EU adopts CSRD omnibus cutting reporting burden",
            "summary": "Board governance and carbon emission disclosure rules.",
            "categories": [], "source_market": "EU", "source_type": "regulator",
        })
        self.assertIn("E", out["esg_pillars"])
        self.assertIn("G", out["esg_pillars"])
        self.assertIn("CSRD", out["regulations"])
        self.assertIn("Omnibus", out["regulations"])
        self.assertIn("EU", out["markets"])

    def test_us_market_detection(self):
        out = enrich_mod.enrich({
            "title": "SEC climate rule challenged in Washington",
            "summary": "California SB 253 also faces scrutiny.",
            "categories": [], "source_market": "GLOBAL", "source_type": "media",
        })
        self.assertIn("US", out["markets"])
        self.assertIn("SEC_Climate_Rule", out["regulations"])

    def test_short_token_false_positives(self):
        """'tells us' / 'a doe' must not trigger US attribution."""
        out = enrich_mod.enrich({
            "title": "Biodiversity study tells us more about deforestation",
            "summary": "Researchers tracked a doe through the forest.",
            "categories": [], "source_market": "GLOBAL", "source_type": "media",
        })
        self.assertNotIn("US", out["markets"])

    def test_dual_market(self):
        out = enrich_mod.enrich({
            "title": "European Commission and SEC diverge on climate disclosure",
            "summary": "Brussels and Washington take different paths.",
            "categories": [], "source_market": "GLOBAL", "source_type": "media",
        })
        self.assertEqual(set(out["markets"]), {"EU", "US"})

    def test_relevance_bounds(self):
        for article in ({"title": "Local bakery opens", "summary": "Bread.",
                         "categories": [], "source_market": "US",
                         "source_type": "media"},
                        {"title": "Net zero carbon emission climate ESG "
                                  "sustainability CSRD SFDR taxonomy",
                         "summary": "Green renewable decarbonisation.",
                         "categories": [], "source_market": "EU",
                         "source_type": "regulator"}):
            score = enrich_mod.enrich(article)["relevance_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_substring_false_positives_rejected(self):
        """Real headlines that leaked in when core terms were substrings.

        "green light", "Greenland" and a political "transition" are not ESG.
        """
        for title, summary in [
            ("EU welcomes Israeli green light to extend Gaza border mission",
             "The border mission is extended."),
            ("Greenland tells US oil firm to forget drilling this year",
             "Drilling paused for the season."),
            ("Hungary's new president for the post-Orban transition",
             "A political transition begins."),
        ]:
            out = enrich_mod.enrich({
                "title": title, "summary": summary, "categories": [],
                "source_market": "EU", "source_type": "media",
            })
            self.assertFalse(enrich_mod.is_esg_relevant(out),
                             f"should not be ESG-relevant: {title}")

    def test_genuine_esg_still_kept(self):
        for title, summary in [
            ("Green bond issuance hits record in energy transition push",
             "Renewable financing grew."),
            ("EU adopts CSRD omnibus cutting reporting burden",
             "Sustainability reporting rules change."),
            ("BlackRock downgrades sustainable infrastructure fund",
             "ESG fund flows shift."),
        ]:
            out = enrich_mod.enrich({
                "title": title, "summary": summary, "categories": [],
                "source_market": "EU", "source_type": "media",
            })
            self.assertTrue(enrich_mod.is_esg_relevant(out),
                            f"should be ESG-relevant: {title}")

    def test_irrelevant_filtered_out(self):
        out = enrich_mod.enrich({
            "title": "Quarterly earnings beat expectations",
            "summary": "Revenue rose 4% on strong demand.",
            "categories": [], "source_market": "US", "source_type": "media",
        })
        self.assertFalse(enrich_mod.is_esg_relevant(out))

    def test_regulator_with_regulation_always_kept(self):
        out = enrich_mod.enrich({
            "title": "Commission adopts CBAM implementing act",
            "summary": "", "categories": [], "source_market": "EU",
            "source_type": "regulator",
        })
        self.assertTrue(enrich_mod.is_esg_relevant(out))


class TestTransform(unittest.TestCase):
    def _article(self, **kw):
        now = datetime.now(timezone.utc)
        base = {
            "article_id": "x", "title": "Net zero carbon emissions plan",
            "url": "https://a.example/1", "canonical_url": "https://a.example/1",
            "summary": "Climate and emission disclosure under CSRD.",
            "categories": [], "source_id": "s1", "source_name": "S1",
            "source_market": "EU", "source_type": "media", "language": "en",
            "feed_url": "https://a.example/rss",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "fetched_at": now.isoformat(),
        }
        base.update(kw)
        return base

    def test_dedupe_by_canonical_url(self):
        rows = [self._article(source_id="s1"),
                self._article(source_id="s2", article_id="y")]
        out = local_job.transform(rows, days=7)
        self.assertEqual(len(out), 1, "same canonical URL collapses to one row")

    def test_regulator_wins_dedupe(self):
        rows = [
            self._article(source_id="media", source_type="media"),
            self._article(source_id="reg", source_type="regulator",
                          article_id="y"),
        ]
        out = local_job.transform(rows, days=7)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["source_id"], "reg",
                         "primary sources outrank trade press")

    def test_out_of_window_dropped(self):
        old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        out = local_job.transform([self._article(published_at=old)], days=7)
        self.assertEqual(out, [])

    def test_rollups(self):
        out = local_job.transform([self._article()], days=7)
        rollups = local_job.build_rollups(out)
        self.assertEqual(rollups["by_market_day"][0]["article_count"], 1)
        self.assertTrue(any(r["regulation"] == "CSRD"
                            for r in rollups["by_regulation"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
