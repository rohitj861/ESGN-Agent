"""Tests for the AI analyst layer.

Covers selection, prompt assembly and the briefing contract checker. No API
key or network access required -- the model call itself is not exercised.

    python tests/test_analyst.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esgn import analyst

ROWS = [
    {
        "title": "EU adopts CSRD omnibus", "source_name": "edie",
        "published_date": "2026-08-13", "canonical_url": "https://a.example/1",
        "primary_market": "EU", "esg_pillars": ["E", "G"],
        "topics": ["regulation_policy"], "regulations": ["CSRD"],
        "summary": "Reporting rules change.", "relevance_score": 0.80,
        "markets": ["EU"],
    },
    {
        "title": "Green bond issuance record", "source_name": "ESG Today",
        "published_date": "2026-08-12", "canonical_url": "https://a.example/2",
        "primary_market": "US", "esg_pillars": ["E"],
        "topics": ["sustainable_finance"], "regulations": [],
        "summary": "Issuance grew.", "relevance_score": 0.60,
        "markets": ["US"],
    },
    {
        "title": "Workforce diversity rule", "source_name": "RI",
        "published_date": "2026-08-11", "canonical_url": "https://a.example/3",
        "primary_market": "EU", "esg_pillars": ["S"], "topics": [],
        "regulations": [], "summary": "New rule.", "relevance_score": 0.40,
        "markets": ["EU"],
    },
]


def good_briefing(rows):
    """A briefing that satisfies every contract rule."""
    stories = "\n".join(
        f"### {r['title']}\n"
        f"- **What happened**: x\n- **Why it matters**: y\n"
        f"- **Who may be affected**: z\n- **What to watch next**: w\n"
        f"Source: {r['source_name']}, {r['published_date']} - {r['canonical_url']}\n"
        for r in rows
    )
    return (
        "## Executive Summary\n"
        + "".join(f"- Development {i}.\n" for i in range(1, 6))
        + "\n## Environment\n" + stories
        + "\n## Terms Explained\n**CSRD** - EU sustainability reporting rule.\n"
    )


class TestSelection(unittest.TestCase):
    def test_top_n_by_relevance(self):
        out = analyst.select_top(ROWS, top_n=2)
        self.assertEqual([r["relevance_score"] for r in out], [0.80, 0.60])

    def test_market_filter(self):
        out = analyst.select_top(ROWS, top_n=10, markets=["US"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["primary_market"], "US")

    def test_each_article_filed_under_exactly_one_section(self):
        sections = analyst.group_by_section(ROWS)
        placed = [r for members in sections.values() for r in members]
        self.assertEqual(len(placed), len(ROWS),
                         "an article must not appear in two sections")

    def test_section_routing(self):
        # Finance topic wins over pillars; otherwise G, then S, then E.
        self.assertEqual(analyst.primary_section(ROWS[1]), "Sustainable Finance")
        self.assertEqual(analyst.primary_section(ROWS[0]), "Governance")
        self.assertEqual(analyst.primary_section(ROWS[2]), "Social")
        self.assertEqual(
            analyst.primary_section({"esg_pillars": ["E"], "topics": []}),
            "Environment",
        )


class TestAssembly(unittest.TestCase):
    """The assembler, not the model, guarantees coverage and citations."""

    ANALYSES = {
        i: {"what_happened": "w", "why_it_matters": "y",
            "who_may_be_affected": "a", "what_to_watch_next": "n"}
        for i in range(1, len(ROWS) + 1)
    }
    BULLETS = [f"Development {i}." for i in range(1, 6)]

    def test_every_article_appears_once(self):
        md = analyst.assemble_briefing(ROWS, self.ANALYSES, self.BULLETS)
        self.assertEqual(md.count("### "), len(ROWS))
        for row in ROWS:
            self.assertEqual(md.count(row["canonical_url"]), 1)

    def test_source_line_is_written_by_code(self):
        """URLs and dates must survive even if the model returns nothing."""
        md = analyst.assemble_briefing(ROWS, {}, self.BULLETS)
        for row in ROWS:
            self.assertIn(
                f"Source: {row['source_name']}, {row['published_date']} - "
                f"{row['canonical_url']}",
                md,
            )

    def test_missing_analysis_degrades_gracefully(self):
        md = analyst.assemble_briefing(ROWS, {}, self.BULLETS)
        self.assertIn("Not stated in the source material.", md)
        self.assertIn("**What happened**", md)

    def test_assembled_output_passes_its_own_contract(self):
        md = analyst.assemble_briefing(
            ROWS, self.ANALYSES, self.BULLETS,
            glossary=[("CSRD", "EU sustainability reporting rule")],
        )
        self.assertEqual(analyst.validate_briefing(md, ROWS), [])

    def test_empty_sections_omitted(self):
        only_env = [{**ROWS[1], "topics": [], "esg_pillars": ["E"]}]
        md = analyst.assemble_briefing(only_env, {}, self.BULLETS)
        self.assertIn("## Environment", md)
        self.assertNotIn("## Social", md)
        self.assertNotIn("## Governance", md)


class TestContractValidation(unittest.TestCase):
    def test_clean_briefing_passes(self):
        self.assertEqual(analyst.validate_briefing(good_briefing(ROWS), ROWS), [])

    def test_missing_executive_summary_caught(self):
        bad = good_briefing(ROWS).replace("## Executive Summary", "## Overview")
        self.assertTrue(any("Executive Summary" in p
                            for p in analyst.validate_briefing(bad, ROWS)))

    def test_dropped_url_caught(self):
        bad = good_briefing(ROWS).replace("https://a.example/2", "")
        problems = analyst.validate_briefing(bad, ROWS)
        self.assertTrue(any("URL" in p for p in problems),
                        "a dropped source URL must be reported")

    def test_too_few_summary_bullets_caught(self):
        bad = (
            "## Executive Summary\n- One.\n- Two.\n\n## Environment\n"
            + "\n## Terms Explained\n**X** - y.\n"
        )
        self.assertTrue(any("bullets" in p
                            for p in analyst.validate_briefing(bad, ROWS)))

    def test_missing_four_point_structure_caught(self):
        bad = good_briefing(ROWS).replace("What to watch next", "Next")
        self.assertTrue(any("What to watch next" in p
                            for p in analyst.validate_briefing(bad, ROWS)))

    def test_missing_glossary_caught(self):
        bad = good_briefing(ROWS).replace("## Terms Explained", "## Notes")
        self.assertTrue(any("Terms Explained" in p
                            for p in analyst.validate_briefing(bad, ROWS)))


class TestEnvLoading(unittest.TestCase):
    def test_existing_env_var_wins_over_file(self):
        import os
        tmp = Path(__file__).parent / "_tmp.env"
        tmp.write_text('ESGN_TEST_KEY="from_file"\n', encoding="utf-8")
        os.environ["ESGN_TEST_KEY"] = "from_shell"
        try:
            analyst.load_env(tmp)
            self.assertEqual(os.environ["ESGN_TEST_KEY"], "from_shell")
        finally:
            os.environ.pop("ESGN_TEST_KEY", None)
            tmp.unlink()

    def test_blank_key_reads_as_absent(self):
        import os
        tmp = Path(__file__).parent / "_tmp2.env"
        tmp.write_text('ESGN_BLANK=""\n', encoding="utf-8")
        try:
            analyst.load_env(tmp)
            self.assertIsNone(os.environ.get("ESGN_BLANK"))
        finally:
            os.environ.pop("ESGN_BLANK", None)
            tmp.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
