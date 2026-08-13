"""AI ESG news analyst.

Takes the curated dataset, feeds the top-relevance articles to an LLM, and
writes a professional weekly briefing as Markdown.

Structured generation, not single-shot. Asked to write all 20 stories in one
response, gpt-4o-mini reliably stopped around 18 and dropped the rest however
firmly the prompt insisted, and rewrote formatting when asked to fix it. So
the model is only ever asked for *analysis* -- four points per article, in
JSON, in small batches -- while this module drives the coverage loop and emits
every heading, section and `Source:` line itself.

The upshot is that the compliance-critical parts (every article covered, every
URL and publication date preserved verbatim) are guaranteed by construction
rather than by the model's cooperation. `validate_briefing` still re-checks
the finished document as a safety net.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_N = 20

PILLAR_SECTIONS = [
    ("Environment", "E"),
    ("Social", "S"),
    ("Governance", "G"),
    ("Sustainable Finance", None),   # topic-driven, not a pillar
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_env(path=None):
    """Minimal .env loader (no python-dotenv dependency).

    Existing environment variables win, so `setx` / shell exports override the
    file rather than the other way round.
    """
    path = Path(path or PROJECT_ROOT / ".env")
    if not path.exists():
        return {}

    loaded = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def get_api_key():
    load_env()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return key or None


# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------
def load_curated(curated_root=None):
    path = Path(curated_root or PROJECT_ROOT / "data" / "curated") / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No curated data at {path}. Run: python run_esgn.py run"
        )
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_top(rows, top_n=DEFAULT_TOP_N, markets=None):
    """Highest-relevance articles, optionally restricted to given markets."""
    if markets:
        wanted = {m.upper() for m in markets}
        rows = [r for r in rows if wanted & set(r.get("markets") or [])]
    ranked = sorted(
        rows,
        key=lambda r: (-(r.get("relevance_score") or 0.0),
                       r.get("published_date") or ""),
    )
    return ranked[:top_n]


def primary_section(row):
    """The single section a story is filed under.

    A story often carries several pillars -- a CSRD item is both Environment
    and Governance -- but a briefing that prints it twice reads as sloppy, so
    each article gets exactly one home. Money first (that is what the
    Sustainable Finance section exists for), then the rarer S/G signals, with
    Environment as the default since nearly every ESG story carries an E.
    """
    pillars = set(row.get("esg_pillars") or [])
    topics = set(row.get("topics") or [])

    if {"sustainable_finance", "carbon_markets"} & topics:
        return "Sustainable Finance"
    if "G" in pillars:
        return "Governance"
    if "S" in pillars:
        return "Social"
    return "Environment"


def group_by_section(rows):
    """Bucket articles into the four briefing sections, one section each."""
    sections = defaultdict(list)
    for row in rows:
        sections[primary_section(row)].append(row)
    return sections


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior ESG and sustainability analyst writing the weekly briefing \
for an institutional investment team covering the EU and US markets.

Your readers are portfolio managers and compliance officers. They are \
financially literate but are NOT regulatory specialists, so every piece of \
jargon must be explained in plain language the first time it appears.

Write in precise, neutral, factual prose. No marketing language, no hype, no \
filler. Never invent facts, figures, dates or URLs: use only what the source \
data provides. If something is unclear from the source material, say so \
rather than guessing."""

BATCH_SIZE = 5

ANALYSIS_SPEC = """\
For each numbered article you receive, write the four analytical points an ESG
investment desk needs. Return STRICT JSON, no prose outside it:

{"stories": [{"id": <article number>,
              "what_happened": "the development itself, one or two sentences",
              "why_it_matters": "significance for ESG investors",
              "who_may_be_affected": "sectors, companies, regions or asset classes",
              "what_to_watch_next": "the concrete next milestone, decision or date"}]}

Rules:
- Return one object for EVERY article you were given. Never skip one.
- Base every statement on the supplied article data. Do not invent figures,
  dates or events. If the source is thin, say what is known and no more.
- Explain jargon in plain language in parentheses at first use, e.g. "CSRD
  (the EU rule requiring large companies to publish audited sustainability
  reports)".
- No markdown headings, no bullet characters, no URLs in these fields."""

SUMMARY_SPEC = """\
Write the executive summary for a weekly ESG briefing. Return STRICT JSON:

{"bullets": ["...", "..."]}

Rules:
- Between 5 and 8 bullets TOTAL, regardless of how many stories exist.
- Each is ONE sentence leading with the concrete development.
- Together they must capture the week: prioritise regulation, large corporate
  moves, capital flows and governance controversies over minor items.
- A reader who stops after the summary should still understand the week.
- Plain language. Explain any acronym you use."""

GLOSSARY_SPEC = """\
Build the glossary for a weekly ESG briefing. Return STRICT JSON:

{"terms": [{"term": "CSRD", "meaning": "plain-language explanation"}]}

Rules:
- Include ONLY terms that actually appear in the briefing text supplied.
- Explain each in one plain-language sentence for a financially literate
  reader who is not a regulatory specialist.
- Order alphabetically. Omit terms needing no explanation."""


def missing_articles(markdown, rows):
    """Articles whose URL never made it into the briefing."""
    return [
        r for r in rows
        if (r.get("canonical_url") or r.get("url") or "") not in markdown
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _json_call(client, model, system, user, temperature):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = getattr(response, "usage", None)
    if usage:
        log.debug("tokens: prompt=%s completion=%s",
                  usage.prompt_tokens, usage.completion_tokens)
    return json.loads(response.choices[0].message.content or "{}"), usage


def analyse_articles(client, model, rows, temperature, batch_size=BATCH_SIZE):
    """Four analytical points per article, keyed by index.

    Batched so no single call carries the whole week, then any article the
    model still skipped is retried on its own. Coverage is driven by this
    loop, not by the model's willingness to keep writing.
    """
    analyses, total_tokens = {}, 0

    for start in range(0, len(rows), batch_size):
        batch = list(enumerate(rows, 1))[start:start + batch_size]
        payload = "\n\n".join(
            f"[{i}] {r.get('title')}\n"
            f"    source: {r.get('source_name')}\n"
            f"    date: {r.get('published_date') or 'unknown'}\n"
            f"    market: {r.get('primary_market')}\n"
            f"    pillars: {', '.join(r.get('esg_pillars') or []) or 'none'}\n"
            f"    regulations: {', '.join(r.get('regulations') or []) or 'none'}\n"
            f"    summary: {(r.get('summary') or '')[:700]}"
            for i, r in batch
        )
        data, usage = _json_call(
            client, model, SYSTEM_PROMPT,
            f"{ANALYSIS_SPEC}\n\n--- ARTICLES ---\n\n{payload}",
            temperature,
        )
        total_tokens += usage.total_tokens if usage else 0
        for story in data.get("stories", []):
            try:
                analyses[int(story["id"])] = story
            except (KeyError, TypeError, ValueError):
                log.warning("unparseable story object skipped: %r", story)

    # Anything still missing gets its own call - guarantees full coverage.
    for i, row in enumerate(rows, 1):
        if i in analyses:
            continue
        log.info("retrying article %d individually: %s", i, row.get("title", "")[:60])
        payload = (
            f"[{i}] {row.get('title')}\n"
            f"    source: {row.get('source_name')}\n"
            f"    date: {row.get('published_date') or 'unknown'}\n"
            f"    summary: {(row.get('summary') or '')[:700]}"
        )
        data, usage = _json_call(
            client, model, SYSTEM_PROMPT,
            f"{ANALYSIS_SPEC}\n\n--- ARTICLES ---\n\n{payload}",
            temperature,
        )
        total_tokens += usage.total_tokens if usage else 0
        for story in data.get("stories", []):
            analyses[i] = story
            break

    log.info("analysed %d/%d articles (%d tokens)",
             len(analyses), len(rows), total_tokens)
    return analyses


def write_executive_summary(client, model, rows, temperature):
    payload = "\n".join(
        f"- {r.get('title')} ({r.get('primary_market')}, "
        f"{r.get('published_date')}): {(r.get('summary') or '')[:220]}"
        for r in rows
    )
    data, _ = _json_call(
        client, model, SYSTEM_PROMPT,
        f"{SUMMARY_SPEC}\n\n--- THIS WEEK'S DEVELOPMENTS ---\n\n{payload}",
        temperature,
    )
    bullets = [str(b).strip().lstrip("-• ").strip()
               for b in data.get("bullets", []) if str(b).strip()]
    return bullets[:8]


def write_glossary(client, model, body, temperature):
    data, _ = _json_call(
        client, model, SYSTEM_PROMPT,
        f"{GLOSSARY_SPEC}\n\n--- BRIEFING TEXT ---\n\n{body[:12000]}",
        temperature,
    )
    terms = []
    for item in data.get("terms", []):
        term = str(item.get("term", "")).strip()
        meaning = str(item.get("meaning", "")).strip().rstrip(".")
        if term and meaning:
            terms.append((term, meaning))
    return sorted(set(terms))


def _story_markdown(row, analysis):
    """Assemble one story. The Source line is written by code, never the model,
    so URLs and dates cannot be dropped, shortened or hallucinated."""
    url = row.get("canonical_url") or row.get("url") or ""
    date_str = row.get("published_date") or "date unknown"
    fields = [
        ("What happened", analysis.get("what_happened")),
        ("Why it matters", analysis.get("why_it_matters")),
        ("Who may be affected", analysis.get("who_may_be_affected")),
        ("What to watch next", analysis.get("what_to_watch_next")),
    ]
    lines = [f"### {row.get('title')}", ""]
    for label, value in fields:
        text = str(value).strip() if value else "Not stated in the source material."
        lines.append(f"**{label}** - {text}")
    lines += ["", f"Source: {row.get('source_name')}, {date_str} - {url}", ""]
    return "\n".join(lines)


def assemble_briefing(rows, analyses, bullets, glossary=None):
    """Build the Markdown deterministically from validated parts."""
    parts = ["## Executive Summary", ""]
    parts += [f"- {b}" for b in bullets]
    parts.append("")

    buckets = defaultdict(list)
    for i, row in enumerate(rows, 1):
        buckets[primary_section(row)].append((i, row))

    for section, _code in PILLAR_SECTIONS:
        members = buckets.get(section, [])
        if not members:
            continue
        parts += [f"## {section}", ""]
        for i, row in members:
            parts.append(_story_markdown(row, analyses.get(i, {})))

    if glossary:
        parts += ["## Terms Explained", ""]
        parts += [f"**{term}** - {meaning}." for term, meaning in glossary]
        parts.append("")

    return "\n".join(parts).strip()


def generate_briefing(rows, model=None, api_key=None, days=7, temperature=0.3,
                      max_repairs=2):
    """Call the model and return Markdown. Raises RuntimeError without a key."""
    api_key = api_key or get_api_key()
    if not api_key:
        raise RuntimeError(
            "No OPENAI_API_KEY found. Paste your key into the .env file "
            "(OPENAI_API_KEY=\"sk-...\") or set it in the environment."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai SDK not installed: pip install openai") from exc

    model = model or os.environ.get("ESGN_MODEL") or DEFAULT_MODEL
    client = OpenAI(api_key=api_key)

    log.info("generating briefing: %d articles, model=%s", len(rows), model)

    # Structured, not single-shot. Asked to write all 20 stories in one go,
    # the model reliably stopped around 18 and dropped the rest no matter how
    # firmly the prompt insisted. Here the model only writes analysis; the
    # code drives coverage and emits every heading and Source line, so URL and
    # date preservation is guaranteed rather than hoped for.
    analyses = analyse_articles(client, model, rows, temperature)
    bullets = write_executive_summary(client, model, rows, temperature)
    body = assemble_briefing(rows, analyses, bullets)
    glossary = write_glossary(client, model, body, temperature)

    return assemble_briefing(rows, analyses, bullets, glossary)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def validate_briefing(markdown, rows):
    """Check the model actually honoured the contract.

    Returns a list of human-readable problems (empty means clean). Cheap
    insurance: a model given nine constraints will silently drop one, and a
    briefing missing source URLs is worse than useless for compliance.
    """
    problems = []

    if not markdown.lstrip().startswith("## Executive Summary"):
        problems.append("does not start with '## Executive Summary'")

    summary = markdown.split("##", 2)
    if len(summary) > 1:
        bullets = re.findall(r"^\s*[-*]\s+\S", summary[1], re.MULTILINE)
        if not 5 <= len(bullets) <= 8:
            problems.append(
                f"executive summary has {len(bullets)} bullets, expected 5-8"
            )

    missing_urls = [
        r.get("canonical_url") or r.get("url")
        for r in rows
        if (r.get("canonical_url") or r.get("url")) not in markdown
    ]
    if missing_urls:
        problems.append(
            f"{len(missing_urls)}/{len(rows)} source URLs missing from the report"
        )

    missing_dates = [
        r["published_date"] for r in rows
        if r.get("published_date") and r["published_date"] not in markdown
    ]
    if missing_dates:
        problems.append(
            f"{len(set(missing_dates))} publication date(s) missing from the report"
        )

    for required in ("What happened", "Why it matters",
                     "Who may be affected", "What to watch next"):
        if required.lower() not in markdown.lower():
            problems.append(f"no story addresses '{required}'")

    if "## Terms Explained" not in markdown:
        problems.append("missing '## Terms Explained' glossary")

    return problems


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def build_header(rows, model, days):
    now = datetime.now(timezone.utc)
    dates = sorted(r["published_date"] for r in rows if r.get("published_date"))
    window = f"{dates[0]} to {dates[-1]}" if dates else "n/a"
    sources = sorted({r.get("source_name") for r in rows if r.get("source_name")})
    return (
        f"# ESG & Sustainability Weekly Briefing\n\n"
        f"**Coverage:** EU and US markets, trailing {days} days ({window})  \n"
        f"**Articles analysed:** {len(rows)} (highest relevance, deduplicated)  \n"
        f"**Sources:** {', '.join(sources)}  \n"
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')} using `{model}`\n\n"
        f"---\n\n"
    )


def write_report(markdown, out_path=None, rows=None, model=DEFAULT_MODEL, days=7):
    out_path = Path(
        out_path
        or PROJECT_ROOT / "reports" / f"esg_briefing_{date.today().isoformat()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = markdown
    if rows is not None:
        body = build_header(rows, model, days) + markdown
    out_path.write_text(body + "\n", encoding="utf-8")
    return out_path


def run(curated_root=None, out_path=None, top_n=None, model=None,
        markets=None, days=7, temperature=0.3):
    """Full analyst leg. Returns (path, problems, article_count)."""
    load_env()
    top_n = top_n or int(os.environ.get("ESGN_TOP_N") or DEFAULT_TOP_N)
    model = model or os.environ.get("ESGN_MODEL") or DEFAULT_MODEL

    rows = select_top(load_curated(curated_root), top_n=top_n, markets=markets)
    if not rows:
        raise RuntimeError("No curated articles matched the given filters.")

    markdown = generate_briefing(rows, model=model, days=days,
                                 temperature=temperature)
    problems = validate_briefing(markdown, rows)
    path = write_report(markdown, out_path, rows=rows, model=model, days=days)
    return path, problems, len(rows)
