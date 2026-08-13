# ESGN Agent

Fetches ESG and sustainability news for the **EU and US markets** over a
trailing **7-day window**, enriches it (ESG pillar, regulation, topic, market),
deduplicates it, and writes a curated dataset.

Source layer is RSS/Atom only — no API keys, no rate limits, no vendor cost.

**Live dashboard:** https://esgn-agent.vercel.app

---

## Quick start

```bash
python run_esgn.py run            # full pipeline: EU + US, last 7 days
python run_esgn.py digest --top 25
```

No installation needed. The fetch, enrich and transform layers run on a bare
CPython install using only the standard library.

```bash
pip install -r requirements.txt   # optional: CSV/Parquet output, Spark, tests
```

---

## Commands

| Command | What it does |
| --- | --- |
| `run` | Fetch + curate (full pipeline) |
| `brief` | Generate the AI weekly ESG briefing (Markdown) |
| `fetch` | Fetch feeds into the raw landing zone only |
| `curate` | Transform raw → curated only (re-runnable offline) |
| `validate` | Probe every feed, report reachability and item counts |
| `digest` | Print the top-ranked stories from curated output |
| `site` | Build the static dashboard (`site/index.html`) |
| `feeds` | List the feed registry |

Useful flags:

```bash
python run_esgn.py run --markets EU              # EU only
python run_esgn.py run --source-types regulator  # primary sources only
python run_esgn.py run --days 14                 # widen the window
python run_esgn.py run --engine spark            # force PySpark
python run_esgn.py run --min-relevance 0.3       # stricter ESG gate
python run_esgn.py digest --market US --top 40
```

---

## AI analyst briefing

Turns the curated dataset into a professional weekly briefing.

```bash
python run_esgn.py brief                       # top 20 by relevance
python run_esgn.py brief --top 30 --markets EU
python run_esgn.py brief --model gpt-4o-mini --out reports/week32.md
```

**Setup:** paste your OpenAI key into `.env` (`OPENAI_API_KEY="sk-..."`).
`.env` is gitignored; `.env.example` is the committed template and must never
contain a real key. A shell/`setx` variable overrides the file.

Output goes to `reports/esg_briefing_<date>.md` with this structure:

- `## Executive Summary` — 5–8 bullets on the week's key developments
- `## Environment` / `## Social` / `## Governance` / `## Sustainable Finance`,
  each section omitted entirely when the source data gives no evidence for it
- Every story covers **what happened, why it matters, who may be affected,
  what to watch next**
- Every source URL and publication date preserved verbatim
- `## Terms Explained` glossary, with jargon also explained inline at first use

### Why generation is structured, not single-shot

The obvious build — one prompt containing all the rules and all 20 articles —
was tried first and does not hold up. `gpt-4o-mini` consistently wrote about
18 of the 20 stories and silently dropped the rest, however explicitly the
prompt demanded full coverage. Feeding the omissions back recovered them, but
the rewrite then quietly reformatted the document: the four points survived,
their bold labels did not, and the executive summary drifted to one bullet per
story (11).

So the model is never asked to produce the document. It is asked only for
**analysis** — the four points per article, as JSON, in batches of five — and
this module assembles the Markdown itself, emitting every section heading and
every `Source:` line in code. Any article the model skips is retried on its
own.

The result: full coverage and verbatim URL/date preservation are **guaranteed
by construction** rather than by the model's cooperation. That matters here,
because a briefing with missing citations is worse than useless for
compliance. `validate_briefing` still re-checks the finished document as a
safety net, prints any failures and exits non-zero.

Cost is negligible: ~6k tokens total across all calls, well under a cent per
run on `gpt-4o-mini`.

## Dashboard site

```bash
python run_esgn.py site      # writes site/index.html
```

One self-contained ~94 KB HTML file — no build step, no external assets, no
runtime, no tracking. Two tabs: the rendered weekly briefing, and a browser
over every curated article filterable by market, ESG pillar and source with
full-text search and relevance/date sorting. Light and dark themes follow the
viewer's system setting.

Deployed to Vercel from this repo. `site/index.html` is **committed as a build
artifact** rather than built on Vercel, because generating it requires the
curated dataset in `data/`, which is gitignored. To refresh what's live:

```bash
python run_esgn.py run && python run_esgn.py brief && python run_esgn.py site
git commit -am "Refresh dashboard" && git push    # Vercel redeploys on push
```

## Architecture

```
feeds.py     28-feed registry (EU / US / GLOBAL, media / regulator / ngo)
   |
fetch.py     concurrent HTTP + RSS 2.0 / Atom / RDF parsing   [stdlib only]
   |         conditional GET (ETag / Last-Modified), retries
   v
data/raw/ingest_date=YYYY-MM-DD/run_id=.../articles.jsonl     [immutable landing zone]
   |
   +-- spark_job.py   PySpark transform      (when a JVM is available)
   +-- local_job.py   pandas / pure-Python   (identical logic, same output)
   |
   v
data/curated/   articles.parquet + articles.csv + articles.jsonl + rollups.json
```

Raw is kept immutable and append-only, so you can re-run `curate` with
different thresholds without re-fetching. Both engines apply the **same**
window filter, relevance gate and dedupe precedence — output is interchangeable.

### Engine selection

`--engine auto` (the default) picks Spark only when **both** PySpark imports
*and* a JVM is present; otherwise it uses the local engine and logs the choice.
For 7 days of RSS (~500 articles) the local engine is faster; Spark earns its
keep when you widen the window, add hundreds of feeds, or join against
holdings/portfolio data.

**Both engines are verified to produce identical output.** On the same raw
snapshot they returned 101 articles each, with identical `article_id` sets and
zero field mismatches across `esg_pillars`, `regulations`, `topics`, `markets`,
`primary_market` and `relevance_score`. `tests/test_spark_job.py` pins this
with a parity test.

---

## Enrichment

Each article gets:

- **`esg_pillars`** — `E` / `S` / `G` from a keyword taxonomy
- **`regulations`** — CSRD, CSDDD, SFDR, ESRS, CBAM, EUDR, EU Taxonomy, EU ETS,
  Omnibus, SEC Climate Rule, CA SB 253/261, IRA, ISSB, TCFD, TNFD, GRI, UK SDR…
- **`topics`** — regulation_policy, reporting_disclosure, sustainable_finance,
  carbon_markets, energy_transition, corporate_action, litigation_enforcement,
  greenwashing, supply_chain
- **`markets`** / **`primary_market`** — `EU`, `US`, `EU_US`, or `GLOBAL`
- **`relevance_score`** — 0.0–1.0 weighted heuristic

Market attribution matches **acronyms case-sensitively** against the original
headline. This is deliberate: lowercasing first turns "US" (the country) into
"us" (the pronoun) and "DOE" (Dept. of Energy) into "doe" (a deer), which fires
constantly on nature and climate copy.

The scorer is keyword-driven on purpose — transparent, cheap, auditable. Swap
`enrich.score_article` for a model call without touching the pipeline.

### Deduplication

Same story syndicated across outlets collapses to one row. Precedence:
**regulator > higher relevance > earliest publisher**, so the primary source
wins over the trade-press rewrite.

1. Exact match on canonical URL (tracking params stripped)
2. Exact match on normalised title within the same day

---

## Feed registry

28 feeds, all verified reachable on 2026-08-14:

- **US media** — ESG Dive, Utility Dive, Trellis, Canary Media, Inside Climate
  News, Grist, ESG News
- **US regulators** — SEC (press + speeches), DOE, CFTC, NOAA
- **EU media** — edie, Responsible Investor, Euractiv, Clean Energy Wire,
  The Guardian Environment, pv magazine, EUobserver
- **EU institutions** — EC Press Corner, EC Energy, European Parliament, ECB, ESMA
- **Global** — ESG Today, Carbon Brief, Climate Home News, WRI

Feed URLs rot. Run `python run_esgn.py validate` before trusting a scheduled
run. Endpoints already probed and rejected are recorded in `feeds.KNOWN_DEAD`
with the reason, so they don't get re-added by accident — including
`energypost.eu`, whose domain now serves casino spam rather than energy news.

**Euractiv is relayed via Google News.** Its own feed answers 403 to every
programmatic client, browser headers and all — the whole domain is blocked,
not just the feed. `feeds.google_news_url()` runs a site-scoped Google News
query instead, and the parser strips the `" - publisher"` suffix Google
appends so it cannot poison title-based dedupe. The trade-off is that item
links are `news.google.com` redirects, so cross-outlet URL dedupe can't match
these against the same story from a direct feed. The same mechanism can
revive any other blocked publisher.

---

## Tests

```bash
python tests/test_pipeline.py     # 25 tests, stdlib unittest, no deps
python tests/test_analyst.py      # 17 tests, no API key needed
python tests/test_spark_job.py    # 10 tests, skips if PySpark/JVM absent
python -m pytest tests/ -q        # if pytest is installed
```

**52 tests, all passing.**

`test_pipeline.py` covers RSS/Atom/RDF parsing, timezone normalisation, URL
canonicalisation, window filtering, ESG classification, market attribution
false positives, dedupe precedence and rollups.

`test_spark_job.py` exercises the Spark transform against in-memory
DataFrames — deliberately avoiding file reads so it runs without Hadoop's
native Windows layer — and asserts Spark/local engine parity.

---

## Known limitations

- **Near-duplicate stories survive.** Dedupe is exact-match on URL and
  normalised title. "Climate Fund Managers Raises Over $180 Million" and
  "Climate Fund Managers Raise $182M" are the same story but survive as two
  rows. Fuzzy matching (MinHash/LSH) would catch these at the cost of a shuffle.
- **Headlines and summaries only.** RSS gives no article body, so classification
  works on ~50 words. Add a fetch-and-extract step if you need full text.
- **English only.** Non-English EU sources (Handelsblatt, Les Echos, Il Sole)
  are not covered; the keyword taxonomy would need translating.
- **Undated items are kept, not dropped.** Some regulator feeds omit dates
  entirely; they are flagged with `has_publish_date = false` rather than
  silently discarded.
## Running Spark

Verified working on this machine with **Temurin JDK 17.0.20 + PySpark 4.2.0 +
Python 3.14**. The JDK is installed at `%USERPROFILE%\.jdks\jdk-17.0.20+8` and
`JAVA_HOME` is set at user scope, so `--engine auto` now selects Spark.

Three Windows-specific problems came up; all three are handled in code:

1. **Spark's Python workers wouldn't start** — Spark invokes bare `python`,
   which Windows' App Execution Alias redirects to the Microsoft Store stub.
   `get_spark` now pins `PYSPARK_PYTHON`/`PYSPARK_DRIVER_PYTHON` to
   `sys.executable`.
2. **All file I/O failed** with `UnsatisfiedLinkError: NativeIO$Windows.access0`.
   Spark reaches the local filesystem through Hadoop, which on Windows needs
   `winutils.exe` + `hadoop.dll`. Those ship only as unofficial community
   builds, so rather than pull unsigned native binaries off GitHub, the job
   detects their absence (`hadoop_native_available()`) and routes reads and
   writes through the driver instead. The Spark transform itself is untouched.
   If you do install winutils and set `HADOOP_HOME`, it automatically switches
   back to native partitioned-Parquet I/O — no code change.
3. **`local[*]` crashed workers** on a 12-core / 15.7 GB box
   ("Python worker exited unexpectedly"). Each worker is a full interpreter
   process. Default master is now `local[4]`, which is stable; override with
   `--master` on a real cluster.

The driver-side I/O fallback means the whole `data/curated` output (Parquet,
CSV, JSONL, rollups) is written by the same code as the local engine — so
`digest` works after a Spark run too.

## Scheduling

```powershell
# Daily at 07:00
schtasks /create /tn "ESGN Daily" /tr "python 'C:\ESGN Agent\run_esgn.py' run" /sc daily /st 07:00
```

Conditional GET means repeat runs mostly return `304 Not Modified` and cost
almost nothing.
