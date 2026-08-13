"""Static site generator.

Renders the curated dataset and the weekly briefing into a single
self-contained `index.html` suitable for any static host (Vercel, Netlify,
GitHub Pages). No build step, no external assets, no runtime.

    python run_esgn.py site
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Minimal Markdown -> HTML
# ---------------------------------------------------------------------------
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")
_SOURCE_RE = re.compile(r"^Source:\s*(.+?),\s*(.+?)\s+-\s+(https?://\S+)\s*$")
_URL_RE = re.compile(r"(https?://[^\s<]+)")
_META_RE = re.compile(r"^\*\*[^*]+:\*\*")


def _inline(text):
    """Escape, then apply the small subset of inline Markdown the briefing uses."""
    out = html.escape(text)
    out = _CODE_RE.sub(r"<code>\1</code>", out)
    out = _BOLD_RE.sub(r"<strong>\1</strong>", out)
    return out


def render_markdown(md):
    """Render the briefing's Markdown.

    Deliberately narrow: it only handles the constructs `analyst.py` emits
    (h1-h3, bold, bullets, horizontal rules, Source lines, paragraphs). The
    briefing is machine-generated, so the input shapes are known and fixed.
    """
    lines, out, in_list = md.splitlines(), [], False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            close_list()
            continue

        if stripped == "---":
            close_list()
            out.append("<hr>")
            continue

        source = _SOURCE_RE.match(stripped)
        if source:
            close_list()
            name, when, url = source.groups()
            out.append(
                f'<p class="source"><span class="src-name">{html.escape(name)}</span>'
                f'<span class="src-sep">·</span>'
                f'<time>{html.escape(when)}</time>'
                f'<a href="{html.escape(url, quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">View source →</a></p>'
            )
            continue

        # The page supplies its own <h1>, so the briefing's title would just
        # repeat it.
        if stripped.startswith("# "):
            close_list()
            continue

        for level, prefix in ((3, "### "), (2, "## ")):
            if stripped.startswith(prefix):
                close_list()
                slug = re.sub(r"[^a-z0-9]+", "-",
                              stripped[len(prefix):].lower()).strip("-")
                out.append(
                    f'<h{level} id="{slug}">{_inline(stripped[len(prefix):])}'
                    f"</h{level}>"
                )
                break
        else:
            if stripped.startswith("- "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{_inline(stripped[2:])}</li>")
            else:
                close_list()
                body = _inline(stripped)
                body = _URL_RE.sub(
                    r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
                    body,
                )
                cls = ' class="meta"' if _META_RE.match(stripped) else ""
                out.append(f"<p{cls}>{body}</p>")

    close_list()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_articles(curated_root=None):
    path = Path(curated_root or PROJECT_ROOT / "data" / "curated") / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No curated data at {path}. Run: python run_esgn.py run")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def latest_briefing(reports_dir=None):
    reports = Path(reports_dir or PROJECT_ROOT / "reports")
    if not reports.exists():
        return None
    candidates = sorted(reports.glob("esg_briefing_*.md"))
    return candidates[-1] if candidates else None


def build_stats(articles):
    markets, pillars, sources, regs = {}, {}, {}, {}
    for a in articles:
        markets[a.get("primary_market")] = markets.get(a.get("primary_market"), 0) + 1
        sources[a.get("source_name")] = sources.get(a.get("source_name"), 0) + 1
        for p in a.get("esg_pillars") or []:
            pillars[p] = pillars.get(p, 0) + 1
        for r in a.get("regulations") or []:
            regs[r] = regs.get(r, 0) + 1
    dates = sorted(a["published_date"] for a in articles if a.get("published_date"))
    return {
        "total": len(articles),
        "markets": dict(sorted(markets.items(), key=lambda kv: -kv[1])),
        "pillars": pillars,
        "sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "regulations": dict(sorted(regs.items(), key=lambda kv: -kv[1])),
        "window": (dates[0], dates[-1]) if dates else ("n/a", "n/a"),
        "source_count": len(sources),
    }


def _slim(article):
    """Only the fields the page actually renders, to keep the payload small."""
    return {
        "t": article.get("title"),
        "u": article.get("canonical_url") or article.get("url"),
        "s": article.get("source_name"),
        "st": article.get("source_type"),
        "d": article.get("published_date"),
        "m": article.get("primary_market"),
        "p": article.get("esg_pillars") or [],
        "tp": article.get("topics") or [],
        "r": article.get("regulations") or [],
        "sc": round(article.get("relevance_score") or 0, 3),
        "sm": (article.get("summary") or "")[:320],
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#fbfbfa; --panel:#fff; --ink:#1a1a18; --muted:#6b6b66; --line:#e6e5e1;
  --accent:#2f6f4f; --accent-soft:#eaf3ee; --chip:#f2f1ed; --shadow:0 1px 2px rgba(0,0,0,.05);
  --e:#2f6f4f; --s:#8a5a2b; --g:#3b5b8c;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#131313; --panel:#1a1a1a; --ink:#ececea; --muted:#9a9a94; --line:#2c2c2c;
  --accent:#6fbf8f; --accent-soft:#18291f; --chip:#242424; --shadow:none;
  --e:#6fbf8f; --s:#d19a5e; --g:#7fa3d8;
}}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px 80px}
header.top{padding:44px 0 26px;border-bottom:1px solid var(--line);margin-bottom:26px}
h1{font-size:clamp(1.6rem,3.6vw,2.2rem);line-height:1.2;margin:0 0 10px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0;font-size:.94rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:22px 0 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow)}
.stat b{display:block;font-size:1.5rem;line-height:1.15;letter-spacing:-.02em}
.stat span{color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.06em}
nav.tabs{display:flex;gap:6px;margin:26px 0 22px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.tab{background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);
  padding:9px 14px;font:inherit;font-size:.94rem;cursor:pointer;margin-bottom:-1px}
.tab[aria-selected=true]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.tab:hover{color:var(--ink)}
.panel[hidden]{display:none}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
input[type=search],select{font:inherit;font-size:.9rem;padding:8px 11px;border:1px solid var(--line);
  border-radius:8px;background:var(--panel);color:var(--ink);min-width:0}
input[type=search]{flex:1 1 240px}
.count{color:var(--muted);font-size:.86rem;margin-left:auto;white-space:nowrap}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px 18px;
  margin-bottom:11px;box-shadow:var(--shadow)}
.card h3{margin:0 0 7px;font-size:1.02rem;line-height:1.4;font-weight:600}
.card h3 a{color:inherit;text-decoration:none}
.card h3 a:hover{color:var(--accent);text-decoration:underline}
.card p{margin:0 0 11px;color:var(--muted);font-size:.9rem}
.tags{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.chip{font-size:.72rem;padding:2.5px 8px;border-radius:999px;background:var(--chip);
  color:var(--muted);white-space:nowrap;letter-spacing:.01em}
.chip.mk{background:var(--accent-soft);color:var(--accent);font-weight:600}
.chip.reg{border:1px solid var(--line);background:transparent}
.chip.E{color:var(--e)}.chip.S{color:var(--s)}.chip.G{color:var(--g)}
.score{font-variant-numeric:tabular-nums;font-size:.72rem;color:var(--muted);margin-left:auto}
.empty{text-align:center;color:var(--muted);padding:50px 0}
article.brief h1{font-size:1.5rem;margin:0 0 6px}
article.brief h2{font-size:1.22rem;margin:36px 0 4px;padding-bottom:7px;
  border-bottom:1px solid var(--line);letter-spacing:-.01em}
article.brief h3{font-size:1.02rem;margin:26px 0 9px;font-weight:600}
article.brief p{margin:0 0 9px}
article.brief ul{margin:0 0 16px;padding-left:20px}
article.brief li{margin-bottom:7px}
article.brief hr{border:0;border-top:1px solid var(--line);margin:20px 0}
article.brief .meta{color:var(--muted);font-size:.88rem;margin-bottom:2px}
article.brief .meta strong{color:var(--ink);font-weight:600}
article.brief strong{font-weight:650}
code{font:.88em ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--chip);
  padding:1.5px 5px;border-radius:5px}
.source{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.82rem;
  color:var(--muted);background:var(--chip);border-radius:8px;padding:7px 11px;margin:11px 0 0!important}
.src-name{font-weight:600;color:var(--ink)}
.src-sep{opacity:.5}
.source a{color:var(--accent);text-decoration:none;margin-left:auto;font-weight:500}
.source a:hover{text-decoration:underline}
footer{margin-top:52px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--muted);font-size:.84rem}
footer a{color:var(--accent)}
.note{background:var(--accent-soft);border-radius:9px;padding:12px 15px;font-size:.86rem;
  color:var(--ink);margin-bottom:20px;border:1px solid var(--line)}
@media(max-width:600px){.wrap{padding:0 14px 60px}header.top{padding:30px 0 20px}
  .count{margin-left:0;width:100%}}
"""

JS = """
const DATA = JSON.parse(document.getElementById('data').textContent);
const $ = s => document.querySelector(s);
const esc = s => String(s??'').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t =>
      t.setAttribute('aria-selected', String(t === tab)));
    document.querySelectorAll('.panel').forEach(p =>
      p.hidden = p.id !== tab.dataset.panel);
    // The briefing is far longer than the article list, so switching to the
    // shorter panel while scrolled deep would land the reader on blank page.
    const nav = document.querySelector('nav.tabs');
    if (window.scrollY > nav.offsetTop) window.scrollTo({ top: nav.offsetTop - 12 });
  });
});

function render() {
  const q = $('#q').value.trim().toLowerCase();
  const mk = $('#mk').value, pl = $('#pl').value, sr = $('#sr').value, sort = $('#sort').value;

  let rows = DATA.filter(a => {
    if (mk && a.m !== mk) return false;
    if (pl && !a.p.includes(pl)) return false;
    if (sr && a.s !== sr) return false;
    if (q && !((a.t||'') + ' ' + (a.sm||'') + ' ' + a.r.join(' ') + ' ' + a.tp.join(' '))
      .toLowerCase().includes(q)) return false;
    return true;
  });

  rows.sort(sort === 'date'
    ? (x, y) => (y.d||'').localeCompare(x.d||'') || y.sc - x.sc
    : (x, y) => y.sc - x.sc || (y.d||'').localeCompare(x.d||''));

  $('#count').textContent = rows.length + ' of ' + DATA.length + ' articles';
  $('#list').innerHTML = rows.length ? rows.map(a => `
    <div class="card">
      <h3><a href="${esc(a.u)}" target="_blank" rel="noopener noreferrer">${esc(a.t)}</a></h3>
      ${a.sm ? `<p>${esc(a.sm)}</p>` : ''}
      <div class="tags">
        <span class="chip mk">${esc(a.m)}</span>
        ${a.p.map(p => `<span class="chip ${p}">${p}</span>`).join('')}
        ${a.r.slice(0,3).map(r => `<span class="chip reg">${esc(r)}</span>`).join('')}
        <span class="chip">${esc(a.s)}</span>
        <span class="chip">${esc(a.d||'undated')}</span>
        <span class="score">${a.sc.toFixed(2)}</span>
      </div>
    </div>`).join('')
    : '<div class="empty">No articles match these filters.</div>';
}

['#q','#mk','#pl','#sr','#sort'].forEach(s => {
  $(s).addEventListener('input', render);
  $(s).addEventListener('change', render);
});
render();
"""


def build_page(articles, briefing_md, stats, generated_at=None):
    generated_at = generated_at or datetime.now(timezone.utc)
    payload = json.dumps([_slim(a) for a in articles], ensure_ascii=False)

    sources = sorted(stats["sources"])
    market_opts = "".join(
        f'<option value="{html.escape(m)}">{html.escape(m)} ({n})</option>'
        for m, n in stats["markets"].items() if m
    )
    source_opts = "".join(
        f'<option value="{html.escape(s)}">{html.escape(s)}</option>'
        for s in sources if s
    )

    brief_html = (
        render_markdown(briefing_md) if briefing_md
        else '<div class="empty">No briefing generated yet. '
             'Run <code>python run_esgn.py brief</code>.</div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESG &amp; Sustainability Intelligence — EU &amp; US</title>
<meta name="description" content="Weekly ESG and sustainability briefing and
 curated news dataset for the EU and US markets.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>ESG &amp; Sustainability Intelligence</h1>
  <p class="sub">EU &amp; US markets · {stats['window'][0]} to {stats['window'][1]}
   · {stats['source_count']} sources · deduplicated</p>
  <div class="stats">
    <div class="stat"><b>{stats['total']}</b><span>Articles</span></div>
    <div class="stat"><b>{stats['markets'].get('EU',0)}</b><span>EU</span></div>
    <div class="stat"><b>{stats['markets'].get('US',0)}</b><span>US</span></div>
    <div class="stat"><b>{stats['markets'].get('EU_US',0)}</b><span>EU + US</span></div>
    <div class="stat"><b>{stats['source_count']}</b><span>Sources</span></div>
    <div class="stat"><b>{len(stats['regulations'])}</b><span>Regulations</span></div>
  </div>
</header>

<nav class="tabs" role="tablist">
  <button class="tab" data-panel="brief" role="tab" aria-selected="true">Weekly Briefing</button>
  <button class="tab" data-panel="articles" role="tab" aria-selected="false">All Articles</button>
</nav>

<section class="panel" id="brief">
  <article class="brief">{brief_html}</article>
</section>

<section class="panel" id="articles" hidden>
  <div class="note">Every article below was fetched from a public RSS/Atom feed,
   scored for ESG relevance and deduplicated. Click a headline for the original source.</div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search headlines, summaries, regulations…"
     aria-label="Search articles">
    <select id="mk" aria-label="Filter by market"><option value="">All markets</option>{market_opts}</select>
    <select id="pl" aria-label="Filter by ESG pillar">
      <option value="">All pillars</option>
      <option value="E">Environment</option>
      <option value="S">Social</option>
      <option value="G">Governance</option>
    </select>
    <select id="sr" aria-label="Filter by source"><option value="">All sources</option>{source_opts}</select>
    <select id="sort" aria-label="Sort order">
      <option value="score">Sort: relevance</option>
      <option value="date">Sort: newest</option>
    </select>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</section>

<footer>
  <p>Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} by
   <a href="https://github.com/rohitj861/ESGN-Agent" target="_blank"
    rel="noopener noreferrer">ESGN Agent</a>.
   Briefing written by <code>gpt-4o-mini</code> from the curated dataset;
   every story links to its original publisher.</p>
  <p>Headlines and summaries remain the property of their respective publishers.
   This is an automated research aid, not investment advice.</p>
</footer>
</div>
<script type="application/json" id="data">{payload}</script>
<script>{JS}</script>
</body>
</html>"""


def build(curated_root=None, reports_dir=None, out_dir=None):
    """Render the static site. Returns (index_path, article_count)."""
    articles = load_articles(curated_root)
    stats = build_stats(articles)

    brief_path = latest_briefing(reports_dir)
    briefing_md = brief_path.read_text(encoding="utf-8") if brief_path else None
    if brief_path:
        log.info("using briefing: %s", brief_path.name)
    else:
        log.warning("no briefing found - the site will show the dataset only")

    out_dir = Path(out_dir or PROJECT_ROOT / "site")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(build_page(articles, briefing_md, stats), encoding="utf-8")

    log.info("site built: %s (%d articles)", index, len(articles))
    return index, len(articles)
