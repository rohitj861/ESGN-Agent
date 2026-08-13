"""Registry of ESG / sustainability RSS + Atom feeds for the EU and US markets.

Feed URLs drift over time (publishers move to new CMSes, regulators restructure
their newsrooms). Run ``python run_esgn.py validate`` to probe every feed and
prune or fix whatever comes back dead before relying on a scheduled run.

Each entry:
    id           stable slug used as the join key in the curated dataset
    name         human readable publisher name
    url          RSS 2.0 / Atom / RDF endpoint
    market       EU | US | GLOBAL  -- the market the outlet primarily covers
    source_type  media | regulator | ngo | exchange
    language     ISO 639-1 code
"""

from __future__ import annotations

import urllib.parse


def google_news_url(query, hl="en-GB", gl="GB", ceid="GB:en"):
    """Site-scoped Google News RSS search.

    Escape hatch for publishers that hard-block programmatic access to their
    own feed (Euractiv answers 403 to every request, browser headers included).
    Google News still indexes them, so the headlines are read from there.

    Trade-off: item links are news.google.com redirects rather than publisher
    URLs, so cross-outlet URL dedupe cannot match these against the same story
    arriving from a direct feed. Same-day title dedupe still applies.
    """
    return (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + f"&hl={hl}&gl={gl}&ceid={ceid}"
    )


# --------------------------------------------------------------------------
# United States - trade press, climate/energy media
# --------------------------------------------------------------------------
US_MEDIA = [
    {
        "id": "esg_today",
        "name": "ESG Today",
        "url": "https://www.esgtoday.com/feed/",
        "market": "GLOBAL",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "esg_dive",
        "name": "ESG Dive",
        "url": "https://www.esgdive.com/feeds/news/",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "utility_dive",
        "name": "Utility Dive",
        "url": "https://www.utilitydive.com/feeds/news/",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "trellis",
        "name": "Trellis (formerly GreenBiz)",
        "url": "https://trellis.net/feed/",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "canary_media",
        "name": "Canary Media",
        "url": "https://www.canarymedia.com/feed",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "inside_climate_news",
        "name": "Inside Climate News",
        "url": "https://insideclimatenews.org/feed/",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "grist",
        "name": "Grist",
        "url": "https://grist.org/feed/",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "esg_news",
        "name": "ESG News",
        "url": "https://esgnews.com/feed/",
        "market": "US",
        "source_type": "media",
        "language": "en",
    },
]

# --------------------------------------------------------------------------
# United States - regulators and agencies
# --------------------------------------------------------------------------
US_REGULATORS = [
    {
        "id": "sec_press",
        "name": "SEC Press Releases",
        "url": "https://www.sec.gov/news/pressreleases.rss",
        "market": "US",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "sec_speeches",
        "name": "SEC Speeches & Statements",
        "url": "https://www.sec.gov/news/speeches-statements.rss",
        "market": "US",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "noaa_news",
        "name": "NOAA",
        "url": "https://www.noaa.gov/rss.xml",
        "market": "US",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "doe_news",
        "name": "US Department of Energy",
        "url": "https://www.energy.gov/rss/articles.xml",
        "market": "US",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "cftc_press",
        "name": "CFTC Press Releases",
        "url": "https://www.cftc.gov/RSS/RSSGP/rssgp.xml",
        "market": "US",
        "source_type": "regulator",
        "language": "en",
    },
]

# --------------------------------------------------------------------------
# European Union - trade press
# --------------------------------------------------------------------------
EU_MEDIA = [
    {
        "id": "edie",
        "name": "edie",
        "url": "https://www.edie.net/feed/",
        "market": "EU",
        "source_type": "media",
        "language": "en",
    },
    {
        # Requested explicitly. The direct feed at euractiv.com/feed/ returns
        # 403 to every programmatic client, so it is relayed via Google News.
        "id": "euractiv",
        "name": "Euractiv (via Google News)",
        "url": google_news_url(
            "site:euractiv.com (climate OR energy OR environment OR "
            "sustainability OR ESG OR emissions) when:7d"
        ),
        "market": "EU",
        "source_type": "media",
        "language": "en",
        "via": "google_news",
    },
    {
        # Requested explicitly. Direct feed, works without workarounds.
        "id": "responsible_investor",
        "name": "Responsible Investor",
        "url": "https://www.responsible-investor.com/feed/",
        "market": "EU",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "guardian_environment",
        "name": "The Guardian - Environment",
        "url": "https://www.theguardian.com/environment/rss",
        "market": "EU",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "pv_magazine",
        "name": "pv magazine",
        "url": "https://www.pv-magazine.com/feed/",
        "market": "EU",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "clean_energy_wire",
        "name": "Clean Energy Wire",
        "url": "https://www.cleanenergywire.org/rss.xml",
        "market": "EU",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "climate_home_news",
        "name": "Climate Home News",
        "url": "https://www.climatechangenews.com/feed/",
        "market": "GLOBAL",
        "source_type": "media",
        "language": "en",
    },
    {
        "id": "eu_observer",
        "name": "EUobserver",
        "url": "https://euobserver.com/rss",
        "market": "EU",
        "source_type": "media",
        "language": "en",
    },
]

# --------------------------------------------------------------------------
# European Union - institutions, regulators, agencies
# --------------------------------------------------------------------------
EU_REGULATORS = [
    {
        "id": "ec_presscorner",
        "name": "European Commission - Press Corner",
        "url": "https://ec.europa.eu/commission/presscorner/api/rss?language=en",
        "market": "EU",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "europarl_press",
        "name": "European Parliament - Press Releases",
        "url": "https://www.europarl.europa.eu/rss/doc/press-releases/en.xml",
        "market": "EU",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "ec_energy",
        "name": "European Commission - Energy",
        "url": "https://energy.ec.europa.eu/node/2/rss_en",
        "market": "EU",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "esma_news",
        "name": "ESMA",
        "url": "https://www.esma.europa.eu/rss.xml",
        "market": "EU",
        "source_type": "regulator",
        "language": "en",
    },
    {
        "id": "ecb_press",
        "name": "European Central Bank",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "market": "EU",
        "source_type": "regulator",
        "language": "en",
    },
]

# --------------------------------------------------------------------------
# Cross-market / NGO / standard setters
# --------------------------------------------------------------------------
GLOBAL_FEEDS = [
    {
        "id": "wri_insights",
        "name": "World Resources Institute",
        "url": "https://www.wri.org/insights/rss.xml",
        "market": "GLOBAL",
        "source_type": "ngo",
        "language": "en",
    },
    {
        "id": "carbon_brief",
        "name": "Carbon Brief",
        "url": "https://www.carbonbrief.org/feed/",
        "market": "GLOBAL",
        "source_type": "media",
        "language": "en",
    },
]

ALL_FEEDS = US_MEDIA + US_REGULATORS + EU_MEDIA + EU_REGULATORS + GLOBAL_FEEDS

# Endpoints probed and rejected on 2026-08-14 -- kept here so nobody re-adds
# them without checking. Re-probe occasionally; several may come back.
KNOWN_DEAD = {
    "energypost.eu/feed/": "domain repurposed, feed now serves casino spam",
    "www.eea.europa.eu/en/newsroom/news/RSS": "404 - EEA site migration, no working RSS found",
    "www.eba.europa.eu/rss.xml": "persistent read timeout",
    "www.epa.gov/newsreleases/search/rss": "405 - blocks non-browser clients",
    "www.euractiv.com/sections/energy-environment/feed/": "403 - Cloudflare block",
    "www.environmentalleader.com/feed/": "404 - feed removed",
    "unfccc.int/rss.xml": "valid RSS but permanently empty channel",
    "environment.ec.europa.eu/node/2/rss_en": "404 - use ec_presscorner instead",
}


def get_feeds(markets=None, source_types=None, feed_ids=None):
    """Filter the registry.

    markets       iterable of EU | US | GLOBAL (None = all)
    source_types  iterable of media | regulator | ngo | exchange (None = all)
    feed_ids      explicit allow-list of feed ids (None = all)
    """
    feeds = ALL_FEEDS

    if feed_ids:
        wanted = {f.lower() for f in feed_ids}
        feeds = [f for f in feeds if f["id"].lower() in wanted]

    if markets:
        wanted = {m.upper() for m in markets}
        # GLOBAL outlets are relevant to any requested market, so keep them
        # unless the caller asked for GLOBAL exclusively.
        if wanted != {"GLOBAL"}:
            wanted = wanted | {"GLOBAL"}
        feeds = [f for f in feeds if f["market"] in wanted]

    if source_types:
        wanted = {s.lower() for s in source_types}
        feeds = [f for f in feeds if f["source_type"] in wanted]

    return feeds
