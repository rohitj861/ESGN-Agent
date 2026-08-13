"""Feed fetching and RSS/Atom/RDF parsing.

Deliberately stdlib-only (urllib + ElementTree) so the ingest layer runs on a
bare Python install with nothing pip-installed. If ``requests`` happens to be
available it is used instead, purely for its connection pooling.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Several publishers reject unrecognised agent strings outright, so present a
# browser-compatible token while still identifying the crawler honestly.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 ESGN-Agent/0.1"
)
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 2
DEFAULT_WORKERS = 8

try:  # optional
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class FeedResult:
    """Outcome of a single feed fetch."""

    def __init__(self, feed, status=None, body=None, error=None,
                 etag=None, last_modified=None, not_modified=False):
        self.feed = feed
        self.status = status
        self.body = body
        self.error = error
        self.etag = etag
        self.last_modified = last_modified
        self.not_modified = not_modified

    @property
    def ok(self):
        return self.error is None and (self.body is not None or self.not_modified)


def _http_get(url, timeout, etag=None, last_modified=None):
    """Return (status, body_bytes_or_None, etag, last_modified).

    body is None when the server answers 304 Not Modified.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip",
    }
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    if _requests is not None:
        resp = _requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 304:
            return 304, None, etag, last_modified
        resp.raise_for_status()
        return (
            resp.status_code,
            resp.content,
            resp.headers.get("ETag", etag),
            resp.headers.get("Last-Modified", last_modified),
        )

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return (
                resp.status,
                raw,
                resp.headers.get("ETag", etag),
                resp.headers.get("Last-Modified", last_modified),
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return 304, None, etag, last_modified
        raise


def fetch_feed(feed, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, cache=None):
    """Fetch one feed with retries and conditional-GET support."""
    cache = cache or {}
    cached = cache.get(feed["url"], {})
    last_exc = None

    for attempt in range(retries + 1):
        try:
            status, body, etag, last_mod = _http_get(
                feed["url"],
                timeout=timeout,
                etag=cached.get("etag"),
                last_modified=cached.get("last_modified"),
            )
            if status == 304:
                log.info("304 not modified: %s", feed["id"])
                return FeedResult(feed, status=304, not_modified=True,
                                  etag=etag, last_modified=last_mod)
            log.info("fetched %s (%d bytes)", feed["id"], len(body or b""))
            return FeedResult(feed, status=status, body=body,
                              etag=etag, last_modified=last_mod)
        except (urllib.error.URLError, socket.timeout, OSError, Exception) as exc:
            last_exc = exc
            if attempt < retries:
                log.debug("retry %d for %s: %s", attempt + 1, feed["id"], exc)

    log.warning("failed %s: %s", feed["id"], last_exc)
    return FeedResult(feed, error=f"{type(last_exc).__name__}: {last_exc}")


def fetch_all(feeds, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES,
              workers=DEFAULT_WORKERS, cache=None):
    """Fetch every feed concurrently. Returns a list of FeedResult."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_feed, f, timeout, retries, cache): f for f in feeds
        }
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return results


# ---------------------------------------------------------------------------
# Cache (ETag / Last-Modified)
# ---------------------------------------------------------------------------
def load_cache(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(path, results):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(path)
    for r in results:
        if r.ok and (r.etag or r.last_modified):
            cache[r.feed["url"]] = {
                "etag": r.etag,
                "last_modified": r.last_modified,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"^\{[^}]+\}")
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _localname(tag):
    """Strip the XML namespace: '{http://purl.org/dc/elements/1.1/}date' -> 'date'."""
    return _TAG_RE.sub("", tag)


def _text(elem):
    if elem is None:
        return None
    parts = list(elem.itertext())
    if not parts:
        return None
    return _WS_RE.sub(" ", "".join(parts)).strip() or None


def strip_html(value):
    if not value:
        return None
    cleaned = _HTML_RE.sub(" ", value)
    cleaned = (cleaned.replace("&nbsp;", " ").replace("&amp;", "&")
               .replace("&lt;", "<").replace("&gt;", ">")
               .replace("&quot;", '"').replace("&#39;", "'"))
    return _WS_RE.sub(" ", cleaned).strip() or None


def _find_child(elem, *names):
    """First direct child whose local name matches any of *names."""
    wanted = {n.lower() for n in names}
    for child in elem:
        if _localname(child.tag).lower() in wanted:
            return child
    return None


def _find_children(elem, *names):
    wanted = {n.lower() for n in names}
    return [c for c in elem if _localname(c.tag).lower() in wanted]


def parse_datetime(value):
    """Parse RFC 822 (RSS) or ISO 8601 (Atom) into an aware UTC datetime."""
    if not value:
        return None
    value = value.strip()

    # RFC 822 / 2822 -- 'Wed, 06 Aug 2025 14:03:11 +0000'
    try:
        dt = parsedate_to_datetime(value)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass

    # ISO 8601 -- '2025-08-06T14:03:11Z' / with offset / date only
    iso = value.replace("Z", "+00:00")
    for candidate in (iso, iso[:19], iso[:10]):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    log.debug("unparseable date: %r", value)
    return None


_TIME_ATTR_RE = re.compile(r'datetime=["\']([^"\']+)["\']', re.IGNORECASE)
# Trailing " - euractiv.com" / " - Reuters" that Google News appends.
_GNEWS_SUFFIX_RE = re.compile(r"\s+-\s+[^-]{2,40}$")


def strip_publisher_suffix(title, publisher=None):
    """Remove the ' - Publisher' tail Google News appends to every headline.

    Left in place it would poison `title_key`, so the same story arriving from
    two outlets would no longer collapse during dedupe.
    """
    if not title:
        return title
    if publisher and title.endswith(f" - {publisher}"):
        return title[: -(len(publisher) + 3)].strip()
    # Fall back to trimming a short trailing segment that looks like a source
    # (a domain, or a handful of words) rather than part of the headline.
    match = _GNEWS_SUFFIX_RE.search(title)
    if match:
        tail = match.group(0).lstrip(" -").strip()
        if "." in tail or len(tail.split()) <= 4:
            return title[: match.start()].strip()
    return title


def date_from_html(raw_html):
    """Recover a timestamp from an HTML <time datetime="..."> in the body.

    Some regulator feeds (ESMA, for one) ship no pubDate at all and only carry
    the date inside the escaped HTML description.
    """
    if not raw_html:
        return None
    match = _TIME_ATTR_RE.search(raw_html)
    return parse_datetime(match.group(1)) if match else None


def _entry_link(entry):
    """Extract the article URL from an RSS <link> or an Atom <link href=...>."""
    for child in _find_children(entry, "link"):
        href = child.get("href")
        rel = (child.get("rel") or "alternate").lower()
        if href and rel == "alternate":
            return href.strip()
        if not href and child.text and child.text.strip():
            return child.text.strip()
    # Fall back to guid when it is a permalink
    guid = _find_child(entry, "guid", "id")
    if guid is not None and guid.text:
        text = guid.text.strip()
        if text.startswith("http"):
            return text
    return None


def canonicalize_url(url):
    """Drop tracking parameters and fragments so duplicates collapse."""
    if not url:
        return None
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return url.strip()

    drop_prefixes = ("utm_", "ito", "at_")
    drop_exact = {
        "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source",
        "cmpid", "smid", "_hsenc", "_hsmi", "sh",
    }
    query = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(drop_prefixes) and k.lower() not in drop_exact
    ]

    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"

    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), netloc, path, urllib.parse.urlencode(query), "")
    )


def parse_feed(body, feed):
    """Parse feed XML into a list of raw article dicts.

    Handles RSS 2.0 (channel/item), Atom (feed/entry) and RSS 1.0 / RDF.
    """
    if not body:
        return []

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        log.warning("XML parse error for %s: %s", feed["id"], exc)
        return []

    root_name = _localname(root.tag).lower()
    if root_name == "rss":
        channel = _find_child(root, "channel")
        entries = _find_children(channel, "item") if channel is not None else []
    elif root_name == "feed":  # Atom
        entries = _find_children(root, "entry")
    elif root_name == "rdf":  # RSS 1.0
        entries = _find_children(root, "item")
        if not entries:
            channel = _find_child(root, "channel")
            entries = _find_children(channel, "item") if channel is not None else []
    else:
        entries = _find_children(root, "item", "entry")

    fetched_at = datetime.now(timezone.utc)
    articles = []

    for entry in entries:
        title = _text(_find_child(entry, "title"))
        url = _entry_link(entry)
        if not title or not url:
            continue

        source_name = feed["name"]
        if feed.get("via") == "google_news":
            source_el = _find_child(entry, "source")
            publisher = _text(source_el) if source_el is not None else None
            title = strip_publisher_suffix(title, publisher)
            if publisher:
                source_name = publisher

        summary_el = _find_child(entry, "description", "summary", "subtitle")
        content_el = _find_child(entry, "encoded", "content")
        raw_body = _text(summary_el) or _text(content_el)
        summary = strip_html(raw_body)

        published_raw = _text(
            _find_child(entry, "pubDate", "published", "date", "updated", "created")
        )
        published_at = parse_datetime(published_raw)
        if published_at is None:
            published_at = date_from_html(raw_body)

        author = _text(_find_child(entry, "creator", "author"))
        if author is None:
            author_el = _find_child(entry, "author")
            if author_el is not None:
                author = _text(_find_child(author_el, "name"))

        categories = []
        for cat in _find_children(entry, "category", "subject"):
            label = cat.get("term") or _text(cat)
            if label:
                categories.append(label.strip())

        canonical = canonicalize_url(url)
        articles.append(
            {
                "article_id": hashlib.sha256(
                    (canonical or url).encode("utf-8")
                ).hexdigest()[:32],
                "title": strip_html(title),
                "url": url,
                "canonical_url": canonical,
                "summary": (summary[:2000] if summary else None),
                "author": author,
                "published_at": published_at.isoformat() if published_at else None,
                "published_raw": published_raw,
                "categories": sorted(set(categories))[:12],
                "source_id": feed["id"],
                "source_name": source_name,
                "source_market": feed["market"],
                "source_type": feed["source_type"],
                "language": feed.get("language", "en"),
                "feed_url": feed["url"],
                "fetched_at": fetched_at.isoformat(),
            }
        )

    return articles


def within_window(article, days, now=None):
    """True when the article was published inside the trailing `days` window.

    Items with no parseable date are kept -- some regulator feeds omit dates
    entirely, and dropping them silently loses real signal. They are flagged
    downstream via `has_publish_date`.
    """
    if not article.get("published_at"):
        return True
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    published = parse_datetime(article["published_at"])
    if published is None:
        return True
    # Allow a small forward skew for publishers with bad clocks / timezones.
    return cutoff <= published <= now + timedelta(hours=12)
