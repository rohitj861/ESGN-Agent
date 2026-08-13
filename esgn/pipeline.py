"""Orchestration: fetch -> raw landing zone -> transform (Spark or local)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import fetch as fetch_mod
from .feeds import get_feeds

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"


def spark_available():
    """True only if pyspark imports AND a JVM can be located."""
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    import os
    import shutil
    if shutil.which("java"):
        return True
    java_home = os.environ.get("JAVA_HOME")
    return bool(java_home and Path(java_home, "bin").exists())


def ingest(markets=None, source_types=None, feed_ids=None, days=7,
           raw_root=None, workers=fetch_mod.DEFAULT_WORKERS,
           timeout=fetch_mod.DEFAULT_TIMEOUT, use_cache=True):
    """Fetch every selected feed and land raw articles as JSONL.

    Returns (raw_dir, stats_dict).
    """
    raw_root = Path(raw_root or DEFAULT_ROOT / "raw")
    feeds = get_feeds(markets, source_types, feed_ids)
    if not feeds:
        raise ValueError("No feeds matched the given filters.")

    log.info("fetching %d feeds", len(feeds))
    cache_path = raw_root.parent / ".cache" / "http_cache.json"
    cache = fetch_mod.load_cache(cache_path) if use_cache else {}

    results = fetch_mod.fetch_all(
        feeds, timeout=timeout, workers=workers, cache=cache
    )

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    raw_dir = raw_root / f"ingest_date={now.date().isoformat()}" / f"run_id={run_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "run_id": run_id,
        "started_at": now.isoformat(),
        "feeds_requested": len(feeds),
        "feeds_ok": 0,
        "feeds_not_modified": 0,
        "feeds_failed": 0,
        "articles_parsed": 0,
        "articles_in_window": 0,
        "window_days": days,
        "failures": [],
        "per_feed": [],
    }

    out_path = raw_dir / "articles.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for result in sorted(results, key=lambda r: r.feed["id"]):
            if result.not_modified:
                stats["feeds_not_modified"] += 1
                stats["per_feed"].append(
                    {"source_id": result.feed["id"], "status": "not_modified",
                     "articles": 0}
                )
                continue
            if not result.ok:
                stats["feeds_failed"] += 1
                stats["failures"].append(
                    {"source_id": result.feed["id"], "url": result.feed["url"],
                     "error": result.error}
                )
                stats["per_feed"].append(
                    {"source_id": result.feed["id"], "status": "failed",
                     "articles": 0, "error": result.error}
                )
                continue

            articles = fetch_mod.parse_feed(result.body, result.feed)
            stats["articles_parsed"] += len(articles)

            kept = [a for a in articles if fetch_mod.within_window(a, days, now)]
            stats["articles_in_window"] += len(kept)
            for article in kept:
                fh.write(json.dumps(article, ensure_ascii=False) + "\n")

            stats["feeds_ok"] += 1
            stats["per_feed"].append(
                {"source_id": result.feed["id"], "status": "ok",
                 "articles_parsed": len(articles), "articles": len(kept)}
            )

    if use_cache:
        fetch_mod.save_cache(cache_path, results)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["raw_path"] = str(out_path)
    (raw_dir / "_ingest_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    log.info(
        "ingest complete: %d/%d feeds ok, %d articles in window",
        stats["feeds_ok"], stats["feeds_requested"], stats["articles_in_window"],
    )
    return raw_root, stats


def curate(raw_root=None, curated_root=None, days=7, min_relevance=0.15,
           engine="auto", master=None):
    """Run the transform leg. engine: auto | spark | local."""
    raw_root = Path(raw_root or DEFAULT_ROOT / "raw")
    curated_root = Path(curated_root or DEFAULT_ROOT / "curated")

    if engine == "auto":
        engine = "spark" if spark_available() else "local"
        log.info("engine auto-selected: %s", engine)

    if engine == "spark":
        from . import spark_job
        count, path = spark_job.run(
            raw_root, curated_root, days=days,
            min_relevance=min_relevance, master=master,
        )
    elif engine == "local":
        from . import local_job
        count, path = local_job.run(
            raw_root, curated_root, days=days, min_relevance=min_relevance
        )
    else:
        raise ValueError(f"unknown engine: {engine!r}")

    log.info("curated %d articles -> %s (engine=%s)", count, path, engine)
    return count, path, engine


def run_pipeline(markets=None, source_types=None, feed_ids=None, days=7,
                 raw_root=None, curated_root=None, min_relevance=0.15,
                 engine="auto", master=None, workers=8, timeout=20,
                 use_cache=True):
    """Full ingest + curate run."""
    raw_root, stats = ingest(
        markets=markets, source_types=source_types, feed_ids=feed_ids,
        days=days, raw_root=raw_root, workers=workers, timeout=timeout,
        use_cache=use_cache,
    )
    count, path, engine_used = curate(
        raw_root=raw_root, curated_root=curated_root, days=days,
        min_relevance=min_relevance, engine=engine, master=master,
    )
    stats["curated_articles"] = count
    stats["curated_path"] = path
    stats["engine"] = engine_used
    return stats


def validate_feeds(markets=None, source_types=None, timeout=15, workers=8):
    """Probe every feed and report reachability + item counts."""
    feeds = get_feeds(markets, source_types)
    results = fetch_mod.fetch_all(feeds, timeout=timeout, workers=workers, cache={})

    report = []
    for result in sorted(results, key=lambda r: r.feed["id"]):
        entry = {
            "source_id": result.feed["id"],
            "name": result.feed["name"],
            "market": result.feed["market"],
            "url": result.feed["url"],
        }
        if not result.ok:
            entry.update({"status": "FAIL", "items": 0, "error": result.error})
        else:
            items = fetch_mod.parse_feed(result.body, result.feed)
            dated = [a for a in items if a["published_at"]]
            entry.update({
                "status": "OK" if items else "EMPTY",
                "items": len(items),
                "dated_items": len(dated),
                "newest": max((a["published_at"] for a in dated), default=None),
            })
        report.append(entry)
    return report
