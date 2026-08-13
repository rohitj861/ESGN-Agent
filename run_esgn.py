#!/usr/bin/env python
"""ESGN Agent CLI.

    python run_esgn.py run                     # full pipeline, EU + US, 7 days
    python run_esgn.py run --markets EU        # EU only
    python run_esgn.py run --engine spark      # force PySpark
    python run_esgn.py fetch --days 7          # ingest only
    python run_esgn.py curate --engine local   # transform only
    python run_esgn.py validate                # probe all feeds, print health
    python run_esgn.py digest --top 25         # print the top stories
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from esgn import pipeline
from esgn.feeds import ALL_FEEDS


def _setup_logging(verbose):
    # Windows consoles default to cp1252; feed titles are full of non-Latin-1
    # characters, so force UTF-8 and never let encoding kill a run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _common_args(parser):
    parser.add_argument("--markets", nargs="+", default=["EU", "US"],
                        choices=["EU", "US", "GLOBAL"],
                        help="Markets to cover (default: EU US)")
    parser.add_argument("--source-types", nargs="+", default=None,
                        choices=["media", "regulator", "ngo", "exchange"])
    parser.add_argument("--feed-ids", nargs="+", default=None,
                        help="Restrict to specific feed ids")
    parser.add_argument("--days", type=int, default=7,
                        help="Trailing window in days (default: 7)")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--curated-root", default=None)
    parser.add_argument("--min-relevance", type=float, default=0.15)
    parser.add_argument("--engine", default="auto",
                        choices=["auto", "spark", "local"])
    parser.add_argument("--master", default=None,
                        help="Spark master (default: local[4])")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore stored ETag / Last-Modified headers")


def cmd_run(args):
    stats = pipeline.run_pipeline(
        markets=args.markets, source_types=args.source_types,
        feed_ids=args.feed_ids, days=args.days, raw_root=args.raw_root,
        curated_root=args.curated_root, min_relevance=args.min_relevance,
        engine=args.engine, master=args.master, workers=args.workers,
        timeout=args.timeout, use_cache=not args.no_cache,
    )
    _print_summary(stats)
    return 0


def cmd_fetch(args):
    _, stats = pipeline.ingest(
        markets=args.markets, source_types=args.source_types,
        feed_ids=args.feed_ids, days=args.days, raw_root=args.raw_root,
        workers=args.workers, timeout=args.timeout, use_cache=not args.no_cache,
    )
    _print_summary(stats)
    return 0


def cmd_curate(args):
    count, path, engine = pipeline.curate(
        raw_root=args.raw_root, curated_root=args.curated_root,
        days=args.days, min_relevance=args.min_relevance,
        engine=args.engine, master=args.master,
    )
    print(f"\nCurated {count} articles via {engine} -> {path}")
    return 0


def cmd_validate(args):
    report = pipeline.validate_feeds(
        markets=args.markets, source_types=args.source_types,
        timeout=args.timeout, workers=args.workers,
    )
    width = max(len(r["source_id"]) for r in report) + 2
    print(f"\n{'FEED':<{width}} {'MKT':<7} {'STATUS':<7} {'ITEMS':>6}  NEWEST")
    print("-" * (width + 46))
    for row in sorted(report, key=lambda r: (r["status"] != "OK", r["source_id"])):
        newest = (row.get("newest") or "-")[:19]
        print(f"{row['source_id']:<{width}} {row['market']:<7} "
              f"{row['status']:<7} {row.get('items', 0):>6}  {newest}")
        if row["status"] == "FAIL":
            print(f"{'':<{width}} -> {row['url']}")
            print(f"{'':<{width}} -> {row['error']}")

    ok = sum(1 for r in report if r["status"] == "OK")
    print(f"\n{ok}/{len(report)} feeds healthy.")
    return 0 if ok else 1


def cmd_digest(args):
    root = Path(args.curated_root or pipeline.DEFAULT_ROOT / "curated")
    path = root / "articles.jsonl"
    if not path.exists():
        print(f"No curated output at {path}. Run: python run_esgn.py run")
        return 1

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.market:
        rows = [r for r in rows if args.market.upper() in (r.get("markets") or [])]
    rows = rows[: args.top]

    print(f"\nTop {len(rows)} ESG stories"
          f"{' (' + args.market.upper() + ')' if args.market else ''}\n")
    for i, row in enumerate(rows, 1):
        tags = "/".join(row.get("esg_pillars") or []) or "-"
        regs = ", ".join(row.get("regulations") or [])
        print(f"{i:>3}. [{row.get('relevance_score'):.2f}] [{tags}] "
              f"[{row.get('primary_market')}] {row.get('title')}")
        print(f"     {row.get('source_name')} · "
              f"{(row.get('published_date') or '?')}"
              f"{' · ' + regs if regs else ''}")
        print(f"     {row.get('canonical_url') or row.get('url')}\n")
    return 0


def cmd_brief(args):
    from esgn import analyst

    try:
        path, problems, count = analyst.run(
            curated_root=args.curated_root, out_path=args.out,
            top_n=args.top, model=args.model, markets=args.markets,
            days=args.days, temperature=args.temperature,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\n{exc}\n")
        return 1

    print(f"\nBriefing written: {path}")
    print(f"Articles analysed: {count}")
    if problems:
        print("\nContract checks FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nRe-run to regenerate, or raise --top for more source material.")
        return 1
    print("Contract checks passed (summary, sections, URLs, dates, glossary).")
    return 0


def cmd_feeds(args):
    print(f"\n{len(ALL_FEEDS)} registered feeds:\n")
    for feed in sorted(ALL_FEEDS, key=lambda f: (f["market"], f["id"])):
        print(f"  {feed['market']:<7} {feed['source_type']:<10} "
              f"{feed['id']:<24} {feed['name']}")
    return 0


def _print_summary(stats):
    print("\n" + "=" * 62)
    print(f"  Run {stats['run_id']}  ({stats['window_days']}-day window)")
    print("=" * 62)
    print(f"  feeds ok / requested : {stats['feeds_ok']}/{stats['feeds_requested']}")
    print(f"  not modified (304)   : {stats['feeds_not_modified']}")
    print(f"  failed               : {stats['feeds_failed']}")
    print(f"  articles parsed      : {stats['articles_parsed']}")
    print(f"  in {stats['window_days']}-day window       : "
          f"{stats['articles_in_window']}")
    if "curated_articles" in stats:
        print(f"  curated (deduped)    : {stats['curated_articles']}")
        print(f"  engine               : {stats['engine']}")
        print(f"  output               : {stats['curated_path']}")
    if stats["failures"]:
        print("\n  Failed feeds:")
        for failure in stats["failures"]:
            print(f"    - {failure['source_id']}: {failure['error']}")
        print("  (run `python run_esgn.py validate` to triage)")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_esgn.py",
        description="Fetch EU + US ESG / sustainability news.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, help_text in [
        ("run", cmd_run, "Fetch + curate (full pipeline)"),
        ("fetch", cmd_fetch, "Fetch feeds into the raw landing zone"),
        ("curate", cmd_curate, "Transform raw -> curated"),
        ("validate", cmd_validate, "Probe every feed and report health"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        _common_args(sp)
        sp.set_defaults(func=func)

    sp = sub.add_parser("digest", help="Print top stories from curated output")
    sp.add_argument("--top", type=int, default=25)
    sp.add_argument("--market", default=None, choices=["EU", "US", "GLOBAL"])
    sp.add_argument("--curated-root", default=None)
    sp.set_defaults(func=cmd_digest)

    sp = sub.add_parser("brief", help="Generate the AI weekly ESG briefing")
    sp.add_argument("--top", type=int, default=None,
                    help="Articles fed to the analyst (default: 20)")
    sp.add_argument("--model", default=None, help="Default: gpt-4o-mini")
    sp.add_argument("--markets", nargs="+", default=None,
                    choices=["EU", "US", "GLOBAL"])
    sp.add_argument("--days", type=int, default=7)
    sp.add_argument("--temperature", type=float, default=0.3)
    sp.add_argument("--out", default=None, help="Output .md path")
    sp.add_argument("--curated-root", default=None)
    sp.set_defaults(func=cmd_brief)

    sp = sub.add_parser("feeds", help="List the feed registry")
    sp.set_defaults(func=cmd_feeds)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
