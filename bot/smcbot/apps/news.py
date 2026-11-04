"""Refresh the ForexFactory calendar used for news analysis/filters.

    python -m smcbot news                 # download + trim -> data/news/ff_calendar.csv
    python -m smcbot news --since 2020-01-01
    docker compose run --rm news

Source mirror (updated daily, times in UTC): smcbot.news.NEWS_URL.
You can also pass your own ForexFactory-style CSV: python -m smcbot news --file my.csv --tz UTC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smcbot.market.news import NEWS_URL, fetch_calendar, load_calendar, save_trimmed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=NEWS_URL)
    ap.add_argument("--file", default="", help="use a local CSV instead of downloading")
    ap.add_argument("--tz", default="UTC", help="time zone of the CSV times")
    ap.add_argument("--since", default="2023-01-01")
    import os
    ap.add_argument("--out", default=str(Path(os.getenv("DATA_DIR", "data")) / "news" / "ff_calendar.csv"))
    a = ap.parse_args(argv)
    src = Path(a.file) if a.file else fetch_calendar(Path(a.out).with_name("forexfactory_calendar_full.csv"), a.url)
    cal = load_calendar(src, tz=a.tz)
    save_trimmed(cal, a.out, a.since)
    print(f"{len(cal):,} USD/EUR/GBP medium+high events, {cal['time'].min()} -> {cal['time'].max()} -> {a.out}")
    return 0

