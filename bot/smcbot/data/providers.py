"""Historical data providers.

DATA_SOURCE=dukascopy : free M1 bid (+ask for the spread) from Dukascopy Bank,
                         cached month by month under DATA_DIR/cache, no API key.
DATA_SOURCE=csv       : MT5 'Export bars' files, one per symbol (CSV_<SYMBOL>).

All providers return M1 bars indexed by NY time with open/high/low/close (bid) and
spread (pips), the format the rest of the bot expects.
"""
from __future__ import annotations

import logging
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smcbot.config import NY_TZ
from smcbot.data.bars import drop_weekend, load_mt5_csv

log = logging.getLogger("smcbot.data")

DUKAS_CODES = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.normalize().replace(day=1)
    while cur < end:
        nxt = cur + pd.offsets.MonthBegin(1)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def _fetch_dukascopy_month(code: str, a: pd.Timestamp, b: pd.Timestamp, side: str,
                           retries: int = 3) -> pd.DataFrame:
    import dukascopy_python as dk
    offer = dk.OFFER_SIDE_BID if side == "bid" else dk.OFFER_SIDE_ASK
    socket.setdefaulttimeout(60)   # dukascopy_python calls requests.get without a timeout; urllib3 falls back to this
    for attempt in range(retries):
        try:
            return dk.fetch(code, dk.INTERVAL_MIN_1, offer,
                            a.to_pydatetime(), b.to_pydatetime())
        except Exception as e:  # network hiccups: retry with backoff
            log.warning("dukascopy %s %s %s..%s failed (%s), retry %d", code, side, a, b, e, attempt + 1)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Dukascopy download failed for {code} {a:%Y-%m}")


def load_dukascopy(symbol: str, start: str, end: str | None, data_dir: str,
                   fetch_ask: bool = True, pip: float = 0.0001) -> pd.DataFrame:
    code = DUKAS_CODES.get(symbol.upper())
    if code is None:
        raise ValueError(f"No Dukascopy code for {symbol}; known: {list(DUKAS_CODES)}")
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    cache = Path(data_dir) / "cache" / symbol.upper()
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    this_month = pd.Timestamp(datetime.now(timezone.utc)).normalize().replace(day=1)
    for a, b in _month_ranges(s, e):
        f = cache / f"{a:%Y-%m}.csv.gz"
        complete_month = a.replace(day=1) < this_month and b >= a.replace(day=1) + pd.offsets.MonthBegin(1)
        if f.exists() and complete_month:
            part = pd.read_csv(f, index_col=0, parse_dates=True)
        else:
            log.info("downloading %s %s ...", symbol, f"{a:%Y-%m}")
            bid = _fetch_dukascopy_month(code, a, b, "bid")
            part = bid[["open", "high", "low", "close"]].copy()
            if fetch_ask and len(bid):
                ask = _fetch_dukascopy_month(code, a, b, "ask")
                spr = (ask["close"] - bid["close"]).reindex(part.index)
                part["spread"] = (spr / pip).clip(lower=0).round(2)
            if complete_month and len(part):
                part.to_csv(f, compression="gzip")
        parts.append(part)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.DatetimeIndex(df.index, tz=None).tz_localize("UTC") if df.index.tz is None \
        else pd.DatetimeIndex(df.index)
    df.index = df.index.tz_convert(NY_TZ)
    df.index.name = "time"
    if "spread" in df:
        df["spread"] = df["spread"].fillna(df["spread"].median())
    return drop_weekend(df[(df["high"] >= df["low"])])


def load_csv_files(pattern: str, server_tz: str) -> pd.DataFrame:
    """One or many MT5 exports: a path, a wildcard (EURUSD_M1*.csv) or a comma list.
    Overlapping chunks are merged; duplicate bars are dropped."""
    import glob
    paths = []
    for part in pattern.split(","):
        part = part.strip()
        hits = sorted(glob.glob(part))
        paths += hits if hits else ([part] if part and not any(c in part for c in "*?[") else [])
    if not paths:
        raise FileNotFoundError(f"No CSV files match {pattern!r}")
    frames = [load_mt5_csv(p, server_tz) for p in paths]
    for p, f in zip(paths, frames):
        if len(f):
            log.info("%s: %s bars %s -> %s", Path(p).name, f"{len(f):,}", f.index[0], f.index[-1])
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def load_symbol(symbol: str, settings) -> pd.DataFrame:
    if settings.data_source == "csv":
        pattern = settings.csv_paths.get(symbol.upper())
        if not pattern:
            raise ValueError(f"Set CSV_{symbol.upper()} to the MT5 export path")
        df = load_csv_files(pattern, settings.mt5_server_tz)
    elif settings.data_source == "dukascopy":
        df = load_dukascopy(symbol, settings.date_from, settings.date_to, settings.data_dir,
                            settings.dukascopy_fetch_ask)
    elif settings.data_source == "mt5":
        from smcbot.data.mt5 import load_mt5
        df = load_mt5(symbol, settings.date_from, settings.date_to, settings.data_dir,
                      settings.mt5_server_tz)
    else:
        raise ValueError(f"Unknown DATA_SOURCE={settings.data_source!r} (use mt5, csv or dukascopy)")
    tz = df.index.tz
    if settings.date_from:
        df = df[df.index >= pd.Timestamp(settings.date_from, tz=tz)]
    if settings.date_to:
        df = df[df.index < pd.Timestamp(settings.date_to, tz=tz)]
    return df
