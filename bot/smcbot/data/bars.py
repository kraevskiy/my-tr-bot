"""Market data: loading MT5 exports and building higher timeframes.

Conventions
-----------
* Bars are indexed by their OPEN time, tz-aware America/New_York.
* Prices are BID (MT5 bars are bid); `spread` column holds the spread in pips.
* "Trading time" = NY time + 7h (naive). In trading time the forex day starts at
  00:00 (= 17:00 NY = DO, 00:00 Kyiv for most of the year) and H4 candles start at
  00/04/08/12/16/20, exactly like MT5 servers that run on GMT+2/+3 with US DST.
* A higher-timeframe bar may only be used after its CLOSE time (open + tf).
"""
from __future__ import annotations

import pandas as pd

from smcbot.config import NY_TZ

TRADING_OFFSET = pd.Timedelta(hours=7)

TF_DELTA = {
    "1min": pd.Timedelta(minutes=1), "3min": pd.Timedelta(minutes=3),
    "5min": pd.Timedelta(minutes=5), "15min": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4),
    "1D": pd.Timedelta(days=1), "1W": pd.Timedelta(days=7),
}


def server_to_ny(ts: pd.Series, server_tz: str) -> pd.Series:
    """MT5 server wall-clock times (naive) -> tz-aware New York times.

    'ny_close': server clock = NY + 7h (GMT+2/+3 following US DST, the usual MT5 setup).
    """
    ts = pd.Series(pd.to_datetime(ts))
    if server_tz == "ny_close":
        return (ts - TRADING_OFFSET).dt.tz_localize(NY_TZ, ambiguous="NaT",
                                                    nonexistent="shift_forward")
    return ts.dt.tz_localize(server_tz, ambiguous="NaT",
                             nonexistent="shift_forward").dt.tz_convert(NY_TZ)


def load_mt5_csv(path: str, server_tz: str = "ny_close", point: float = 0.00001,
                 pip: float = 0.0001) -> pd.DataFrame:
    """Load an MT5 'Export bars' file (tab separated, <DATE> <TIME> <OPEN> ...).

    server_tz: 'ny_close' -> server clock = NY + 7h (typical GMT+2/+3 brokers);
               otherwise an IANA tz name, e.g. 'UTC' or 'Europe/Kyiv'.
    """
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if head.startswith((b"\xff\xfe", b"\xfe\xff")) or head[1:200:2].count(b"\x00") > 20:
        enc = "utf-16"
    elif head.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    raw = pd.read_csv(path, sep=None, engine="python", encoding=enc)
    raw.columns = [c.strip("<>").lower() for c in raw.columns]
    ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str),
                        format="mixed")
    idx = server_to_ny(ts, server_tz)
    df = pd.DataFrame({
        "open": raw["open"].astype(float).values,
        "high": raw["high"].astype(float).values,
        "low": raw["low"].astype(float).values,
        "close": raw["close"].astype(float).values,
    }, index=pd.DatetimeIndex(idx, name="time"))
    if "spread" in raw.columns:
        df["spread"] = raw["spread"].astype(float).values * point / pip
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def to_trading_time(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(NY_TZ).tz_localize(None) + TRADING_OFFSET


def from_trading_time(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (idx - TRADING_OFFSET).tz_localize(NY_TZ, ambiguous="NaT",
                                              nonexistent="shift_forward")


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate M1 bars into tf, aligned to the forex trading day (17:00 NY)."""
    if tf == "1min":
        return m1.copy()
    t = m1.copy()
    t.index = to_trading_time(m1.index)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "spread" in t.columns:
        agg["spread"] = "median"
    if tf == "1W":
        key = t.index.normalize() - pd.to_timedelta(t.index.weekday, unit="D")
        out = t.groupby(key).agg(agg)
    else:
        out = t.resample(tf, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open"])
    out.index = from_trading_time(pd.DatetimeIndex(out.index))
    out = out[~out.index.isna()]
    out.index.name = "time"
    return out


def drop_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the weekend gap: Fri 17:00 -> Sun 17:00 NY."""
    wd, hr = df.index.weekday, df.index.hour
    weekend = ((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 17))
    return df[~weekend]


def build_timeframes(m1: pd.DataFrame, tfs) -> dict[str, pd.DataFrame]:
    out = {"1min": m1}
    for tf in tfs:
        if tf not in out:
            out[tf] = resample(m1, tf)
    return out
