"""Sessions, killzones and opening levels, all computed in New York time.

Kyiv times from the course map to NY as: Kyiv = NY + 7h for ~48 weeks a year.
Computing in NY time keeps US-session events (08:30 NY news, 00:00 NY midnight)
correct through DST changes.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from smcbot.data.bars import to_trading_time


def _t(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def in_window(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Boolean mask: NY wall-clock time in [start, end). Handles windows across midnight."""
    tod = idx.hour * 60 + idx.minute
    s, e = _t(start), _t(end)
    s_m, e_m = s.hour * 60 + s.minute, e.hour * 60 + e.minute
    if s_m <= e_m:
        return np.asarray((tod >= s_m) & (tod < e_m))
    return np.asarray((tod >= s_m) | (tod < e_m))


def killzone_labels(idx: pd.DatetimeIndex, killzones) -> np.ndarray:
    lab = np.array([""] * len(idx), dtype=object)
    for name, s, e in killzones:
        lab[in_window(idx, s, e)] = name
    return lab


def trading_day(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Forex trading date (day starts 17:00 NY)."""
    return to_trading_time(idx).normalize()


def daily_levels(m1: pd.DataFrame, asia_ny=("19:00", "02:00"),
                 cbdr_ny=("16:00", "20:00")) -> pd.DataFrame:
    """Per trading day: DO, NYM, Asia high/low, CBDR high/low, PDH/PDL.

    Values are attached to the trading day they are USED in. Asia/CBDR of a day are
    complete at 02:00 / 20:00 NY; callers must respect `asia_end`/`cbdr_end`.
    """
    td = trading_day(m1.index)
    g = m1.groupby(td)
    out = pd.DataFrame({"do": g["open"].first(), "dh": g["high"].max(),
                        "dl": g["low"].min()})
    out["pdh"] = out["dh"].shift(1)
    out["pdl"] = out["dl"].shift(1)

    nym_mask = (m1.index.hour == 0) & (m1.index.minute == 0)
    nym = pd.Series(m1["open"].values[nym_mask], index=td[nym_mask])
    out["nym"] = nym[~nym.index.duplicated()].reindex(out.index)

    a = in_window(m1.index, *asia_ny)
    ag = m1[a].groupby(td[a])
    out["asia_high"] = ag["high"].max().reindex(out.index)
    out["asia_low"] = ag["low"].min().reindex(out.index)

    # CBDR 16:00-20:00 NY belongs to the NEXT trading day (it starts at 17:00 NY);
    # the 16:00-17:00 hour is the tail of the previous day -> group by end time.
    c = in_window(m1.index, *cbdr_ny)
    cdays = trading_day(m1.index[c] + pd.Timedelta(hours=4))
    cg = m1[c].groupby(cdays)
    out["cbdr_high"] = cg["high"].max().reindex(out.index)
    out["cbdr_low"] = cg["low"].min().reindex(out.index)
    return out


def weekly_levels(m1: pd.DataFrame) -> pd.DataFrame:
    tt = to_trading_time(m1.index)
    wk = tt.normalize() - pd.to_timedelta(tt.weekday, unit="D")
    g = m1.groupby(wk)
    out = pd.DataFrame({"wo": g["open"].first(), "wh": g["high"].max(),
                        "wl": g["low"].min()})
    out["pwh"] = out["wh"].shift(1)
    out["pwl"] = out["wl"].shift(1)
    return out


def monthly_levels(m1: pd.DataFrame) -> pd.DataFrame:
    tt = to_trading_time(m1.index)
    mk = tt.to_period("M")
    g = m1.groupby(mk)
    out = pd.DataFrame({"mo": g["open"].first(), "mh": g["high"].max(),
                        "ml": g["low"].min()})
    out["pmh"] = out["mh"].shift(1)
    out["pml"] = out["ml"].shift(1)
    return out
