"""How sessions relate to each other: Asia -> Frankfurt -> London -> New York.

One row per trading day (17:00 NY -> 17:00 NY) and symbol. Session windows use each market's
own clock, so daylight-saving mismatches between the US and Europe are handled:

  ASIA  19:00-02:00 New York (Tokyo/Sydney, the bot's Asia range)
  FRA   08:00-09:00 Frankfurt  (Frankfurt open, the hour before London)
  LON   08:00-12:00 London     (London morning, before New York)
  NY    08:00-12:00 New York   (NY AM, incl. 08:30 data)
  PM    12:00-17:00 New York   (NY afternoon to the daily close)

For each session: open, close, high, low, direction, range (pips), whether it traded through
the Asia high/low (and earlier sessions' high/low) and whether it closed back inside (sweep).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smcbot.config import NY_TZ

SESSIONS = (
    ("ASIA", NY_TZ, "19:00", "02:00"),
    ("FRA", "Europe/Berlin", "08:00", "09:00"),
    ("LON", "Europe/London", "08:00", "12:00"),
    ("NY", NY_TZ, "08:00", "12:00"),
    ("PM", NY_TZ, "12:00", "17:00"),
)


def _trading_day(idx_ny: pd.DatetimeIndex) -> pd.Index:
    return (idx_ny + pd.Timedelta(hours=7)).strftime("%Y-%m-%d")


def session_table(m1: pd.DataFrame, pip: float = 0.0001) -> pd.DataFrame:
    idx = m1.index.tz_convert(NY_TZ)
    day = _trading_day(idx)
    o, h, l, c = (m1[k].values for k in ("open", "high", "low", "close"))
    base = pd.DataFrame({"day": day, "o": o, "h": h, "l": l, "c": c}, index=idx)
    out = base.groupby("day").agg(d_open=("o", "first"), d_high=("h", "max"), d_low=("l", "min"),
                                  d_close=("c", "last"), n=("o", "size"))
    out = out[out["n"] > 600]                 # skip half days / data holes
    for name, tz, a, b in SESSIONS:
        loc = idx.tz_convert(tz)
        mins = loc.hour * 60 + loc.minute
        sa, sb = int(a[:2]) * 60 + int(a[3:]), int(b[:2]) * 60 + int(b[3:])
        mask = (mins >= sa) | (mins < sb) if sa > sb else (mins >= sa) & (mins < sb)
        seg = base[mask]
        g = seg.groupby("day")
        s = pd.DataFrame({"open": g["o"].first(), "close": g["c"].last(), "high": g["h"].max(),
                          "low": g["l"].min(),
                          "t_high": g["h"].idxmax(), "t_low": g["l"].idxmin()})
        s.columns = [f"{name}_{k}" for k in s.columns]
        out = out.join(s, how="left")
    out = out.dropna(subset=[f"{n}_open" for n, *_ in SESSIONS])
    for name, *_ in SESSIONS:
        mv = (out[f"{name}_close"] - out[f"{name}_open"]) / pip
        out[f"{name}_move"] = mv
        out[f"{name}_dir"] = np.sign(mv).astype(int)
        out[f"{name}_range"] = (out[f"{name}_high"] - out[f"{name}_low"]) / pip
    out["DAY_move"] = (out["d_close"] - out["d_open"]) / pip
    out["DAY_dir"] = np.sign(out["DAY_move"]).astype(int)
    out["DAY_range"] = (out["d_high"] - out["d_low"]) / pip
    # interaction with the Asia range (and London with Frankfurt, NY with London)
    for name, ref in (("FRA", "ASIA"), ("LON", "ASIA"), ("NY", "ASIA"), ("NY", "LON"), ("LON", "FRA"),
                      ("PM", "NY")):
        hi, lo = out[f"{ref}_high"], out[f"{ref}_low"]
        tk_h = out[f"{name}_high"] > hi
        tk_l = out[f"{name}_low"] < lo
        out[f"{name}_x_{ref}"] = np.select([tk_h & tk_l, tk_h, tk_l], ["both", "high", "low"], "none")
        cl = out[f"{name}_close"]
        # sweep = took one side and closed back inside the reference range
        out[f"{name}_sweep_{ref}"] = np.select([tk_h & ~tk_l & (cl < hi), tk_l & ~tk_h & (cl > lo)],
                                               ["high", "low"], "none")
    # which session made the day's high / low
    for ext, col in (("high", "d_high"), ("low", "d_low")):
        who = pd.Series("other", index=out.index)
        for name, *_ in SESSIONS[::-1]:
            hit = np.isclose(out[f"{name}_{ext}"], out[col])
            who = who.where(~hit, name)
        out[f"day_{ext}_by"] = who
    out["weekday"] = pd.to_datetime(out.index).day_name().str[:3]
    return out


def add_news(t: pd.DataFrame, cal: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flags per trading day: HIGH-impact USD release in NY AM, EUR/GBP in London morning, tier-1 tags."""
    from smcbot.market.news import PAIR_CCY
    t = t.copy()
    if cal is None or not len(cal):
        return t
    e = cal[cal["impact"] == "High"].copy()
    e["ny"] = e["time"].dt.tz_convert(NY_TZ)
    e["day"] = _trading_day(pd.DatetimeIndex(e["ny"]))
    e["hhmm"] = e["ny"].dt.hour * 60 + e["ny"].dt.minute
    other = [c for c in PAIR_CCY.get(symbol, ()) if c != "USD"]
    usd_am = e[(e["currency"] == "USD") & e["hhmm"].between(7 * 60, 11 * 60 + 59)]
    eu_lon = e[e["currency"].isin(other) & e["hhmm"].between(1 * 60, 7 * 60 + 59)]
    usd_pm = e[(e["currency"] == "USD") & e["hhmm"].between(12 * 60, 16 * 60 + 59)]   # FOMC 14:00
    t["news_usd_am"] = t.index.isin(set(usd_am["day"]))
    t["news_eu_london"] = t.index.isin(set(eu_lon["day"]))
    t["news_usd_pm"] = t.index.isin(set(usd_pm["day"]))
    tags = e[e["tag"] != ""].groupby("day")["tag"].agg(lambda x: "+".join(sorted(set(x))))
    t["news_tags"] = t.index.map(tags).fillna("")
    return t
