"""Economic calendar (ForexFactory) -> news context for signals and trades.

Source: ForexFactory calendar CSV (date,time,currency,impact,event,actual,forecast,previous),
times in UTC. A community mirror refreshed daily is downloaded by `python -m smcbot news`:
  https://github.com/janickfarrell/newfac/releases/download/calendar-data/forexfactory_calendar.csv
A trimmed copy (USD/EUR/GBP, Medium+High, 2023+) ships in news/ff_calendar.csv;
`smcbot news` writes a fresh one to $DATA_DIR/news/ff_calendar.csv, which takes priority.

What we compute per signal/trade (all in minutes, NaN when nothing within the horizon):
  news_prev_min / news_prev   - last HIGH event of the pair's currencies before the signal
  news_next_min / news_next   - next HIGH event after the signal
  news_in_trade               - HIGH events between fill and exit (trades only)
  news_day                    - tier-1 events of that NY day (NFP, CPI, FOMC, ECB, BOE, ...)
  news_bucket                 - pre_30 | post_0_30 | post_30_90 | day_tier1 | none

Filters (cfg.news_mode):
  off     - annotate only
  block   - drop signals with a HIGH event in [signal - news_before_min, signal + news_after_min]
            (with news_scope="tier1": only tier-1 events)
  after   - news-driven sweep: keep only signals whose sweep happened within news_after_min
            AFTER a HIGH event (the release created the liquidity grab)
  no_day  - skip whole NY days that have a tier-1 event
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NEWS_URL = "https://github.com/janickfarrell/newfac/releases/download/calendar-data/forexfactory_calendar.csv"
PAIR_CCY = {"EURUSD": ("USD", "EUR"), "GBPUSD": ("USD", "GBP"), "DXY": ("USD",)}
TIER1 = [
    ("NFP", r"^Non-Farm Employment Change"),
    ("CPI", r"^(Core )?CPI (m/m|y/y)"),
    ("FOMC", r"^(FOMC Statement|Federal Funds Rate|FOMC Press Conference)"),
    ("ECB", r"^(Main Refinancing Rate|ECB Press Conference|Monetary Policy Statement)"),
    ("BOE", r"^(Official Bank Rate|MPC Official Bank Rate Votes|BOE Monetary Policy Report)"),
    ("PCE", r"^Core PCE Price Index"),
    ("GDP", r"^(Advance |Prelim |Final )?GDP q/q"),
    ("RETAIL", r"^(Core )?Retail Sales m/m"),
    ("PMI", r"^(ISM (Manufacturing|Services) PMI|Flash (Manufacturing|Services) PMI)"),
]
HORIZON_MIN = 240


def tier1_tag(event: str) -> str:
    for tag, rx in TIER1:
        if re.search(rx, str(event)):
            return tag
    return ""


def load_calendar(path: str | Path, tz: str = "UTC", currencies=("USD", "EUR", "GBP"),
                  impacts=("High", "Medium", "Holiday")) -> pd.DataFrame:
    """Read a ForexFactory-style CSV; returns rows with tz-aware UTC `time`."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "datetime" in df.columns:                       # already-normalized file
        t = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    else:
        tm = df["time"].str.strip()
        hol = df["event"].str.contains("Holiday", case=False) & ~tm.str.match(r"^\d{1,2}:\d{2}")
        df.loc[hol, "impact"] = "Holiday"
        tm = tm.where(~hol, "00:00")
        ok = tm.str.match(r"^\d{1,2}:\d{2}")
        df = df[ok].copy()
        tm = tm[ok]
        t = pd.to_datetime(df["date"].str.strip() + " " + tm, format="%a %b %d %Y %H:%M", errors="coerce")
        # holidays are calendar days: keep the date itself (00:00 UTC) whatever tz the file uses
        t = pd.Series([x.tz_localize("UTC") if imp == "Holiday" else x.tz_localize(tz).tz_convert("UTC")
                                     for x, imp in zip(t, df["impact"])], index=df.index)
    df = df.assign(time=t).dropna(subset=["time"])
    df = df[df["currency"].isin(currencies) & df["impact"].isin(impacts)]
    df["tag"] = [tier1_tag(e) for e in df["event"]]
    cols = ["time", "currency", "impact", "event", "tag"] + [c for c in ("actual", "forecast", "previous") if c in df]
    return df[cols].sort_values("time").drop_duplicates(["time", "currency", "event"]).reset_index(drop=True)


def save_trimmed(cal: pd.DataFrame, path: str | Path, since: str = "2023-01-01") -> None:
    c = cal[cal["time"] >= pd.Timestamp(since, tz="UTC")].copy()
    c.insert(0, "datetime", c.pop("time").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(path, index=False)


def fetch_calendar(dest: str | Path, url: str = NEWS_URL, timeout: int = 120) -> Path:
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def _events_for(cal: pd.DataFrame, symbol: str, high_only=True, tier1_only=False):
    ccy = PAIR_CCY.get(symbol, (symbol[:3], symbol[3:6]))
    m = cal["currency"].isin(ccy)
    if high_only:
        m &= cal["impact"] == "High"
    if tier1_only:
        m &= cal["tag"] != ""
    e = cal[m]
    ns = e["time"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ns]")
    names = (e["currency"] + " " + e["event"]).values
    return ns, names, e["tag"].values


def _naive(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        s = s.dt.tz_localize("America/New_York")
    return s.dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ns]")


def annotate(df: pd.DataFrame, cal: pd.DataFrame, time_col: str = "signal_time") -> pd.DataFrame:
    """Add news context columns to signals/trades/candidates (needs `symbol` and `time_col`)."""
    if df is None or not len(df) or cal is None or not len(cal):
        return df
    out = df.copy()
    for c in ("news_prev_min", "news_next_min"):
        out[c] = np.nan
    for c in ("news_prev", "news_next", "news_in_trade", "news_day", "news_bucket"):
        out[c] = ""
    one = np.timedelta64(1, "m")
    for sym, g in out.groupby("symbol"):
        ns, names, tags = _events_for(cal, sym)
        t1 = tags != ""
        if not len(ns):
            continue
        t = _naive(g[time_col])
        k = np.searchsorted(ns, t, side="right")          # events strictly after t start at k
        prev_ok = k > 0
        prev_min = np.where(prev_ok, (t - ns[np.maximum(k - 1, 0)]) / one, np.nan)
        next_ok = k < len(ns)
        next_min = np.where(next_ok, (ns[np.minimum(k, len(ns) - 1)] - t) / one, np.nan)
        prev_min[prev_min > HORIZON_MIN] = np.nan
        next_min[next_min > HORIZON_MIN] = np.nan
        out.loc[g.index, "news_prev_min"] = prev_min
        out.loc[g.index, "news_next_min"] = next_min
        out.loc[g.index, "news_prev"] = [names[j - 1] if ok and not np.isnan(pm) else ""
                                         for j, ok, pm in zip(k, prev_ok, prev_min)]
        out.loc[g.index, "news_next"] = [names[j] if ok and not np.isnan(nm) else ""
                                         for j, ok, nm in zip(k, next_ok, next_min)]
        # tier-1 events on the same NY calendar day
        ev_day = pd.DatetimeIndex(ns[t1]).tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
        day_map = pd.Series(tags[t1]).groupby(ev_day).agg(lambda x: "+".join(sorted(set(x)))).to_dict()
        g_day = pd.DatetimeIndex(t).tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
        out.loc[g.index, "news_day"] = [day_map.get(d, "") for d in g_day]
        if "fill_time" in g and "exit_time" in g:
            ft, et = g["fill_time"], g["exit_time"]
            vals = []
            for a, b in zip(ft, et):
                if pd.isna(a) or pd.isna(b):
                    vals.append("")
                    continue
                a_, b_ = _naive(pd.Series([a]))[0], _naive(pd.Series([b]))[0]
                i0, i1 = np.searchsorted(ns, a_, "left"), np.searchsorted(ns, b_, "right")
                vals.append("; ".join(names[i0:i1]))
            out.loc[g.index, "news_in_trade"] = vals
    pm, nm = out["news_prev_min"], out["news_next_min"]
    b = np.where(nm <= 30, "pre_30",
        np.where(pm <= 30, "post_0_30",
        np.where(pm <= 90, "post_30_90",
        np.where(out["news_day"] != "", "day_tier1", "none"))))
    out["news_bucket"] = b
    H = holidays(cal)
    days = pd.to_datetime(out[time_col]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d") \
        if pd.to_datetime(out[time_col]).dt.tz is not None else pd.to_datetime(out[time_col]).dt.strftime("%Y-%m-%d")
    out["holiday"] = ["+".join(c for c in (sym[:3], sym[3:6]) if d in H.get(c, set()))
                      for d, sym in zip(days, out["symbol"])]
    return out


def news_filter(sigs: pd.DataFrame, cal: pd.DataFrame, cfg) -> tuple[pd.DataFrame, int]:
    """Apply cfg.news_mode to signals. Returns (kept, n_dropped)."""
    mode = getattr(cfg, "news_mode", "off")
    if mode == "off" or not len(sigs) or cal is None or not len(cal):
        return sigs, 0
    tier1 = getattr(cfg, "news_scope", "high") == "tier1"
    keep = np.ones(len(sigs), bool)
    for sym, g in sigs.groupby("symbol"):
        ns, _, tags = _events_for(cal, sym, tier1_only=tier1)
        pos = sigs.index.get_indexer(g.index)
        if not len(ns):
            if mode == "after":
                keep[pos] = False
            continue
        t = _naive(g["signal_time"])
        if mode == "block":
            lo = t - np.timedelta64(int(cfg.news_after_min), "m")     # event shortly before
            hi = t + np.timedelta64(int(cfg.news_before_min), "m")    # event shortly after
            hit = np.searchsorted(ns, hi, "right") - np.searchsorted(ns, lo, "left") > 0
            keep[pos] &= ~hit
        elif mode == "after":
            ext = g["ext_ns"].values.astype("datetime64[ns]") if "ext_ns" in g else t
            k = np.searchsorted(ns, ext, "right") - 1
            ok = k >= 0
            dt = np.where(ok, (ext - ns[np.maximum(k, 0)]) / np.timedelta64(1, "m"), np.inf)
            keep[pos] &= ok & (dt <= cfg.news_after_min)
        elif mode == "no_day":
            d_ev = set(pd.DatetimeIndex(ns).tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d"))
            d_sig = pd.DatetimeIndex(t).tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
            keep[pos] &= ~np.isin(np.array(d_sig), list(d_ev))
        else:
            raise ValueError(f"unknown news_mode {mode}")
    return sigs[keep].reset_index(drop=True), int((~keep).sum())


def load_default(path: str | Path | None = None) -> pd.DataFrame | None:
    """Calendar from NEWS_CSV / data/news/ff_calendar.csv; None if missing."""
    import os
    root = Path(__file__).resolve().parents[2]          # repo root (smcbot/market/news.py)
    cands = [path, os.getenv("NEWS_CSV"), Path(os.getenv("DATA_DIR", "data")) / "news" / "ff_calendar.csv",
             "data/news/ff_calendar.csv", root / "news" / "ff_calendar.csv"]
    for c in cands:
        if c and Path(c).exists():
            return load_calendar(c, tz=os.getenv("NEWS_TZ", "UTC"))
    return None


def holidays(cal: pd.DataFrame) -> dict[str, set]:
    """Bank-holiday dates (YYYY-MM-DD) per currency; EUR also counts German holidays."""
    if cal is None or not len(cal):
        return {}
    h = cal[cal["impact"] == "Holiday"]
    out = {}
    for c, g in h.groupby("currency"):
        ev = g["event"].astype(str)
        keep = (ev == "Bank Holiday") | (ev.str.startswith("German") if c == "EUR" else False)
        out[c] = set(g.loc[keep, "time"].dt.strftime("%Y-%m-%d"))
    return out


def holiday_filter(sigs: pd.DataFrame, cal: pd.DataFrame, cfg) -> tuple[pd.DataFrame, int]:
    """cfg.holiday_mode: us_ny | any | any_ny (see Config)."""
    mode = getattr(cfg, "holiday_mode", "off")
    if mode == "off" or not len(sigs) or cal is None:
        return sigs, 0
    H = holidays(cal)
    day = pd.to_datetime(sigs["signal_time"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    ny = sigs["killzone"].astype(str).eq("NY") if "killzone" in sigs else pd.Series(True, index=sigs.index)
    us = day.isin(H.get("USD", set()))
    anyh = pd.Series([d in H.get(sym[:3], set()) or d in H.get(sym[3:6], set())
                      for d, sym in zip(day, sigs["symbol"])], index=sigs.index)
    drop = {"us_ny": us & ny, "any": anyh, "any_ny": anyh & ny}[mode]
    return sigs[~drop].reset_index(drop=True), int(drop.sum())
