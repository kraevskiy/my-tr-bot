"""PD arrays: FVG (L9) and Order Blocks (L10).

Each zone records when it becomes known (`known_at`) and when it is invalidated
(`invalid_at`), so consumers can query "active zones at time t" without look-ahead.

FVG bull: low[i+1] > high[i-1] -> zone [high[i-1], low[i+1]]; invalid on a CLOSE
          below the zone low (price passed the whole zone with bodies).
OB bull : bearish candle k that tests a key level (local sweep or active bull FVG),
          engulfed by a CLOSE above open[k] within M bars, with an FVG in the move.
          Levels: open (FX default entry), mt = 50% of body, wick = low[k].
          Invalid on a CLOSE below MT (HTF only, M15+).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FAR_FUTURE = np.datetime64("2262-01-01T00:00:00", "ns")


def _utc_ns(s: pd.Series, fill_max: bool = False) -> np.ndarray:
    """tz-aware timestamps -> naive UTC datetime64[ns]; NaT -> far future if fill_max."""
    ts = pd.to_datetime(s, utc=True).dt.tz_localize(None)
    arr = ts.values.astype("datetime64[ns]")
    if fill_max:
        arr = np.where(np.isnat(arr), FAR_FUTURE, arr)
    return arr


def find_fvgs(df: pd.DataFrame, bar: pd.Timedelta, min_size: float) -> pd.DataFrame:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    idx = df.index
    rows = []
    for i in range(1, len(df) - 1):
        if l[i + 1] - h[i - 1] >= min_size:
            rows.append((1, h[i - 1], l[i + 1], i))
        if l[i - 1] - h[i + 1] >= min_size:
            rows.append((-1, h[i + 1], l[i - 1], i))
    out = []
    for d, lo, hi, i in rows:
        inv = pd.NaT
        rest = c[i + 2:]
        hit = np.flatnonzero(rest < lo) if d == 1 else np.flatnonzero(rest > hi)
        if hit.size:
            inv = idx[i + 2 + hit[0]] + bar
        out.append({"type": "FVG", "dir": d, "lo": lo, "hi": hi, "mid": (lo + hi) / 2,
                    "time": idx[i], "known_at": idx[i + 1] + bar, "invalid_at": inv})
    return pd.DataFrame(out)


def find_obs(df: pd.DataFrame, bar: pd.Timedelta, fvgs: pd.DataFrame,
             max_engulf: int = 3, sweep_lookback: int = 10,
             use_mt_invalidation: bool = True) -> pd.DataFrame:
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    idx = df.index
    size = len(df)
    if len(fvgs):
        f_dir = fvgs["dir"].values
        f_lo, f_hi = fvgs["lo"].values, fvgs["hi"].values
        f_known = _utc_ns(fvgs["known_at"])
        f_inv = _utc_ns(fvgs["invalid_at"], fill_max=True)
    t_utc = idx.tz_convert("UTC").tz_localize(None).values.astype("datetime64[ns]")
    out = []
    for k in range(sweep_lookback, size - 1):
        for d in (1, -1):
            if d == 1 and not c[k] < o[k]:
                continue
            if d == -1 and not c[k] > o[k]:
                continue
            # key level test: local sweep of the recent extreme, or tap of an active FVG
            if d == 1:
                swept = l[k] < l[k - sweep_lookback:k].min()
            else:
                swept = h[k] > h[k - sweep_lookback:k].max()
            tapped = False
            if not swept and len(fvgs):
                t = t_utc[k]
                act = (f_dir == d) & (f_known <= t) & (f_inv > t)
                if d == 1:
                    tapped = bool((act & (l[k] <= f_hi) & (l[k] >= f_lo)).any())
                else:
                    tapped = bool((act & (h[k] >= f_lo) & (h[k] <= f_hi)).any())
            if not (swept or tapped):
                continue
            # engulfing close beyond the OB open within max_engulf bars
            j_end = min(size, k + 1 + max_engulf)
            eng = None
            for j in range(k + 1, j_end):
                if (d == 1 and c[j] > o[k]) or (d == -1 and c[j] < o[k]):
                    eng = j
                    break
            if eng is None:
                continue
            # impulse proof: an FVG in the move (PROPOSAL)
            seg_l, seg_h = l[k:eng + 2], h[k:eng + 2]
            impulse = False
            for m in range(1, len(seg_l) - 1):
                if (d == 1 and seg_l[m + 1] > seg_h[m - 1]) or (d == -1 and seg_h[m + 1] < seg_l[m - 1]):
                    impulse = True
                    break
            if not impulse:
                continue
            mt = (o[k] + c[k]) / 2
            lo, hi = (l[k], o[k]) if d == 1 else (o[k], h[k])
            inv = pd.NaT
            if use_mt_invalidation:
                rest = c[eng + 1:]
                hit = np.flatnonzero(rest < mt) if d == 1 else np.flatnonzero(rest > mt)
                if hit.size:
                    inv = idx[eng + 1 + hit[0]] + bar
            out.append({"type": "OB", "dir": d, "lo": lo, "hi": hi, "mid": mt,
                        "open_lvl": o[k], "wick": l[k] if d == 1 else h[k],
                        "body_ratio": abs(c[k] - o[k]) / max(h[k] - l[k], 1e-12),
                        "swept": bool(swept), "time": idx[k],
                        "known_at": idx[eng] + bar, "invalid_at": inv})
    return pd.DataFrame(out)


def active_at(zones: pd.DataFrame, t: pd.Timestamp, d: int | None = None) -> pd.DataFrame:
    if zones is None or not len(zones):
        return zones
    m = (zones["known_at"] <= t) & (zones["invalid_at"].isna() | (zones["invalid_at"] > t))
    if d is not None:
        m &= zones["dir"] == d
    return zones[m]
