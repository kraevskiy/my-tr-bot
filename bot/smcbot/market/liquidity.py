"""Liquidity pools (L5, PRO tiers).

Tiers (PRO "Перед тим, як укласти угоду"):
  MAJOR  : PDH/PDL, PWH/PWL, PMH/PML
  MEDIUM : H1 swing highs/lows, Asia high/low (session liquidity, L27/L32)
  MINOR  : M15 swing highs/lows

A pool is a horizontal level with `known_at` (when it exists) and `taken_at`
(first time the M1 price traded beyond it). BSL = above highs (dir=+1),
SSL = below lows (dir=-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smcbot.config import NY_TZ
from smcbot.data.bars import TF_DELTA, from_trading_time, to_trading_time
from smcbot.market.sessions import daily_levels, monthly_levels, weekly_levels
from smcbot.market.structure import swing_levels

TIER_RANK = {"MAJOR": 3, "MEDIUM": 2, "MINOR": 1}


def _first_breach(m1_t: np.ndarray, m1_h: np.ndarray, m1_l: np.ndarray,
                  known: np.ndarray, level: np.ndarray, side: np.ndarray) -> np.ndarray:
    """For each pool, first M1 bar time after `known` where price trades beyond level."""
    out = np.full(len(level), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    n = len(m1_t)
    B = 1024
    nb = (n + B - 1) // B
    pad_h = np.full(nb * B, -np.inf)
    pad_l = np.full(nb * B, np.inf)
    pad_h[:n], pad_l[:n] = m1_h, m1_l
    bmax = pad_h.reshape(nb, B).max(axis=1)
    bmin = pad_l.reshape(nb, B).min(axis=1)
    start = np.searchsorted(m1_t, known, side="left")
    for k in range(len(level)):
        s = start[k]
        if s >= n:
            continue
        L, up = level[k], side[k] == 1
        b0 = s // B
        end0 = min((b0 + 1) * B, n)
        seg = m1_h[s:end0] > L if up else m1_l[s:end0] < L
        hit = np.flatnonzero(seg)
        if hit.size:
            out[k] = m1_t[s + hit[0]]
            continue
        blk = np.flatnonzero(bmax[b0 + 1:] > L) if up else np.flatnonzero(bmin[b0 + 1:] < L)
        if not blk.size:
            continue
        b = b0 + 1 + blk[0]
        seg = m1_h[b * B:min((b + 1) * B, n)] > L if up else m1_l[b * B:min((b + 1) * B, n)] < L
        out[k] = m1_t[b * B + np.flatnonzero(seg)[0]]
    return out


def _naive_utc(idx) -> np.ndarray:
    return pd.DatetimeIndex(idx).tz_convert("UTC").tz_localize(None).values.astype("datetime64[ns]")


def build_pools(m1: pd.DataFrame, tfs: dict, fractal_n: dict,
                asia_ny=("19:00", "02:00")) -> pd.DataFrame:
    rows = []
    # MAJOR: previous day / week / month extremes, known from the start of the period
    dl = daily_levels(m1, asia_ny=asia_ny)
    day_start = from_trading_time(pd.DatetimeIndex(dl.index))
    for lvl, d, name in (("pdh", 1, "PDH"), ("pdl", -1, "PDL")):
        for t, v in zip(day_start, dl[lvl].values):
            if not np.isnan(v):
                rows.append((t, v, d, "MAJOR", name))
    wl = weekly_levels(m1)
    wk_start = from_trading_time(pd.DatetimeIndex(wl.index))
    for lvl, d, name in (("pwh", 1, "PWH"), ("pwl", -1, "PWL")):
        for t, v in zip(wk_start, wl[lvl].values):
            if not np.isnan(v):
                rows.append((t, v, d, "MAJOR", name))
    ml = monthly_levels(m1)
    mo_start = from_trading_time(pd.DatetimeIndex(ml.index.to_timestamp()))
    for lvl, d, name in (("pmh", 1, "PMH"), ("pml", -1, "PML")):
        for t, v in zip(mo_start, ml[lvl].values):
            if not np.isnan(v):
                rows.append((t, v, d, "MAJOR", name))
    # MEDIUM: Asia range (known when Asia ends: 02:00 NY of the trading day)
    for t_day, ah, al in zip(day_start, dl["asia_high"].values, dl["asia_low"].values):
        asia_end = (t_day.tz_localize(None).normalize() + pd.Timedelta(days=1)
                    + pd.Timedelta(hours=int(asia_ny[1][:2]))).tz_localize(t_day.tz, nonexistent="shift_forward", ambiguous=False)
        if not np.isnan(ah):
            rows.append((asia_end, ah, 1, "MEDIUM", "ASIA_H"))
            rows.append((asia_end, al, -1, "MEDIUM", "ASIA_L"))
    # MEDIUM: London session range 02:00-05:00 NY (known at 05:00 NY). Used only when
    # cfg.session_pools is on (setups filter names starting with "LON" otherwise).
    idx_ny = m1.index.tz_convert(NY_TZ)
    mins = idx_ny.hour * 60 + idx_ny.minute
    lm = (mins >= 120) & (mins < 300)
    if lm.any():
        seg = pd.DataFrame({"h": m1["high"].values[lm], "l": m1["low"].values[lm]},
                           index=idx_ny[lm].normalize())
        g = seg.groupby(level=0).agg(h=("h", "max"), l=("l", "min"))
        for day, hh, ll in zip(g.index, g["h"].values, g["l"].values):
            t = day + pd.Timedelta(hours=5)
            rows.append((t, hh, 1, "MEDIUM", "LON_H"))
            rows.append((t, ll, -1, "MEDIUM", "LON_L"))
    # MEDIUM / MINOR swings
    for tf, tier in (("1h", "MEDIUM"), ("15min", "MINOR")):
        sw = swing_levels(tfs[tf], fractal_n[tf], TF_DELTA[tf])
        for t, k, p in zip(sw["known_at"], sw["kind"], sw["price"]):
            rows.append((t, p, 1 if k == "high" else -1, tier, f"{tf}_{'SH' if k == 'high' else 'SL'}"))

    pools = pd.DataFrame(rows, columns=["known_at", "level", "dir", "tier", "name"])
    pools = pools.sort_values("known_at").reset_index(drop=True)
    m1_t = _naive_utc(m1.index)
    taken = _first_breach(m1_t, m1["high"].values, m1["low"].values,
                          _naive_utc(pools["known_at"]), pools["level"].values,
                          pools["dir"].values)
    pools["known_ns"] = _naive_utc(pools["known_at"])
    pools["taken_ns"] = taken
    pools["rank"] = pools["tier"].map(TIER_RANK)
    return pools
