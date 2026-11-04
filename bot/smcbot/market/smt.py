"""SMT divergence between correlated pairs (course L28), used as annotation or filter.

At the sweep extreme of a signal on pair A, look at pair B (EURUSD <-> GBPUSD):
  bullish SMT (long): A swept a low, B did NOT trade below its own low of the prior window
  bearish SMT (short): A swept a high, B did NOT trade above its prior high
Windows (PROPOSAL): prior = [t_s - 3h, t_s - 15m), around = [t_s - 15m, t_s + 15m].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIRS = {"EURUSD": "GBPUSD", "GBPUSD": "EURUSD"}


def smt_flags(df: pd.DataFrame, data: dict, ext_col: str = "ext_ns",
              prior=pd.Timedelta(hours=3), near=pd.Timedelta(minutes=15)) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    if df is None or not len(df) or ext_col not in df:
        return out
    for sym, g in df.groupby("symbol"):
        other = PAIRS.get(sym)
        if other not in data:
            continue
        m1 = data[other]
        idx = m1.index
        hi, lo = m1["high"].values, m1["low"].values
        for ix, r in g.iterrows():
            if pd.isna(r.get(ext_col)):
                continue
            t = pd.Timestamp(r[ext_col]).tz_localize("UTC") if pd.Timestamp(r[ext_col]).tzinfo is None \
                else pd.Timestamp(r[ext_col])
            a, b, c = idx.searchsorted(t - prior), idx.searchsorted(t - near), idx.searchsorted(t + near)
            if b <= a or c <= b:
                continue
            if int(r["side"]) == 1:
                out[ix] = float(lo[b:c].min() > lo[a:b].min())
            else:
                out[ix] = float(hi[b:c].max() < hi[a:b].max())
    return out
