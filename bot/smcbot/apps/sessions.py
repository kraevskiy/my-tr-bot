"""Session relationship study (Asia -> Frankfurt -> London -> NY) + news, per trading day.

    python -m smcbot sessions            # uses DATA_SOURCE / CSV_* like the backtest
Writes results/sessions_<symbol>.csv (one row per day) and prints the main tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from smcbot.market import news as newsmod
from smcbot.data.providers import load_symbol
from smcbot.research.sessions import add_news, session_table
from smcbot.settings import load_settings


def pct(x):
    return round(100 * float(np.mean(x)))


def main(argv=None) -> int:
    s, _ = load_settings()
    cal = newsmod.load_default()
    out = Path(s.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sym in s.symbols:
        pip = 0.01 if sym.endswith("JPY") else 0.0001
        t = add_news(session_table(load_symbol(sym, s), pip), cal, sym)
        t.to_csv(out / f"sessions_{sym}.csv")
        print(f"\n===== {sym}: {len(t)} days =====")
        for a, b in (("ASIA", "FRA"), ("ASIA", "LON"), ("LON", "NY"), ("NY", "PM")):
            print(f"{a}->{b} same direction: {pct(t[f'{a}_dir'] == t[f'{b}_dir'])}%")
        print("day high made by:", t["day_high_by"].value_counts(normalize=True).mul(100).round(0).to_dict())
        print("day low made by: ", t["day_low_by"].value_counts(normalize=True).mul(100).round(0).to_dict())
        fwd = (t["d_close"] - t["FRA_close"]) / pip
        print("after Frankfurt -> day close, avg pips by FRA vs Asia:",
              fwd.groupby(t["FRA_x_ASIA"]).mean().round(1).to_dict())
        if "news_tags" in t:
            nfp = t["news_tags"].str.contains("NFP")
            print(f"NY takes both London sides: NFP {pct(t.loc[nfp, 'NY_x_LON'] == 'both')}% vs other "
                  f"{pct(t.loc[~nfp, 'NY_x_LON'] == 'both')}%")
    return 0

