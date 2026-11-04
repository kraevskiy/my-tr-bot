"""Synthetic M1 data for testing the plumbing (NOT for judging the strategy).

Random walk with session-dependent volatility and occasional mean reversion so that
sweeps, FVGs and structure breaks occur. Weekend gap: Fri 17:00 -> Sun 17:00 NY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smcbot.config import NY_TZ


def make_m1(start="2024-01-07 17:00", days=60, seed=7, price=1.0850,
            spread_pips=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz=NY_TZ), periods=days * 1440, freq="1min")
    wd, hr = idx.weekday, idx.hour
    weekend = ((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 17))
    idx = idx[~weekend]
    hr = idx.hour
    vol = np.full(len(idx), 0.6)                      # pips per minute, Asia
    vol[(hr >= 2) & (hr < 5)] = 1.6                   # London open
    vol[(hr >= 8) & (hr < 11)] = 1.8                  # NY open
    vol[(hr >= 5) & (hr < 8)] = 1.1
    vol *= np.exp(rng.normal(0, 0.25, len(idx)).cumsum() * 0.002)
    steps = rng.standard_t(4, len(idx)) * vol * 0.0001 * 0.6
    drift = np.sin(np.arange(len(idx)) / (1440 * 3.0)) * 0.0000015
    close = price + np.cumsum(steps + drift)
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, 1, (2, len(idx)))) * vol * 0.0001 * 0.5
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "spread": spread_pips}, index=pd.DatetimeIndex(idx, name="time"))
