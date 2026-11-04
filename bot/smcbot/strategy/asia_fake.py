"""Setup AF "Asia fake-out at the London open" (added in v5; research name S2+).

Rules (all times New York; Kyiv = NY+7, NY+6 for a few weeks a year):
  1. Asia range = high/low of M5 bars 19:00-23:55 of the trading day (needs >= 80% of bars).
  2. 02:00-04:55: the first M5 close beyond the Asia high/low (previous close inside),
     breakout candle body >= af_body_k * ATR14(M5), in the breakout direction. One event per day per pair.
  3. Day range so far (from 17:00, before the breakout bar) >= af_adr_min * ATR14(D1).
  4. Within af_fade_bars M5 bars a close back inside the range whose candle body >= af_reclaim_body_k * ATR14(M5)
     -> market entry against the breakout at the next bar.
  5. Stop beyond the fake-out extreme + af_sl_buffer_pips (+ spread for shorts), at least af_min_sl_pips.
     Target af_tp_r R, no partial, close at cfg.close_time_ny (12:00).
Research (lab/hyp/E, G): 179 trades 2024-2026, +0.57R/trade with a 1.0/1.4 pip spread; train/valid/test
+0.41/+0.87/+0.67; expect ~+0.4R live. Monthly correlation with the v4 setup -0.28.
Everything here uses only bars closed before the signal time (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smcbot.config import NY_TZ, Config, Instrument

NY = NY_TZ


def _hm(s: str) -> int:
    h, m = (int(x) for x in s.split(":"))
    return h * 60 + m


def build_m5(m1: pd.DataFrame) -> pd.DataFrame:
    """M5 bars on the NY clock with the context columns the setup needs."""
    idx = m1.index.tz_convert(NY) if m1.index.tz is not None else m1.index.tz_localize("UTC").tz_convert(NY)
    x = m1.set_axis(idx)
    m5 = x[["open", "high", "low", "close"]].resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    i5 = m5.index
    m5["mod"] = i5.hour * 60 + i5.minute
    m5["tday"] = (i5 + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    d = m5.groupby("tday").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    tr_d = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    m5["atr_d"] = m5["tday"].map(tr_d.rolling(14).mean().shift(1))
    g = m5.groupby("tday")
    m5["day_hi_prev"] = g["high"].cummax().groupby(m5["tday"]).shift(1)
    m5["day_lo_prev"] = g["low"].cummin().groupby(m5["tday"]).shift(1)
    tr5 = pd.concat([m5.high - m5.low, (m5.high - m5.close.shift()).abs(),
                     (m5.low - m5.close.shift()).abs()], axis=1).max(axis=1)
    m5["atr5"] = tr5.rolling(14).mean()
    return m5


def asia_fake_signals(m1: pd.DataFrame, cfg: Config, inst: Instrument) -> pd.DataFrame:
    if m1 is None or m1.empty:
        return pd.DataFrame()
    m5 = build_m5(m1)
    a0, a1 = _hm(cfg.af_asia_start), _hm(cfg.af_asia_end)
    w0, w1 = _hm(cfg.af_win_start), _hm(cfg.af_win_end)
    mm = m5["mod"].values
    in_asia = (mm >= a0) | (mm < a1) if a0 > a1 else (mm >= a0) & (mm < a1)
    asia = m5[in_asia].groupby("tday").agg(hi=("high", "max"), lo=("low", "min"), nb=("high", "size"))
    need = int(((a1 - a0) % 1440) // 5 * 0.8)
    asia = asia[asia.nb >= need]
    hi = m5["tday"].map(asia.hi).values
    lo = m5["tday"].map(asia.lo).values
    O, H, L, C = (m5[k].values for k in ("open", "high", "low", "close"))
    atr5, atrd = m5["atr5"].values, m5["atr_d"].values
    td = m5["tday"].values
    pc = np.r_[np.nan, C[:-1]]
    same = np.r_[False, td[1:] == td[:-1]]
    body = C - O
    win = (mm >= w0) & (mm < w1)
    up = win & same & (C > hi) & (pc <= hi) & (body > 0) & (body >= cfg.af_body_k * atr5)
    dn = win & same & (C < lo) & (pc >= lo) & (body < 0) & (-body >= cfg.af_body_k * atr5)
    cand = np.where(up | dn)[0]
    pip, sp = inst.pip, inst.default_spread_pips * inst.pip
    seen, rows = set(), []
    tclose = m5.index + pd.Timedelta(minutes=5)
    for i in cand:
        if td[i] in seen:
            continue
        seen.add(td[i])                                   # only the first breakout of the day counts
        s = 1 if up[i] else -1
        level = hi[i] if s == 1 else lo[i]
        adr = (m5["day_hi_prev"].values[i] - m5["day_lo_prev"].values[i]) / atrd[i]
        if not np.isfinite(adr) or adr < cfg.af_adr_min:
            continue
        j = None
        for k in range(i + 1, min(i + 1 + cfg.af_fade_bars, len(m5))):
            if td[k] != td[i]:
                break
            if (s == 1 and C[k] < level) or (s == -1 and C[k] > level):
                j = k
                break
        if j is None or abs(C[j] - O[j]) < cfg.af_reclaim_body_k * atr5[j]:
            continue
        side = -s
        ext = H[i:j + 1].max() if s == 1 else L[i:j + 1].min()
        entry = C[j] + (sp if side == 1 else 0.0)
        stop = ext - cfg.af_sl_buffer_pips * pip if side == 1 else ext + cfg.af_sl_buffer_pips * pip + sp
        risk = (entry - stop) * side
        if risk < cfg.af_min_sl_pips * pip:
            stop = entry - side * cfg.af_min_sl_pips * pip
            risk = cfg.af_min_sl_pips * pip
        rows.append({
            "symbol": inst.symbol, "side": side, "setup": "AF", "signal_time": tclose[j],
            "order": "market", "entry": entry, "sl": stop, "tp": entry + side * cfg.af_tp_r * risk,
            "risk": risk, "risk_pips": risk / pip, "rr": cfg.af_tp_r, "pool": "ASIA_H" if s == 1 else "ASIA_L",
            "pool_level": level, "tier": "MEDIUM", "killzone": "LONDON", "grade": "AF",
            "risk_mult": cfg.af_risk_mult, "partial_r": 99.0, "be_at_r": cfg.af_be_at_r, "score": np.nan,
            "af_adr_used": adr, "af_break_time": tclose[i], "af_extreme": ext,
        })
    return pd.DataFrame(rows)
