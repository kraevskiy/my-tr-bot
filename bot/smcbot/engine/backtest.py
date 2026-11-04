"""Trade simulation on M1 bid bars with spread, PRO risk rules and BE variants.

Fill model (conservative):
  long : limit fills when ask low (bid low + spread) <= entry; SL on bid low; TP on bid high
  short: limit fills when bid high >= entry; SL on ask high; TP on ask low
  If the stop and a profit level are both touched inside one M1 bar, the stop wins.
Order expiry: end of the killzone the signal belongs to; also cancelled if price
reaches TP before filling.

Management after the partial (PRO: 50% at 1R), variants from Q5:
  a: SL -> entry right away
  b: SL untouched (BE is reached by the partial itself)
  c: SL -> entry once the first M5 higher low (lower high for shorts) is confirmed
  d: trail SL behind every new confirmed M5 swing
Leftovers are closed at cfg.close_time_ny (12:00 NY = 19:00 Kyiv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smcbot.config import Config, Instrument
from smcbot.data.bars import TF_DELTA, resample
from smcbot.market.structure import swing_levels


def _kz_end(t: pd.Timestamp, killzones) -> pd.Timestamp:
    tod = t.hour * 60 + t.minute
    for _, s, e in killzones:
        sm = int(s[:2]) * 60 + int(s[3:])
        em = int(e[:2]) * 60 + int(e[3:])
        if sm <= tod < em:
            return t.normalize() + pd.Timedelta(minutes=em)
    return t


def _shadow(d, sl, o, h, l, spr, ts, t0, hours):
    """Outcome of a market entry at the signal (for signals that never filled):
    MFE/MAE in R within `hours`, and a simple +2R / -1R race."""
    entry = o[0] + (spr[0] if d == 1 else 0.0)
    R = (entry - sl) * d
    if R <= 0:
        return {"shadow_R2": np.nan, "shadow_mfe_R": np.nan, "shadow_mae_R": np.nan}
    end = np.searchsorted(ts.values, (t0 + pd.Timedelta(hours=hours)).to_datetime64(), side="left") \
        if hasattr(ts, "values") else len(o)
    end = max(1, min(end, len(o)))
    fav = (h[:end] - entry) if d == 1 else (entry - (l[:end] + spr[:end]))
    adv = (entry - l[:end]) if d == 1 else ((h[:end] + spr[:end]) - entry)
    mfe, mae = fav.max() / R, adv.max() / R
    r2 = 0.0
    for k in range(end):
        if adv[k] >= R:
            r2 = -1.0
            break
        if fav[k] >= 2 * R:
            r2 = 2.0
            break
    return {"shadow_R2": r2, "shadow_mfe_R": mfe, "shadow_mae_R": mae}


def simulate_trade(sig: pd.Series, m1: pd.DataFrame, m5_swings: pd.DataFrame,
                   cfg: Config, inst: Instrument, mode: str) -> dict:
    d = int(sig["side"])
    entry, sl0, tp = float(sig["entry"]), float(sig["sl"]), float(sig["tp"])
    t0 = sig["signal_time"]
    market = str(sig.get("order", "limit")) == "market"
    kzs = cfg.killzones() if hasattr(cfg, "killzones") else cfg.killzones_ny
    expiry = t0 + pd.Timedelta(minutes=cfg.order_expiry_min) if getattr(cfg, "order_expiry_min", 0) \
        else _kz_end(t0, kzs)
    if expiry <= t0:
        expiry = t0 + pd.Timedelta(minutes=60)
    ch, cm = (int(x) for x in cfg.close_time_ny.split(":"))
    close_at = t0.normalize() + pd.Timedelta(hours=ch, minutes=cm)
    if t0.hour >= 17:                       # evening (Asia) signal: belongs to the next trading day
        close_at += pd.Timedelta(days=1)
    if close_at <= t0 and t0.hour >= ch:
        return {"status": "late"}           # signal at/after the daily close time: no trade
    if close_at <= t0:
        close_at = t0 + pd.Timedelta(hours=2)
    partial_on = getattr(cfg, "partial_enabled", True)
    hold_min, hold_end = getattr(cfg, "max_hold_min", 0), None
    slip = getattr(cfg, "slippage_pips", 0.0) * inst.pip      # extra cost on stop and market/time exits
    thru = getattr(cfg, "fill_through_pips", 0.0) * inst.pip   # stress test: limit must trade through the level
    lock_r = getattr(cfg, "lock_r_after_partial", 0.0)        # after the partial move the stop to +X R
    inval = float(sig.get("inval", np.nan)) if pd.notna(sig.get("inval", np.nan)) else np.nan
    inval_on = bool(getattr(cfg, "exit_on_fvg_fail", False)) and not np.isnan(inval)
    inval_buf = getattr(cfg, "fvg_fail_buffer_pips", 0.0) * inst.pip
    p_r = float(sig.get("partial_r", cfg.partial_at_r)) if pd.notna(sig.get("partial_r", np.nan)) else cfg.partial_at_r

    seg = m1.loc[t0:]
    if seg.empty:
        return {"status": "no_data"}
    ts = seg.index
    o, h, l, c = (seg[k].values for k in ("open", "high", "low", "close"))
    spr = (seg["spread"].values if "spread" in seg else np.full(len(seg), inst.default_spread_pips)) * inst.pip

    if market:
        entry = o[0] + (spr[0] if d == 1 else 0.0) + d * slip
        tp = entry + (tp - float(sig["entry"]))      # keep the planned distance to target
    R = (entry - sl0) * d
    if R <= 0:
        return {"status": "invalid"}

    kind = "low" if d == 1 else "high"
    sw = m5_swings[m5_swings["kind"] == kind]
    sw_known, sw_price, sw_time = sw["known_at"].values, sw["price"].values, sw["time"].values

    filled, fill_t = market, (ts[0] if market else None)
    sl = sl0
    rem, realized = 1.0, 0.0
    partial_t = None
    exits = []
    mfe = mae = 0.0
    for k in range(len(ts)):
        t = ts[k]
        bid_h, bid_l = h[k], l[k]
        ask_h, ask_l = h[k] + spr[k], l[k] + spr[k]
        if not filled:
            if t >= expiry:
                res = {"status": "expired"}
                res.update(_shadow(d, sl0, o, h, l, spr, ts, t0, cfg.shadow_hours))
                return res
            hit_tp_first = (bid_h >= tp) if d == 1 else (ask_l <= tp)
            touch = (ask_l <= entry - thru) if d == 1 else (bid_h >= entry + thru)
            if hit_tp_first:          # TP reached before (or in the same bar as) the fill
                res = {"status": "missed"}
                res.update(_shadow(d, sl0, o, h, l, spr, ts, t0, cfg.shadow_hours))
                return res
            if touch:
                filled, fill_t = True, t
            else:
                continue
        if hold_min and hold_end is None:
            hold_end = fill_t + pd.Timedelta(minutes=hold_min)
        fav = bid_h if d == 1 else ask_l
        adv = bid_l if d == 1 else ask_h
        if t >= close_at or (hold_end is not None and t >= hold_end):
            px = (o[k] if d == 1 else o[k] + spr[k]) - d * slip
            realized += rem * (px - entry) * d / R
            exits.append(("TIME" if t >= close_at else "HOLD", t, px, rem))
            rem = 0.0
            break
        mae = max(mae, (entry - adv) * d / R)
        stop_hit = (bid_l <= sl) if d == 1 else (ask_h >= sl)
        if inval_on and not stop_hit and partial_t is None and (t.minute + 1) % 5 == 0:
            cl = c[k] if d == 1 else c[k] + spr[k]
            if (cl - inval) * d < -inval_buf:          # M5 close beyond the FVG: the entry zone failed
                realized += rem * ((cl - entry) * d - slip) / R
                exits.append(("FVG_FAIL", t, cl, rem))
                rem = 0.0
                break
        if stop_hit:
            realized += rem * ((sl - entry) * d - slip) / R
            exits.append(("SL" if partial_t is None else "SL_AFTER_PARTIAL", t, sl, rem))
            rem = 0.0
            break
        mfe = max(mfe, (fav - entry) * d / R)
        if partial_on and partial_t is None and (fav - entry) * d >= p_r * R:
            px = entry + d * p_r * R
            realized += cfg.partial_frac * p_r
            rem -= cfg.partial_frac
            partial_t = t
            exits.append(("PARTIAL", t, px, cfg.partial_frac))
            if mode == "a":
                sl = entry
            if lock_r:
                sl = entry + d * lock_r * R if (lock_r * R - (sl - entry) * d) > 0 else sl
        if (fav - tp) * d >= 0:
            realized += rem * (tp - entry) * d / R
            exits.append(("TP", t, tp, rem))
            rem = 0.0
            break
        if partial_t is not None and mode in ("c", "d"):
            # M5 swings that formed after the partial and are confirmed by now
            new = (sw_known <= t.to_datetime64()) & (sw_time > partial_t.to_datetime64())
            if new.any():
                lvl = sw_price[new][-1]
                if mode == "c":
                    if (lvl - entry) * d > 0 and (entry - sl) * d > 0:
                        sl = entry
                else:
                    cand = lvl - d * (cfg.sl_buffer_pips * inst.pip + spr[k])
                    if (cand - sl) * d > 0:
                        sl = cand
    if not filled:
        res = {"status": "expired"}
        res.update(_shadow(d, sl0, o, h, l, spr, ts, t0, cfg.shadow_hours))
        return res
    if rem > 0:  # data ended
        realized += rem * (c[-1] - entry) * d / R
        exits.append(("EOD", ts[-1], c[-1], rem))
    last = exits[-1]
    return {"status": "filled", "fill_time": fill_t, "exit_time": last[1], "exit_kind": last[0],
            "R_result": realized, "partial": partial_t is not None, "fill_price": entry,
            "mfe_R": mfe, "mae_R": mae,
            "fill_wait_min": (fill_t - t0).total_seconds() / 60,
            "duration_min": (last[1] - fill_t).total_seconds() / 60}


def forward_outcomes(cands: pd.DataFrame, m1_by_symbol: dict, instruments: dict,
                     hours: float = 4.0) -> pd.DataFrame:
    """For every logged candidate with a stop: what a market entry at the signal would have
    done (+2R/-1R race, MFE/MAE in R within `hours`). Lets us judge filters offline."""
    if cands is None or not len(cands):
        return cands
    out = cands.copy()
    for col in ("fwd_R2", "fwd_mfe_R", "fwd_mae_R"):
        out[col] = np.nan
    for sym, g in out.groupby("symbol"):
        m1 = m1_by_symbol[sym]
        idx = m1.index
        o, h, l = m1["open"].values, m1["high"].values, m1["low"].values
        inst = instruments[sym]
        spr = (m1["spread"].values if "spread" in m1 else np.full(len(m1), inst.default_spread_pips)) * inst.pip
        span = int(hours * 60) + 1
        for ix, r in g.iterrows():
            if pd.isna(r.get("sl")) or pd.isna(r.get("signal_time")):
                continue
            t0 = pd.Timestamp(r["signal_time"])
            k = idx.searchsorted(t0)
            if k >= len(idx):
                continue
            sl_ = slice(k, min(k + span, len(idx)))
            res = _shadow(int(r["side"]), float(r["sl"]), o[sl_], h[sl_], l[sl_], spr[sl_],
                          idx[sl_], t0, hours)
            out.loc[ix, ["fwd_R2", "fwd_mfe_R", "fwd_mae_R"]] = [res["shadow_R2"], res["shadow_mfe_R"], res["shadow_mae_R"]]
    return out


def m5_swings_for(m1_by_symbol: dict, cfg: Config) -> dict:
    swings = {}
    for sym, m1 in m1_by_symbol.items():
        m5 = resample(m1, "5min")
        sw = swing_levels(m5, cfg.fractal_n["5min"], TF_DELTA["5min"])
        sw["known_at"] = pd.to_datetime(sw["known_at"]).dt.tz_convert("UTC").dt.tz_localize(None).values
        sw["time"] = pd.to_datetime(sw["time"]).dt.tz_convert("UTC").dt.tz_localize(None).values
        swings[sym] = sw
    return swings


def run_backtest(signals: pd.DataFrame, m1_by_symbol: dict, cfg: Config,
                 instruments: dict, mode: str = "b", swings: dict | None = None) -> pd.DataFrame:
    if signals is None or not len(signals):
        return pd.DataFrame()
    swings = swings or m5_swings_for(m1_by_symbol, cfg)

    sigs = signals.sort_values("signal_time").reset_index(drop=True)
    balance = cfg.starting_balance
    trades = []
    open_until = {}            # symbol -> exit time of the open position
    open_dir = {}
    day_stats = {}             # trading day -> dict(loss_R, sl_count, place_losses{}, stopped)

    def log_blocked(s, day, status):
        """Record a signal skipped by a risk rule, with the R it would have made (simulate_trade is pure)."""
        sym = s["symbol"]
        res = simulate_trade(s, m1_by_symbol[sym], swings[sym], cfg, instruments[sym], mode)
        rec = s.to_dict()
        rec.update(status=status, day=day, mode=mode,
                   shadow_R=res["R_result"] if res["status"] == "filled" else np.nan)
        trades.append(rec)

    for _, s in sigs.iterrows():
        sym = s["symbol"]
        t = s["signal_time"]
        day = (t.tz_localize(None) + pd.Timedelta(hours=7)).normalize()
        ds = day_stats.setdefault(day, {"loss_R": 0.0, "sl": 0, "place": {}, "stopped": False})
        # realize stats of trades closed before t
        for tr in trades:
            if not tr.get("_counted") and tr["status"] == "filled" and tr["exit_time"] <= t:
                tr["_counted"] = True
                dd = day_stats.setdefault(tr["day"], {"loss_R": 0.0, "sl": 0, "place": {}, "stopped": False})
                if tr["R_result"] < 0:
                    dd["loss_R"] += -tr["R_result"] * float(tr.get("risk_mult", 1.0) or 1.0)
                    dd["sl"] += 1
                    key = (tr["symbol"], round(tr["pool_level"], 4))
                    dd["place"][key] = dd["place"].get(key, 0) + 1
                    if dd["place"][key] >= cfg.max_losses_same_place:
                        dd["stopped"] = True
                if dd["loss_R"] * cfg.risk_pct >= cfg.daily_max_loss_pct - 1e-12 or dd["sl"] >= cfg.daily_max_sl:
                    dd["stopped"] = True
        if ds["stopped"]:
            log_blocked(s, day, "blocked_daily")
            continue
        if sym in open_until and open_until[sym] > t:
            log_blocked(s, day, "blocked_pair")
            continue
        # correlated exposure: no second pair in the same direction at the same time
        scaled = False
        if cfg.block_same_direction and any(u > t and open_dir.get(o) == s["side"] for o, u in open_until.items() if o != sym):
            if cfg.corr_second_risk_mult <= 0:
                log_blocked(s, day, "blocked_corr")
                continue
            scaled = True                   # take the second correlated trade at reduced risk
        inst = instruments[sym]
        res = simulate_trade(s, m1_by_symbol[sym], swings[sym], cfg, inst, mode)
        rec = s.to_dict()
        rec.update(res)
        rec["day"] = day
        rec["mode"] = mode
        rec["corr_scaled"] = scaled
        if res["status"] == "filled":
            mult = float(rec.get("risk_mult", 1.0) or 1.0) * (cfg.corr_second_risk_mult if scaled else 1.0)
            rec["risk_mult"] = mult         # the daily-loss tracker reads this
            risk_money = balance * cfg.risk_pct * mult
            rec["R_w"] = rec["R_result"] * mult
            pips = rec["risk"] / inst.pip
            lots = np.floor(risk_money / (pips * inst.pip_value_per_lot) / inst.lot_step) * inst.lot_step
            rec["lots"] = lots
            rec["pnl"] = rec["R_result"] * risk_money
            balance += rec["pnl"]
            rec["balance"] = balance
            open_until[sym] = res["exit_time"]
            open_dir[sym] = s["side"]
        trades.append(rec)
    df = pd.DataFrame(trades)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


def metrics(trades: pd.DataFrame, oos_from: str | None = None) -> dict:
    """One-line metrics for experiment tables, with an in-sample / out-of-sample split."""
    isb = trades["status"].str.startswith("blocked") if len(trades) else pd.Series(dtype=bool)
    blocked, trades = trades[isb], trades[~isb]      # blocked rows are logged, not counted as signals
    out = {"signals": len(trades), "blocked": len(blocked),
           "blocked_shadow_R": blocked["shadow_R"].sum() if "shadow_R" in blocked else 0.0}
    if not len(trades):
        return out
    st = trades["status"]
    f = trades[st == "filled"]
    out["fill%"] = 100 * len(f) / len(trades)
    nf = trades[st.isin(["missed", "expired"])]
    if "shadow_R2" in nf and len(nf):
        out["unfilled_shadow2R"] = nf["shadow_R2"].mean()

    def block(x, pfx):
        r = x["R_result"] if len(x) else pd.Series(dtype=float)
        wins, losses = r[r > 0.05], r[r < -0.05]
        eq = r.cumsum()
        return {f"{pfx}trades": len(r),
                f"{pfx}win%": 100 * len(wins) / len(r) if len(r) else np.nan,
                f"{pfx}avgR": r.mean() if len(r) else np.nan,
                f"{pfx}totalR": r.sum(),
                f"{pfx}PF": wins.sum() / -losses.sum() if len(losses) else np.nan,
                f"{pfx}maxDD_R": (eq.cummax() - eq).max() if len(r) else np.nan}
    out.update(block(f, ""))
    if len(f) > 2:
        r = f["R_result"]
        out["t_stat"] = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std(ddof=1) > 0 else np.nan
        yrs = pd.to_datetime(f["signal_time"], utc=True).dt.tz_convert("America/New_York").dt.year
        for y, v in r.groupby(yrs.values).sum().items():
            out[f"y{y}"] = v
    if "mfe_R" in f and len(f):
        out["avg_mfe_R"] = f["mfe_R"].mean()
    if "R_w" in f and len(f) and (f.get("risk_mult", pd.Series([1.0])) != 1.0).any():
        w = f["R_w"]
        eqw = w.cumsum()
        out["w_totalR"], out["w_maxDD_R"] = w.sum(), (eqw.cummax() - eqw).max()
        out["w_t_stat"] = w.mean() / (w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 2 else np.nan
    if oos_from:
        cut = pd.Timestamp(oos_from, tz="America/New_York")
        t = pd.to_datetime(f["signal_time"], utc=True).dt.tz_convert("America/New_York")
        out.update(block(f[t < cut], "IS_"))
        out.update(block(f[t >= cut], "OOS_"))
    return out


def summarize(trades: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    f = trades[trades["status"] == "filled"] if len(trades) else trades
    if not len(f):
        return pd.DataFrame()

    def stats(x: pd.DataFrame) -> pd.Series:
        r = x["R_result"]
        eq = r.cumsum()
        dd = (eq.cummax() - eq).max()
        wins, losses = r[r > 0.05], r[r < -0.05]
        return pd.Series({
            "trades": len(r), "win%": 100 * len(wins) / len(r),
            "be%": 100 * (len(r) - len(wins) - len(losses)) / len(r),
            "avgR": r.mean(), "totalR": r.sum(),
            "PF": wins.sum() / -losses.sum() if len(losses) else np.inf,
            "maxDD_R": dd,
        })
    if by is None:
        return stats(f).to_frame("all").T
    return f.groupby(by).apply(stats, include_groups=False)
