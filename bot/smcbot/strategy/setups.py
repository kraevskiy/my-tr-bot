"""Entry setups.

Core: S1 "sweep -> Shift (M5 and M3 in sync) -> return to the cause of the break".
Long logic only; shorts run the same code on a price-mirrored copy of the market
(open/high/low/close -> -open/-low/-high/-close), so both sides share one code path.

Tags (Q7 "all setups"; derived from S1 context):
  S1  with H4 trend and D1 bias
  S2  counter-trend to H4, allowed only after an M15 BOS in the trade direction (PRO)
  S3  London killzone + Asia low/high swept (PO3 London manipulation)
  S4  NY killzone continuing the London move
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smcbot.config import Config, Instrument
from smcbot.data.bars import TF_DELTA, build_timeframes
from smcbot.market.liquidity import build_pools
from smcbot.market.sessions import killzone_labels
from smcbot.market.structure import fractal_swings, market_structure, swing_levels
from smcbot.market.zones import FAR_FUTURE, _utc_ns, find_fvgs, find_obs


CONT_MAX_BARS = 288   # stage-A horizon for S11; variants narrow it with cfg.cont_window_bars


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["open"], out["close"] = -df["open"], -df["close"]
    out["high"], out["low"] = -df["low"], -df["high"]
    return out


def _ns(idx) -> np.ndarray:
    return pd.DatetimeIndex(idx).tz_convert("UTC").tz_localize(None).values.astype("datetime64[ns]")


@dataclass
class Shift:
    pos: int        # bar where the close broke the swing high
    ext_pos: int    # bar of the leg extreme (sweep low)
    level: float    # swing high that was broken


def find_shifts(df: pd.DataFrame, n: int, lookback: int) -> list[Shift]:
    """Bullish Shift: close above the last confirmed swing high that precedes the
    lowest low of the recent leg (L14/L22/L32), first close only."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    sw = fractal_swings(df, n)["sh"].values
    sh_pos = np.flatnonzero(sw)
    out, used_ext = [], set()
    for i in range(lookback, len(df)):
        w0 = i - lookback
        s = w0 + int(np.argmin(l[w0:i]))
        if l[i] < l[s] or s in used_ext:
            continue
        # last swing high formed before s and already confirmed by bar i
        k = np.searchsorted(sh_pos, s) - 1
        while k >= 0 and sh_pos[k] + n > i:
            k -= 1
        if k < 0:
            continue
        p = sh_pos[k]
        lvl = h[p]
        if c[i] <= lvl or lvl <= l[s]:
            continue
        if s + 1 <= i - 1 and c[s:i].max() > lvl:
            continue
        used_ext.add(s)
        out.append(Shift(i, s, lvl))
    return out


class SideModel:
    """All detectors for one side (original = longs, mirrored = shorts)."""

    def __init__(self, m1: pd.DataFrame, cfg: Config, inst: Instrument, side: int):
        self.side, self.cfg, self.inst = side, cfg, inst
        m1s = m1 if side == 1 else mirror(m1)
        self.etf, self.stf = cfg.tf_entry          # entry TF and sync TF, default ("5min", "3min")
        tf_list = sorted(set((self.stf, self.etf, "15min", "1h", "4h", "1D", "1W")) - {"1min"})
        self.tfs = build_timeframes(m1s, tf_list)
        self.m1 = m1s
        n = cfg.fractal_n
        # structure states, indexed by close time
        self.state = {}
        for tf in ("15min", "1h", "4h", "1D", "1W"):
            st, _ = market_structure(self.tfs[tf], n[tf])
            st = st.copy()
            st["close_ns"] = _ns(st.index + TF_DELTA[tf])
            self.state[tf] = st
        # L39 candle model on D1: bias for the NEXT day (long frame; shorts via mirror)
        d1 = self.tfs["1D"]
        h_, l_, c_ = d1["high"].values, d1["low"].values, d1["close"].values
        cb = np.zeros(len(d1), int)
        for k in range(1, len(d1)):
            up_rev = l_[k] < l_[k - 1] and c_[k] >= l_[k - 1]      # swept PDL, closed back inside
            up_con = h_[k] > h_[k - 1] and c_[k] > h_[k - 1]       # took PDH and closed above
            dn_rev = h_[k] > h_[k - 1] and c_[k] <= h_[k - 1]
            dn_con = l_[k] < l_[k - 1] and c_[k] < l_[k - 1]
            bull, bear = up_rev or up_con, dn_rev or dn_con
            cb[k] = 1 if bull and not bear else (-1 if bear and not bull else 0)
        self.candle_bias = (_ns(d1.index + TF_DELTA["1D"]), cb)
        # H1 swings for Premium/Discount
        self.h1_sw = swing_levels(self.tfs["1h"], n["1h"], TF_DELTA["1h"])
        self.h1_sw["known_ns"] = _ns(self.h1_sw["known_at"])
        # POI zones (bullish in this side's frame) on M15/H1/H4 and FTA (bearish) on H1/H4
        zl = []
        for tf in cfg.tf_poi:
            f = find_fvgs(self.tfs[tf], TF_DELTA[tf], inst.pip * 0.5)
            o = find_obs(self.tfs[tf], TF_DELTA[tf], f, cfg.ob_max_engulf_bars)
            for z in (f, o):
                if len(z):
                    z = z.copy()
                    z["tf"] = tf
                    zl.append(z)
        zones = pd.concat(zl, ignore_index=True) if zl else pd.DataFrame()
        zones["known_ns"] = _utc_ns(zones["known_at"])
        zones["inv_ns"] = _utc_ns(zones["invalid_at"], fill_max=True)
        self.zones = zones
        # pools in this side's frame
        pools = build_pools(m1s, self.tfs, n, cfg.asia_ny)
        self.pools = pools
        # entry TF data
        # names m5/m3 kept for history: they are the entry TF and the sync TF
        self.m5, self.m3 = self.tfs[self.etf], self.tfs[self.stf]
        e_min = TF_DELTA[self.etf] / pd.Timedelta(minutes=1)
        s_min = TF_DELTA[self.stf] / pd.Timedelta(minutes=1)
        self.k_e = 5.0 / e_min                     # cfg *_bars are M5 bars -> entry-TF bars
        nn = lambda tf: n.get(tf, 2)
        self.m5_ns = _ns(self.m5.index)
        self.m3_ns = _ns(self.m3.index)
        self.m5_fvg = find_fvgs(self.m5, TF_DELTA[self.etf], cfg.min_fvg_pips * inst.pip)
        lb_min = cfg.sweep_lookback_bars * 5
        self.sh5 = find_shifts(self.m5, nn(self.etf), max(4, int(round(lb_min / e_min))))
        self.sh3 = find_shifts(self.m3, nn(self.stf), max(4, int(round(lb_min / s_min))))
        # NY midnight opens (PO3 / L42)
        idx_ny = m1s.index.tz_convert("America/New_York")
        mk = (idx_ny.hour == 0) & (idx_ny.minute == 0)
        self.nym_ns = _ns(m1s.index[mk])
        self.nym_px = m1s["open"].values[mk]
        # ADR(10) and the part of today's range already used, per entry-TF bar (trading day 17:00 NY)
        e = self.m5
        tday = (e.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)).normalize()
        g = pd.DataFrame({"d": tday, "h": e["high"].values, "l": e["low"].values})
        run_rng = (g.groupby("d")["h"].cummax() - g.groupby("d")["l"].cummin()).values
        drng = g.groupby("d").agg(h=("h", "max"), l=("l", "min"))
        adr = (drng["h"] - drng["l"]).rolling(10, min_periods=5).mean().shift(1)
        self.adr_used = run_rng / adr.reindex(tday).values          # NaN in the first days

    # ---------- context lookups (no look-ahead: only bars closed by t) ----------
    def _prep_lookups(self):
        if getattr(self, "_lk", None) is not None:
            return
        lk = {}
        for tf, st in self.state.items():
            lk[tf] = (st["close_ns"].values, st["trend"].values)
        lk["candle"] = self.candle_bias
        sw = self.h1_sw
        for kind in ("high", "low"):
            k = sw[sw["kind"] == kind].sort_values("known_ns")
            lk["h1_" + kind] = (k["known_ns"].values, k["price"].values)
        self._lk = lk
        self._memo = {}

    def trend_at(self, tf: str, t_ns) -> int:
        ns, tr = self._lk[tf]
        k = np.searchsorted(ns, t_ns, side="right") - 1
        return 0 if k < 0 else int(tr[k])

    def pd_range_at(self, t_ns):
        self._prep_lookups()
        out = []
        for kind in ("high", "low"):
            ns, px = self._lk["h1_" + kind]
            k = np.searchsorted(ns, t_ns, side="right") - 1
            if k < 0:
                return None
            out.append(px[k])
        H, L = out
        return (min(H, L), max(H, L))

    def poi_touched(self, low_s: float, t_s_ns, t_sig_ns, min_close: float, tfs=None):
        z = self.zones
        if not len(z):
            return None
        m = (z["dir"].values == 1) & (z["known_ns"].values <= t_s_ns) & (z["inv_ns"].values > t_sig_ns)
        m &= (low_s <= z["hi"].values) & (min_close >= z["lo"].values)
        if tfs is not None:
            m &= np.isin(z["tf"].values, list(tfs))
        if not m.any():
            return None
        order = {"4h": 3, "1h": 2, "15min": 1}
        tf_arr, ty_arr = z["tf"].values[m], z["type"].values[m]
        j = int(np.argmax([order.get(t, 0) for t in tf_arr]))
        return f"{ty_arr[j]}_{tf_arr[j]}"

    def obstacle_above(self, entry: float, t_ns, lon: bool = False):
        """Nearest target/obstacle: untaken BSL (MEDIUM+) or active opposing H1/H4 zone (FTA)."""
        p = self.pools
        m = (p["dir"].values == 1) & (p["known_ns"].values <= t_ns) & (p["rank"].values >= 2)
        tk = p["taken_ns"].values
        m &= (np.isnat(tk) | (tk > t_ns)) & (p["level"].values > entry)
        if not lon:
            m &= ~pd.Series(p["name"].values).str.startswith("LON").values
        cand = [(lvl, f"POOL_{nm}") for lvl, nm in zip(p["level"].values[m], p["name"].values[m])]
        z = self.zones
        if len(z):
            mz = (z["dir"].values == -1) & np.isin(z["tf"].values, ["1h", "4h"])
            mz &= (z["known_ns"].values <= t_ns) & (z["inv_ns"].values > t_ns) & (z["lo"].values > entry)
            cand += [(lvl, f"FTA_{ty}_{tf}") for lvl, ty, tf in
                     zip(z["lo"].values[mz], z["type"].values[mz], z["tf"].values[mz])]
        if not cand:
            return None, None
        return min(cand, key=lambda x: x[0])

    def crt_at(self, tf: str, t_ns) -> bool:
        """Bullish CRT (this side's frame): the last closed candle swept the previous candle's low
        and closed back inside its range."""
        key = "_crt_" + tf
        if not hasattr(self, key):
            d = self.tfs[tf]
            lo, hi, cl = d["low"].values, d["high"].values, d["close"].values
            ok = np.zeros(len(d), bool)
            ok[1:] = (lo[1:] < lo[:-1]) & (cl[1:] > lo[:-1]) & (cl[1:] < hi[:-1])
            setattr(self, key, (_ns(d.index + TF_DELTA[tf]), ok))
        ns, ok = getattr(self, key)
        k = np.searchsorted(ns, t_ns, side="right") - 1
        return bool(k >= 0 and ok[k])

    def pool_above(self, entry: float, t_ns, min_rank: int = 3, names=None):
        """Nearest untaken pool above entry (BSL in this side's frame) with rank >= min_rank."""
        p = self.pools
        m = (p["dir"].values == 1) & (p["known_ns"].values <= t_ns) & (p["rank"].values >= min_rank)
        tk = p["taken_ns"].values
        m &= (np.isnat(tk) | (tk > t_ns)) & (p["level"].values > entry)
        if names:
            m &= pd.Series(p["name"].values).str.startswith(tuple(names)).values
        if not m.any():
            return None, None
        j = np.flatnonzero(m)[np.argmin(p["level"].values[m])]
        return float(p["level"].values[j]), str(p["name"].values[j])

    # ---------- stage A: variant-independent candidate features ----------
    def base_candidates(self) -> list[dict]:
        """Every M5 Shift with the features that do not depend on strategy knobs.

        `swept` holds every pool taken by the leg and reclaimed within sweep_return_bars;
        the pool actually used is picked per variant in stage B (session_pools/sweep_pools).
        `cont_prev` lists earlier swept legs this Shift could continue (S11, higher low)."""
        if getattr(self, "_base", None) is not None:
            return self._base
        cfg, inst = self.cfg, self.inst
        m5 = self.m5
        o5, l5, c5 = m5["open"].values, m5["low"].values, m5["close"].values
        sh3_close = np.array([self.m3_ns[s.pos] for s in self.sh3], dtype="datetime64[ns]") + TF_DELTA[self.stf].to_timedelta64()
        sh3_lows = np.array([self.m3["low"].values[s.ext_pos] for s in self.sh3])
        bar5 = TF_DELTA[self.etf].to_timedelta64()
        fv = self.m5_fvg
        fpos = np.searchsorted(self.m5_ns, _utc_ns(fv["time"])) if len(fv) else np.array([], int)
        fknown = _utc_ns(fv["known_at"]) if len(fv) else np.array([], "datetime64[ns]")
        p = self.pools
        out = []
        for sh in self.sh5:
            i, s = sh.pos, sh.ext_pos
            t_s, t_shift, low_s = self.m5_ns[s], self.m5_ns[i] + bar5, l5[s]
            row = {"i": i, "s": s, "t_s": t_s, "t_shift": t_shift, "low_s": low_s,
                   "shift_level": sh.level, "close_i": c5[i], "base_reject": "", "swept": [],
                   "cont_prev": []}
            win_start = self.m5_ns[max(0, s - int(round(cfg.sweep_lookback_bars * self.k_e)))]
            m = (p["dir"].values == -1) & (p["known_ns"].values <= t_s)
            tk = p["taken_ns"].values
            m &= ~np.isnat(tk) & (tk >= win_start) & (tk < self.m5_ns[s] + bar5)
            m &= (p["level"].values > low_s) & (p["level"].values < c5[i])
            sw = p[m]
            if sw.empty:
                row["base_reject"] = "no_sweep"
            else:
                bpos = np.searchsorted(self.m5_ns, sw["taken_ns"].values, side="right") - 1
                keep = [j for j, (bp, lvl) in enumerate(zip(bpos, sw["level"].values))
                        if (c5[bp:min(len(c5), bp + max(1, int(round(cfg.sweep_return_bars * self.k_e))) + 1)] > lvl).any()]
                if not keep:
                    row["base_reject"] = "no_reclaim"
                else:
                    k_ = sw.iloc[keep]
                    row["swept"] = list(zip(k_["name"].values, k_["tier"].values,
                                            k_["level"].values, k_["rank"].values))
            # S11 continuation: earlier swept legs, no lower low since, this low is higher
            lo_win = s - int(CONT_MAX_BARS * self.k_e)
            for j in range(len(out) - 1, -1, -1):
                pj = out[j]
                if pj["s"] < lo_win:
                    break
                if pj["swept"] and pj["s"] < s and low_s > pj["low_s"] and l5[pj["s"]:s].min() >= pj["low_s"]:
                    row["cont_prev"].append(j)
            mm = (sh3_close >= t_s) & (sh3_close <= t_shift + cfg.sync_window_bars * np.timedelta64(5, "m")) & \
                 (np.abs(sh3_lows - low_s) <= inst.pip * 1.0)
            row["t_m3"] = sh3_close[mm].min() if mm.any() else np.datetime64("NaT", "ns")
            row["fvg_hi"] = row["fvg_mid"] = np.nan
            row["fvg_known"] = np.datetime64("NaT", "ns")
            if len(fv):
                cand = np.flatnonzero((fpos > s) & (fpos <= i) & (fv["dir"].values == 1))
                row["fvg2_hi"], row["fvg2_known"] = np.nan, np.datetime64("NaT", "ns")
                if cand.size:
                    k = cand[0]
                    row.update(fvg_hi=fv["hi"].values[k], fvg_mid=fv["mid"].values[k], fvg_known=fknown[k])
                    k2 = cand[-1]                                   # the last FVG of the leg (closest to price)
                    row.update(fvg2_hi=fv["hi"].values[k2], fvg2_known=fknown[k2])
            row["leg_hi"] = float(m5["high"].values[s:i + 1].max())
            row["fvg_lo"] = np.nan
            row["ifvg_hi"] = row["ifvg_lo"] = np.nan
            if len(fv):
                if not np.isnan(row["fvg_hi"]):
                    row["fvg_lo"] = fv["lo"].values[cand[0]]
                # IFVG: bearish M5 FVG of the down leg (before the low) that price closed back above by the Shift
                lb_e = int(round(cfg.sweep_lookback_bars * self.k_e))
                bc = np.flatnonzero((fpos >= s - lb_e) & (fpos < s) & (fv["dir"].values == -1) & (fv["lo"].values > low_s))
                best = None
                for k in bc:
                    hi_k = fv["hi"].values[k]
                    if (c5[fpos[k] + 2:i + 1] > hi_k).any() and (best is None or hi_k < best[0]):
                        best = (hi_k, fv["lo"].values[k])
                if best:
                    row["ifvg_hi"], row["ifvg_lo"] = best
            ks = [k for k in range(max(0, s - 3), min(i, s + 2)) if c5[k] < o5[k]]
            row["ob_open"] = o5[ks[-1]] if ks else np.nan
            row["min_close"] = c5[s:i + 1].min()
            out.append(row)
        self._base = out
        return out

    @staticmethod
    def _pick_pool(swept, cfg: Config):
        """Pool used by a variant: highest tier, then the lowest level (deepest sweep first)."""
        c = [x for x in swept if (cfg.session_pools or not x[0].startswith("LON"))]
        if cfg.sweep_pools:
            c = [x for x in c if x[0].startswith(tuple(cfg.sweep_pools))]
        if not c:
            return None
        return sorted(c, key=lambda x: (-x[3], x[2]))[0]

    def nym_at(self, t_ns):
        """NY-midnight open of the current NY day (this side's frame), None if unknown."""
        k = np.searchsorted(self.nym_ns, t_ns, side="right") - 1
        if k < 0 or (t_ns - self.nym_ns[k]) > np.timedelta64(24, "h"):
            return None
        return float(self.nym_px[k])

    # ---------- stage B: apply a strategy variant ----------
    def signals(self, cfg: Config | None = None, log: list | None = None) -> list[dict]:
        cfg = cfg or self.cfg
        inst = self.inst
        self._prep_lookups()
        m5 = self.m5
        c5 = m5["close"].values
        spread = m5["spread"].values if "spread" in m5 else np.full(len(m5), inst.default_spread_pips)
        bar5 = TF_DELTA[self.etf].to_timedelta64()
        kz_def = cfg.killzones()
        from collections import Counter
        self.funnel = fn = Counter()
        out = []
        for b in self.base_candidates():
            fn["shift_m5"] += 1
            rec = {"side": self.side, "t_shift": b["t_shift"], "ext_ns": b["t_s"], "reasons": ""}
            base = self._base
            pk = self._pick_pool(b["swept"], cfg)
            cont = False
            if pk is None and cfg.allow_continuation:
                for j in b["cont_prev"]:
                    if b["s"] - base[j]["s"] > cfg.cont_window_bars * self.k_e:
                        break
                    pj = self._pick_pool(base[j]["swept"], cfg)
                    if pj is not None:
                        pk, cont = ("CONT_" + pj[0], pj[1], pj[2], pj[3]), True
                        break
            if pk is None:
                why = b["base_reject"] or "pool_filtered"
                fn["x_" + why] += 1
                if log is not None:
                    rec["reasons"] = why
                    log.append(rec)
                continue
            b = dict(b, pool=pk[0], tier=pk[1], pool_level=pk[2])
            fails = []
            has_m3 = not np.isnat(b["t_m3"])
            if cfg.require_m3_sync and not has_m3:
                fails.append("no_m3_sync")
            t_sig = max(b["t_shift"], b["t_m3"]) if (cfg.require_m3_sync and has_m3) else b["t_shift"]
            # entry selection
            fvg_ok, ob_ok = not np.isnan(b["fvg_hi"]), not np.isnan(b["ob_open"])
            mode = cfg.entry_mode
            entry, zone, order = None, None, "limit"
            if mode == "market":
                zone, order = "MARKET", "market"
            elif mode in ("fvg_edge", "fvg_ce"):
                if fvg_ok:
                    entry, zone = (b["fvg_hi"] if mode == "fvg_edge" else b["fvg_mid"]), "FVG_M5"
                elif ob_ok:
                    entry, zone = b["ob_open"], "OB_M5"
            elif mode == "ob_open":
                if ob_ok:
                    entry, zone = b["ob_open"], "OB_M5"
                elif fvg_ok:
                    entry, zone = b["fvg_hi"], "FVG_M5"
            elif mode == "fvg_last":
                if fvg_ok:
                    entry, zone = b["fvg2_hi"], "FVG2_M5"
                elif ob_ok:
                    entry, zone = b["ob_open"], "OB_M5"
            elif mode == "ote":                        # Fibonacci 62-79% of the Shift leg (low -> leg high)
                entry, zone = b["leg_hi"] - cfg.ote_level * (b["leg_hi"] - b["low_s"]), "OTE"
            elif mode == "ifvg":                       # retest of the inverted bearish FVG (now support)
                if not np.isnan(b["ifvg_hi"]):
                    entry, zone = b["ifvg_hi"], "IFVG_M5"
            elif mode == "retest":                     # breaker-style: retest of the broken swing high
                entry, zone = b["shift_level"], "RETEST_M5"
            elif mode == "retest_fvg":                 # whichever comes first: FVG edge or retest
                opts = [(b["shift_level"], "RETEST_M5")] + ([(b["fvg_hi"], "FVG_M5")] if fvg_ok else [])
                entry, zone = max(opts, key=lambda x: x[0])
            elif mode == "nearest":
                opts = [(b["fvg_hi"], "FVG_M5")] if fvg_ok else []
                opts += [(b["ob_open"], "OB_M5")] if ob_ok else []
                if opts:
                    entry, zone = max(opts, key=lambda x: x[0])
            else:
                raise ValueError(f"unknown entry_mode {mode}")
            if zone == "FVG_M5":
                t_sig = max(t_sig, b["fvg_known"])
            elif zone == "FVG2_M5":
                t_sig = max(t_sig, b["fvg2_known"])
            leg = b["leg_hi"] - b["low_s"]
            fvg_ret = (b["leg_hi"] - b["fvg_hi"]) / leg if (fvg_ok and leg > 0) else np.nan
            if (cfg.fvg_min_retrace and not (fvg_ret >= cfg.fvg_min_retrace)) or \
               (cfg.fvg_max_retrace and not (fvg_ret <= cfg.fvg_max_retrace)):
                fails.append("fvg_retrace")
            unicorn = bool(fvg_ok and not np.isnan(b["fvg_lo"]) and b["fvg_lo"] <= b["shift_level"] <= b["fvg_hi"])
            if cfg.require_unicorn and not unicorn:
                fails.append("no_unicorn")
            crt4, crt1 = self.crt_at("4h", t_sig), self.crt_at("1h", t_sig)
            if cfg.require_crt != "off" and not {"h4": crt4, "h1": crt1, "any": crt4 or crt1}[cfg.require_crt]:
                fails.append("no_crt")
            sig_pos = int(np.searchsorted(self.m5_ns, t_sig - bar5, side="right") - 1)
            sig_pos = max(0, min(sig_pos, len(m5) - 1))
            if order == "market":
                entry = c5[sig_pos]
            if zone is None:
                fails.append("no_entry_zone")
                entry = c5[sig_pos]
            kz = killzone_labels(pd.DatetimeIndex([pd.Timestamp(t_sig - np.timedelta64(1, "s"), tz="UTC")]).tz_convert("America/New_York"), kz_def)[0]
            if not kz:
                fails.append("outside_kz")
            elif cfg.require_asia_london and kz == "LONDON" and b["pool"] != "ASIA_L":
                fails.append("not_asia_sweep")
            spr = float(spread[sig_pos]) * inst.pip
            sl = b["low_s"] - cfg.sl_buffer_pips * inst.pip - spr
            if order == "limit" and entry >= c5[sig_pos]:
                fails.append("entry_vs_price")
            if entry <= sl:
                fails.append("entry_below_sl")
            risk = max(entry - sl, 1e-9)
            risk_pips = risk / inst.pip
            if cfg.min_sl_pips and risk_pips < cfg.min_sl_pips:
                fails.append("sl_too_small")
            if cfg.max_sl_pips and risk_pips > cfg.max_sl_pips:
                fails.append("sl_too_big")
            if cfg.max_spread_frac and spr / risk > cfg.max_spread_frac:
                fails.append("spread_cost")
            h4 = self.trend_at("4h", t_sig)
            d1 = self.trend_at("1D", t_sig)
            m15 = self.trend_at("15min", t_sig)
            w1 = self.trend_at("1W", t_sig)
            cbias = self.trend_at("candle", t_sig)
            if cfg.bias_mode == "h4":
                setup = "S1" if h4 == 1 else ("S2" if m15 == 1 else "")
                if not setup:
                    fails.append("trend")
                if setup == "S1" and cfg.require_d1_bias and d1 != 1:
                    fails.append("d1_bias")
            else:
                setup = "S1" if h4 == 1 else "S2"
                if cfg.bias_mode == "daily_candle" and cbias != 1:
                    fails.append("candle_bias")
            if cfg.avoid_candle_against and cbias == -1:
                fails.append("candle_against")
            if cfg.require_m15_align and m15 != 1:
                fails.append("m15_against")
            if setup == "S1" and not cfg.allow_s1:
                fails.append("s1_disabled")
            elif setup == "S2" and not cfg.allow_s2:
                fails.append("s2_disabled")
            rng = self.pd_range_at(t_sig)
            pd_pos = np.nan if rng is None or rng[1] == rng[0] else (entry - rng[0]) / (rng[1] - rng[0])
            if cfg.require_pd and (rng is None or entry >= (rng[0] + rng[1]) / 2):
                fails.append("not_discount")
            key = ("poi", b["s"], int(t_sig.astype("int64")))
            if key not in self._memo:
                self._memo[key] = self.poi_touched(b["low_s"], b["t_s"], t_sig, b["min_close"])
            poi_any = self._memo[key]
            poi = poi_any if (poi_any and poi_any.split("_")[-1] in cfg.poi_tfs) else \
                self.poi_touched(b["low_s"], b["t_s"], t_sig, b["min_close"], cfg.poi_tfs) if poi_any else None
            if cfg.require_poi_or_major and poi is None and b["tier"] != "MAJOR":
                fails.append("no_poi")
            delay = (t_sig - b["t_s"]) / np.timedelta64(1, "m")
            if cfg.max_delay_min and delay > cfg.max_delay_min:
                fails.append("stale")
            adr_used = float(self.adr_used[sig_pos])          # bar sig_pos closes at the signal time
            if cfg.max_adr_used and not (adr_used <= cfg.max_adr_used):
                fails.append("adr_used")
            nym = self.nym_at(b["t_s"])
            po3_dist = np.nan if nym is None else (nym - b["low_s"]) / inst.pip
            if cfg.po3_max_pips:
                if cfg.po3_soft:      # only "the sweep happened within X pips of the NY-midnight open"
                    if nym is None or po3_dist > cfg.po3_max_pips:
                        fails.append("po3")
                elif nym is None or not (0 < po3_dist <= cfg.po3_max_pips) or entry >= nym:
                    fails.append("po3")
            # setup grade (A = 3-4 of: sweep near the NY-midnight open, fast reaction, M15 aligned,
            # daily candle model not against); used for risk weighting (grade_mode)
            q = [bool(po3_dist <= cfg.grade_po3_pips), bool(delay <= cfg.grade_delay_min),
                 m15 == 1, cbias != -1]
            score = int(sum(q))
            grade = "A" if score >= 3 else "B"
            risk_mult = 1.0
            if cfg.grade_mode == "weight":
                risk_mult = cfg.grade_risk_a if grade == "A" else cfg.grade_risk_b
            elif cfg.grade_mode == "a_only" and grade != "A":
                fails.append("grade_b")
            risk_mult *= cfg.kz_risk_mult.get(kz, 1.0)
            okey = ("obst", round(entry, 6), int(t_sig.astype("int64")), cfg.session_pools)
            if okey not in self._memo:
                self._memo[okey] = self.obstacle_above(entry, t_sig, lon=cfg.session_pools)
            obst, obst_name = self._memo[okey]
            rr_obst = (obst - entry) / risk if obst is not None else np.inf
            if rr_obst < cfg.min_rr:
                fails.append("target_too_close")
            # nearest untaken MAJOR pool (PDH/PWH/PMH in this frame) - target / partial candidates
            mkey = ("major", round(entry, 6), int(t_sig.astype("int64")), tuple(cfg.target_pools))
            if mkey not in self._memo:
                self._memo[mkey] = self.pool_above(entry, t_sig, 3, cfg.target_pools or None)
            maj, maj_name = self._memo[mkey]
            rr_major = (maj - entry) / risk if maj is not None else np.inf
            partial_r = cfg.partial_at_r
            if cfg.partial_mode == "pool" and maj is not None and cfg.pool_rr_min <= rr_major < partial_r:
                partial_r = rr_major * cfg.pool_front_frac          # bank half just before the level reacts
            # target by tp_mode
            if cfg.tp_mode in ("majorcap", "majorext"):
                k_r = cfg.tp_base_r
                tp, tp_kind = entry + k_r * risk, f"FIXED_{k_r:g}R"
                if maj is not None:
                    if cfg.tp_mode == "majorcap" and cfg.pool_rr_min <= rr_major < k_r:
                        tp, tp_kind = entry + rr_major * cfg.pool_front_frac * risk, "POOL_" + maj_name
                    elif cfg.tp_mode == "majorext" and k_r < rr_major <= cfg.pool_rr_max:
                        tp, tp_kind = entry + rr_major * cfg.pool_front_frac * risk, "POOL_" + maj_name
            elif cfg.tp_mode.startswith("fixed"):
                k_r = float(cfg.tp_mode[5:])
                tp, tp_kind = entry + k_r * risk, f"FIXED_{k_r:g}R"
            elif cfg.tp_mode == "pool":
                tp, tp_kind = (obst, obst_name) if obst is not None else (entry + cfg.fixed_tp_r_medium_minor * risk, "FIXED_4R")
            elif b["tier"] == "MAJOR" and obst is not None:
                tp, tp_kind = obst, obst_name
            else:
                tp = entry + cfg.fixed_tp_r_medium_minor * risk if obst is None else \
                    min(obst, entry + cfg.fixed_tp_r_medium_minor * risk)
                tp_kind = "FIXED_4R" if obst is None or tp < obst else obst_name
            tags = [setup or "-"]
            if cont:
                tags.append("S11")
            if str(b["pool"]).startswith(("ASIA", "LON")):
                tags.append("SR")
            if kz == "LONDON" and b["pool"] == "ASIA_L":
                tags.append("S3")
            if kz == "NY":
                ld = self._london_move(t_sig)
                if ld is not None and ld > 0:
                    tags.append("S4")
            row = {
                "side": self.side, "setup": "+".join(tags), "signal_ns": t_sig,
                "entry": entry, "sl": sl, "tp": tp, "risk": risk, "rr": (tp - entry) / risk,
                "order": order, "pool": b["pool"], "tier": b["tier"], "pool_level": b["pool_level"],
                "poi": poi or "", "zone": zone or "", "tp_kind": tp_kind, "killzone": kz,
                "h4": h4, "d1": d1, "w1": w1, "m15": m15, "ext_ns": b["t_s"],
                "risk_pips": risk_pips, "rr_obstacle": rr_obst, "pd_pos": pd_pos,
                "candle_bias": cbias, "spread_frac": spr / risk,
                "cont": cont, "po3_dist": po3_dist, "adr_used": adr_used,
                "rr_major": rr_major, "major_pool": maj_name or "", "partial_r": partial_r,
                "inval": (b["fvg_lo"] if zone == "FVG_M5" else np.nan),
                "score": score, "grade": grade, "risk_mult": risk_mult,
                "fvg_ret": fvg_ret, "unicorn": unicorn, "crt_h4": crt4, "crt_h1": crt1,
                "has_ifvg": not np.isnan(b["ifvg_hi"]),
                "m3_sync": has_m3, "delay_min": (t_sig - b["t_s"]) / np.timedelta64(1, "m"),
            }
            for f in fails:
                fn["x_" + f] += 1
            if log is not None:
                rec.update(row)
                rec["reasons"] = ",".join(fails)
                log.append(rec)
            if fails:
                continue
            fn["signal"] += 1
            out.append(row)
        return out

    def _london_move(self, t_ns):
        """Net London move (02:00->05:00 NY) of the current day, in this side's frame."""
        t = pd.Timestamp(t_ns, tz="UTC").tz_convert("America/New_York")
        day = t.normalize()
        a, b = day + pd.Timedelta(hours=2), day + pd.Timedelta(hours=5)
        seg = self.m1.loc[a:b - pd.Timedelta(minutes=1)]
        if seg.empty:
            return None
        return seg["close"].iat[-1] - seg["open"].iat[0]


def unmirror(sig: dict) -> dict:
    out = dict(sig)
    if sig.get("side") != -1:
        return out
    for k in ("entry", "sl", "tp", "pool_level", "inval"):
        if k in out and out[k] is not None and not (isinstance(out[k], float) and np.isnan(out[k])):
            out[k] = -out[k]
    for k in ("h4", "d1", "w1", "m15", "candle_bias"):
        if k in out and out[k] is not None and not (isinstance(out[k], float) and np.isnan(out[k])):
            out[k] = -out[k]
    if isinstance(out.get("pool"), str):
        out["pool"] = out["pool"].replace("_L", "_H").replace("SL", "SH").replace("PDL", "PDH") \
            .replace("PWL", "PWH").replace("PML", "PMH")
    return out


def _to_frame(rows: list[dict], symbol: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([unmirror(r) for r in rows])
    df["signal_time"] = pd.to_datetime(df["signal_ns"] if "signal_ns" in df else df["t_shift"]) \
        .dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df["symbol"] = symbol
    return df.sort_values("signal_time").reset_index(drop=True)


class SymbolModels:
    """Both sides for one symbol; detectors are computed once, variants are cheap."""

    def __init__(self, m1: pd.DataFrame, cfg: Config, inst: Instrument):
        self.inst = inst
        self.m1 = m1
        self.models = [SideModel(m1, cfg, inst, side) for side in (1, -1)]
        for m in self.models:
            m.base_candidates()

    def signals(self, cfg: Config, funnel: dict | None = None, log: list | None = None) -> pd.DataFrame:
        rows, logs = [], []
        for m in self.models:
            rows += m.signals(cfg, logs if log is not None else None)
            if funnel is not None:
                for k, v in m.funnel.items():
                    funnel[k] = funnel.get(k, 0) + v
        if log is not None and logs:
            log.append(_to_frame(logs, self.inst.symbol))
        out = _to_frame(rows, self.inst.symbol)
        if cfg.asia_fake_only:
            out = pd.DataFrame()
        if cfg.asia_fake and (not cfg.af_symbols or self.inst.symbol in cfg.af_symbols):
            from smcbot.strategy.asia_fake import asia_fake_signals
            af = asia_fake_signals(self.m1, cfg, self.inst)
            if funnel is not None:
                funnel["signal_AF"] = funnel.get("signal_AF", 0) + len(af)
            if len(af):
                out = pd.concat([out, af], ignore_index=True) if len(out) else af
                out = out.sort_values("signal_time").reset_index(drop=True)
        return out


def generate_signals(m1: pd.DataFrame, cfg: Config, inst: Instrument,
                     funnel: dict | None = None) -> pd.DataFrame:
    return SymbolModels(m1, cfg, inst).signals(cfg, funnel)
