"""Prop-firm 2-phase challenge simulator on a closed-trade list (rolling starts + block bootstrap).

  python -m smcbot prop results/<run>/<variant>/trades_c.csv --risk 0.5,0.75,1,1.5 --paths 10000 --seed 7
  options: --target1 10 --target2 5 --daily 5 --max 10 --min-days 4 --trailing --mae --months 24

Account = 100 (percent units). Phase 1 -> fresh account -> phase 2. Day = trading day (17:00 NY start).
Only closed-trade P&L is modelled: intraday floating-loss breaches are not seen (--mae adds a conservative check).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smcbot.config import NY_TZ

TZ = NY_TZ
EPOCH = np.datetime64("1970-01-01", "D")


@dataclass(frozen=True)
class Rules:
    t1: float = 10.0
    t2: float = 5.0
    daily: float = 5.0
    maxloss: float = 10.0
    min_days: int = 4
    trailing: bool = False
    mae: bool = False


def load(path: str) -> pd.DataFrame:
    t = pd.read_csv(path)
    f = t[t["status"] == "filled"].copy()
    for c in ("exit_time", "fill_time"):
        f[c] = pd.to_datetime(f[c], utc=True).dt.tz_convert(TZ)
    rm = f["risk_mult"] if "risk_mult" in f else 1.0
    f["R_w"] = f["R_w"].fillna(f["R_result"] * rm) if "R_w" in f else f["R_result"] * rm
    f["mae_w"] = f["mae_R"] * rm
    f["day"] = (f["exit_time"].dt.tz_localize(None) + pd.Timedelta(hours=7)).dt.normalize()
    return f.sort_values("exit_time").reset_index(drop=True)


def arrays(f: pd.DataFrame):
    day = (f["day"].values.astype("datetime64[D]") - EPOCH).astype(int)
    return day, f["R_w"].to_numpy(float), f["mae_w"].fillna(0).to_numpy(float)


def replay(day_i, rw, mae_w, risk, rules: Rules) -> dict:
    bal = start = dopen = peak = 100.0
    phase, days, first, first_all, cur, d1, dd, res = 1, set(), None, None, None, np.nan, 0.0, "censored"
    d_total = np.nan
    for k in range(len(rw)):
        d = int(day_i[k])
        if first is None:
            first = d
            first_all = d if first_all is None else first_all
        if d != cur:
            cur, dopen = d, bal
            peak = max(peak, bal)
        if rules.mae and (bal - mae_w[k] * risk) - dopen <= -rules.daily / 100 * dopen:
            res = "fail_daily"        # conservative: worst excursion assumed on top of the closed balance
            break
        bal += rw[k] * risk
        dd = max(dd, start - bal)
        days.add(d)
        if bal - dopen <= -rules.daily / 100 * dopen:
            res = "fail_daily"
            break
        floor = peak - rules.maxloss / 100 * start if rules.trailing else start * (1 - rules.maxloss / 100)
        if bal <= floor:
            res = "fail_max"
            break
        if bal >= start * (1 + (rules.t1 if phase == 1 else rules.t2) / 100) and len(days) >= rules.min_days:
            if phase == 2:
                res, d_total = "pass", d - first_all
                break
            d1 = d - first
            phase, bal, start, dopen, peak, days, first, cur = 2, 100.0, 100.0, 100.0, 100.0, set(), None, None
    return {"result": res, "phase": phase, "d1": d1, "d_total": d_total, "dd": dd / start * 100,
            "n_trades": k + 1 if len(rw) else 0}


def _agg(rs: list[dict], horizons=()) -> dict:
    n = len(rs)
    res = np.array([r["result"] for r in rs])
    p = {k: (res == k).sum() for k in ("pass", "fail_daily", "fail_max", "censored")}
    dt = np.array([r["d_total"] for r in rs if r["result"] == "pass"], float)
    d1 = np.array([r["d1"] for r in rs if r["result"] == "pass"], float)
    out = {"P(pass)": p["pass"] / n, "P(fail_daily)": p["fail_daily"] / n, "P(fail_max)": p["fail_max"] / n,
           "P(censored)": p["censored"] / n,
           "P(pass|decided)": p["pass"] / max(1, n - p["censored"]),
           "med d1": np.median(d1) if len(d1) else np.nan,
           "med d_total": np.median(dt) if len(dt) else np.nan,
           "p90 d_total": np.percentile(dt, 90) if len(dt) else np.nan,
           "max dd %": max(r["dd"] for r in rs)}
    for h in horizons:
        out[f"P(pass<={h}d)"] = (dt <= h).sum() / n
    return out


def rolling(arrs, risks, rules: Rules) -> pd.DataFrame:
    day, rw, mae = arrs
    starts = np.searchsorted(day, np.unique(day), side="left")
    rows = {r: _agg([replay(day[s:], rw[s:], mae[s:], r, rules) for s in starts]) for r in risks}
    return pd.DataFrame(rows).T.rename_axis("risk %")


def bootstrap(arrs, risks, rules: Rules, paths: int, seed: int, months: int) -> pd.DataFrame:
    day, rw, mae = arrs
    dts = pd.to_datetime(day.astype("datetime64[D]"))
    per = pd.PeriodIndex(dts, freq="M")
    blocks = []
    for p in pd.period_range(per.min(), per.max(), freq="M"):      # empty months stay in the sample
        m = per == p
        s = (np.datetime64(p.start_time.date(), "D") - EPOCH).astype(int)
        blocks.append((p.days_in_month, day[m] - s, rw[m], mae[m]))
    idx = np.random.default_rng(seed).integers(0, len(blocks), size=(paths, months))   # common random numbers
    out = {r: [] for r in risks}
    for row in idx:
        off, parts = 0, []
        for b in row:
            n, dd_, r_, m_ = blocks[b]
            parts.append((dd_ + off, r_, m_))
            off += n
        d = np.concatenate([p[0] for p in parts])
        w = np.concatenate([p[1] for p in parts])
        m = np.concatenate([p[2] for p in parts])
        for r in risks:
            out[r].append(replay(d, w, m, r, rules))
    return pd.DataFrame({r: _agg(v, (91, 182, 365)) for r, v in out.items()}).T.rename_axis("risk %")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="smcbot prop", description="prop-firm challenge simulator")
    ap.add_argument("trades")
    ap.add_argument("--risk", default="0.5,0.75,1,1.5", help="comma-separated risk %% per 1R")
    ap.add_argument("--target1", type=float, default=10.0)
    ap.add_argument("--target2", type=float, default=5.0)
    ap.add_argument("--daily", type=float, default=5.0)
    ap.add_argument("--max", type=float, default=10.0, dest="maxloss")
    ap.add_argument("--min-days", type=int, default=4)
    ap.add_argument("--trailing", action="store_true")
    ap.add_argument("--mae", action="store_true")
    ap.add_argument("--paths", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--months", type=int, default=24, help="bootstrap horizon in months")
    a = ap.parse_args(argv)
    f = load(a.trades)
    if f.empty:
        print("no filled trades in", a.trades)
        return 2
    rules = Rules(a.target1, a.target2, a.daily, a.maxloss, a.min_days, a.trailing, a.mae)
    risks = [float(x) for x in a.risk.split(",")]
    arrs = arrays(f)
    span = max((f["day"].max() - f["day"].min()).days / 30.4375, 1e-9)
    mean_r, per_month = f["R_w"].mean(), len(f) / span * f["R_w"].mean()
    print(f"rules: {rules}")
    print("LIMITATION: only closed-trade P&L is modelled; intraday floating-loss breaches are not seen, "
          "so real failure rates are higher. --mae adds a conservative adverse-excursion check.")
    print(f"trades {len(f)} | span {span:.1f} months | mean R_w/trade {mean_r:.3f} | E[R_w/month] {per_month:.2f}")
    for r in risks:
        pm = per_month * r
        print(f"  risk {r:g}%: E[{pm:.2f} %/month] -> months to +{a.target1:g}%: "
              f"{a.target1 / pm if pm > 0 else float('inf'):.1f}")
    pd.options.display.width = 200
    print("\nROLLING starts (every trading day as a start)")
    print(rolling(arrs, risks, rules).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nBOOTSTRAP ({a.paths} paths x {a.months} months, monthly blocks, seed {a.seed})")
    print(bootstrap(arrs, risks, rules, a.paths, a.seed, a.months).to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
