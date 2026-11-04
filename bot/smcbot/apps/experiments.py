"""Run many strategy variants on the same data and compare them.

  python -m smcbot experiments                       # all variants from experiments.json
  python -m smcbot experiments --only V3,V4 --modes b
  python -m smcbot experiments --file my_variants.json --synthetic

Detectors (structure, zones, pools, shifts) are computed once per symbol; each variant
only re-applies the filters/entry/exit rules, so dozens of variants are cheap.

Output: RESULTS_DIR/<run_id>/
  experiments.csv        one row per variant x stop mode (IS / OOS split by OOS_FROM)
  report.md              ranked table + funnel per variant, readable by a human (and by Claude)
  config.json            base config + data period
  <variant>/signals.csv, trades_<mode>.csv, funnel.json
  baseline/candidates.csv.gz   every M5 Shift with features and ALL reasons it was rejected
Also appends to RESULTS_DIR/experiments_history.csv.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


from smcbot.engine.backtest import forward_outcomes, m5_swings_for, metrics, run_backtest
from smcbot.config import Instrument
from smcbot.data.providers import load_symbol
from smcbot.settings import load_settings
from smcbot.strategy.setups import SymbolModels
from smcbot.market.smt import smt_flags
from smcbot.market import news as newsmod
from smcbot.data.synthetic import make_m1

DEFAULT_SPREADS = {"EURUSD": 1.0, "GBPUSD": 1.4}
LOG_VARIANTS = {"baseline"}          # candidates log is large; keep it for the baseline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="experiments.json")
    ap.add_argument("--only", default="", help="comma separated variant names")
    ap.add_argument("--modes", default="", help="stop modes to test, e.g. ab (default from file)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--log-all", action="store_true", help="write candidates log for every variant")
    a = ap.parse_args(argv)

    settings, cfg0 = load_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("smcbot.exp")
    spec = json.loads(Path(a.file).read_text(encoding="utf-8"))
    from smcbot.config import PRESETS
    base_cfg = cfg0.replace(**PRESETS[spec.get("preset", "course")]).replace(**spec.get("base", {}))
    modes = a.modes or spec.get("modes", "b")
    variants = spec["variants"]
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        variants = [v for v in variants if v["name"] in keep]
    oos_from = os.getenv("OOS_FROM", "2026-01-01").strip() or None

    data, insts = {}, {}
    for sym in settings.symbols:
        data[sym] = make_m1(days=160, seed=len(data) + 1, price=1.085 if sym == "EURUSD" else 1.27) \
            if a.synthetic else load_symbol(sym, settings)
        insts[sym] = Instrument(sym, default_spread_pips=DEFAULT_SPREADS.get(sym, 1.5))
        log.info("%s: %s M1 bars %s -> %s", sym, f"{len(data[sym]):,}", data[sym].index[0], data[sym].index[-1])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(settings.results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "base_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(base_cfg).items()},
        "oos_from": oos_from, "modes": modes, "file": a.file,
        "data": {s: [str(d.index[0]), str(d.index[-1]), len(d)] for s, d in data.items()},
    }, indent=2, default=str), encoding="utf-8")

    cal = newsmod.load_default()
    log.info("news calendar: %s", f"{len(cal):,} events" if cal is not None else "not found (news columns off)")
    t0 = time.time()
    model_cache = {}

    def models_for(c):
        """Detectors depend only on the entry/sync timeframes; build once per tf_entry."""
        key = tuple(c.tf_entry)
        if key not in model_cache:
            mc = base_cfg.replace(tf_entry=key)
            model_cache[key] = {s: SymbolModels(data[s], mc, insts[s]) for s in data}
        return model_cache[key]

    models_for(base_cfg)
    swings = m5_swings_for(data, base_cfg)
    log.info("detectors ready in %.0fs", time.time() - t0)

    rows, funnels = [], {}
    for v in variants:
        name = v["name"]
        cfg = base_cfg.replace(**v.get("params", {}))
        vdir = out / name
        vdir.mkdir(exist_ok=True)
        funnel, cand_logs = {}, [] if (a.log_all or name in LOG_VARIANTS or v.get("log")) else None
        sigs = pd.concat([models_for(cfg)[s].signals(cfg, funnel, cand_logs) for s in data], ignore_index=True)
        if len(sigs):
            sigs["smt"] = smt_flags(sigs, data)
            if cfg.smt_mode == "require":
                funnel["x_no_smt"] = int((sigs["smt"] != 1).sum())
                sigs = sigs[sigs["smt"] == 1].reset_index(drop=True)
                funnel["signal"] = len(sigs)
            if cfg.news_mode != "off":
                if cal is None:
                    raise SystemExit("news_mode needs data/news/ff_calendar.csv or NEWS_CSV (run: python -m smcbot news)")
                sigs, dropped = newsmod.news_filter(sigs, cal, cfg)
                funnel["x_news"] = dropped
                funnel["signal"] = len(sigs)
            if cfg.holiday_mode != "off":
                sigs, dropped = newsmod.holiday_filter(sigs, cal, cfg)
                funnel["x_holiday"] = dropped
                funnel["signal"] = len(sigs)
            sigs = newsmod.annotate(sigs, cal)
        funnels[name] = funnel
        (vdir / "funnel.json").write_text(json.dumps(funnel, indent=1), encoding="utf-8")
        if cand_logs:
            cl = pd.concat(cand_logs, ignore_index=True)
            cl = forward_outcomes(cl, data, insts, cfg.shadow_hours)
            cl["smt"] = smt_flags(cl, data)
            cl = newsmod.annotate(cl, cal)
            cl.drop(columns=[c for c in ("signal_ns", "t_shift", "ext_ns") if c in cl], errors="ignore") \
              .to_csv(vdir / "candidates.csv.gz", index=False, compression="gzip")
        if len(sigs):
            sigs.drop(columns=["signal_ns", "ext_ns"], errors="ignore").to_csv(vdir / "signals.csv", index=False)
        for mode in modes:
            tr = run_backtest(sigs, data, cfg, insts, mode, swings) if len(sigs) else pd.DataFrame()
            if len(tr) and cal is not None:
                tr = newsmod.annotate(tr, cal)
            if len(tr):
                tr.drop(columns=["signal_ns", "ext_ns"], errors="ignore").to_csv(vdir / f"trades_{mode}.csv", index=False)
            m = metrics(tr, oos_from)
            rows.append({"variant": name, "group": v.get("group", ""), "mode": mode, **m,
                         "params": json.dumps(v.get("params", {})), "why": v.get("why", "")})
        log.info("%-16s signals=%4d  %s", name, len(sigs),
                 " ".join(f"{r['mode']}:{r.get('totalR', 0):+.1f}R/{r.get('trades', 0)}"
                          + (f" (risk-weighted {r['w_totalR']:+.1f}R)" if pd.notna(r.get("w_totalR", np.nan)) else "")
                          for r in rows if r["variant"] == name))

    res = pd.DataFrame(rows)
    res.insert(0, "run_id", run_id)
    res.to_csv(out / "experiments.csv", index=False)
    hist = Path(settings.results_dir) / "experiments_history.csv"
    res.to_csv(hist, mode="a", index=False, header=not hist.exists())
    write_report(out, res, funnels, oos_from, data)
    # deep HTML report for the variants marked "log": true (the candidates under study)
    from smcbot.research.analytics import build_report
    for v in variants:
        if v.get("log"):
            for mode in modes:
                tp = out / v["name"] / f"trades_{mode}.csv"
                if tp.exists():
                    try:
                        build_report(tp, out / v["name"] / f"analysis_{mode}.html", f"{v['name']} ({mode})",
                                     out, mode, out / v["name"] / "candidates.csv.gz")
                        log.info("analysis -> %s", out / v["name"] / f"analysis_{mode}.html")
                    except Exception as e:      # the report must never break a run
                        log.warning("analysis for %s failed: %s", v["name"], e)
    print((out / "report.md").read_text(encoding="utf-8"))
    log.info("done in %.0fs -> %s", time.time() - t0, out)
    return 0


def write_report(out: Path, res: pd.DataFrame, funnels: dict, oos_from, data) -> None:
    cols = ["variant", "mode", "signals", "fill%", "trades", "win%", "avgR", "t_stat", "totalR", "PF", "maxDD_R",
            "w_totalR", "w_maxDD_R", "w_t_stat",
            "IS_trades", "IS_totalR", "OOS_trades", "OOS_totalR", "y2024", "y2025", "y2026", "unfilled_shadow2R"]
    cols = [c for c in cols if c in res]
    lines = [f"# Experiments {out.name}", ""]
    lines += [f"Data: " + "; ".join(f"{s} {d.index[0]:%Y-%m-%d}..{d.index[-1]:%Y-%m-%d}" for s, d in data.items()),
              f"IS = before {oos_from}, OOS = from {oos_from}. Ranked by IS_totalR (choose on IS, confirm on OOS).",
              "t_stat = avgR / standard error; |t| < 2 means the result is not distinguishable from zero.", ""]
    best = res.sort_values("IS_totalR" if "IS_totalR" in res else "totalR", ascending=False)
    lines.append(best[cols].round(2).to_markdown(index=False) if hasattr(best, "to_markdown") else best[cols].round(2).to_string(index=False))
    lines += ["", "## Funnel per variant", "", "shift_m5 = all M5 Shifts; signal = passed everything; x_* = how many candidates failed that filter "
              "(one candidate can fail several, so x_* do not add up).", ""]
    fk = sorted({k for f in funnels.values() for k in f})
    ft = pd.DataFrame(funnels).reindex(fk).fillna(0).astype(int)
    lines.append(ft.to_string())
    lines += ["", "Notes: fill% = filled / signals; unfilled_shadow2R = average result of a market entry with "
              "+2R/-1R for signals that never filled (positive = we are missing good trades); "
              "avg_mfe_R = how far price went in our favour on average before exit."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

