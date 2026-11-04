"""Backtest the strategy. Configuration comes from environment variables (.env).

  python -m smcbot backtest              # uses DATA_SOURCE, SYMBOLS, DATE_FROM ... from env
  python -m smcbot backtest --synthetic  # demo on random data (results mean nothing)

Outputs in RESULTS_DIR: signals.csv, trades_<mode>.csv, summary.txt
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


from smcbot.engine.backtest import run_backtest, summarize
from smcbot.config import Instrument
from smcbot.data.providers import load_symbol
from smcbot.settings import load_settings
from smcbot.strategy.setups import generate_signals
from smcbot.market import news as newsmod
from smcbot.data.synthetic import make_m1

DEFAULT_SPREADS = {"EURUSD": 1.0, "GBPUSD": 1.4}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="demo run on random data")
    a = ap.parse_args(argv)

    settings, cfg = load_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("smcbot")

    data, insts = {}, {}
    for sym in settings.symbols:
        if a.synthetic:
            data[sym] = make_m1(days=120, seed=len(data) + 1, price=1.085 if sym == "EURUSD" else 1.27)
        else:
            data[sym] = load_symbol(sym, settings)
        insts[sym] = Instrument(sym, default_spread_pips=DEFAULT_SPREADS.get(sym, 1.5))
        df = data[sym]
        if df.empty:
            log.error("%s: no data for the selected period", sym)
            return 1
        log.info("%s: %s M1 bars %s -> %s", sym, f"{len(df):,}", df.index[0], df.index[-1])

    out = Path(settings.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"Risk {cfg.risk_pct:.2%}/trade, daily stop {cfg.daily_max_loss_pct:.2%} or {cfg.daily_max_sl} SL, "
              f"min RR {cfg.min_rr}, symbols {', '.join(data)}")
        funnel = {}
        sigs = pd.concat([generate_signals(data[s], cfg, insts[s], funnel) for s in data],
                         ignore_index=True)
        print("\nFilter funnel (M5 shifts -> signals):")
        for k, v in sorted(funnel.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v}")
        cal = newsmod.load_default()
        if len(sigs) and cfg.news_mode != "off" and cal is not None:
            sigs, dropped = newsmod.news_filter(sigs, cal, cfg)
            print(f"  {'x_news':22s} {dropped}")
        if len(sigs) and cfg.holiday_mode != "off" and cal is not None:
            sigs, dropped = newsmod.holiday_filter(sigs, cal, cfg)
            print(f"  {'x_holiday':22s} {dropped}")
        if not len(sigs):
            print("No signals.")
        else:
            sigs.drop(columns=["signal_ns", "ext_ns"], errors="ignore").to_csv(out / "signals.csv", index=False)
            print(f"\nSignals: {len(sigs)}")
            for mode in settings.modes:
                tr = run_backtest(sigs, data, cfg, insts, mode)
                if cal is not None and len(tr):
                    tr = newsmod.annotate(tr, cal)
                tr.to_csv(out / f"trades_{mode}.csv", index=False)
                print(f"\n=== Stop management variant {mode} ===")
                print("orders:", tr["status"].value_counts().to_dict())
                print(summarize(tr).round(2).to_string())
                if "R_w" in tr and (tr.get("risk_mult", 1.0) != 1.0).any():
                    f = tr[tr["status"] == "filled"]
                    eq = f["R_w"].cumsum()
                    print(f"risk-weighted by grade (A x{cfg.grade_risk_a}, B x{cfg.grade_risk_b}): "
                          f"total {f['R_w'].sum():.2f}R, maxDD {(eq.cummax() - eq).max():.2f}R; "
                          f"A: {(f['grade'] == 'A').sum()} trades avg {f.loc[f['grade'] == 'A', 'R_result'].mean():.2f}R, "
                          f"B: {(f['grade'] == 'B').sum()} trades avg {f.loc[f['grade'] == 'B', 'R_result'].mean():.2f}R")
                if mode == cfg.be_mode:
                    for col, title in (("setup", "setup"), ("killzone", "killzone"),
                                       ("tier", "swept liquidity tier"), ("symbol", "symbol")):
                        print(f"\nby {title}:\n", summarize(tr, col).round(2).to_string())
    text = buf.getvalue()
    print(text)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    log.info("results written to %s", out)
    return 0

