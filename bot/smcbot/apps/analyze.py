"""Build the deep-analysis HTML report for one variant of an experiments run.

  python -m smcbot analyze results/20261102_172956 A_spread12          # stop mode b
  python -m smcbot analyze results/20261102_172956 A_spread12 --mode a
  python -m smcbot analyze results/<run> --best                        # best variant by t-stat

Writes <run>/<variant>/analysis_<mode>.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


from smcbot.research.analytics import build_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--mode", default="b")
    ap.add_argument("--best", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    variant = a.variant
    if a.best or not variant:
        e = pd.read_csv(run / "experiments.csv")
        e = e[e["mode"] == a.mode]
        variant = e.sort_values("t_stat" if "t_stat" in e else "totalR", ascending=False)["variant"].iloc[0]
    trades = run / variant / f"trades_{a.mode}.csv"
    cands = next(iter(run.glob("*/candidates.csv.gz")), None)
    out = run / variant / f"analysis_{a.mode}.html"
    res = build_report(trades, out, f"{variant} (stop mode {a.mode}) — {run.name}", run, a.mode, cands)
    print(f"{variant}: {res['headline']}")
    print(f"report -> {out}")
    return 0

