"""Parameter-neighbourhood robustness test: random +-pct variants of a preset, then a summary.

  python -m smcbot neighbours --n 60 --pct 0.2 --seed 7 --preset v6 --out results/nb60.json
  python -m smcbot experiments --file results/nb60.json
  python -m smcbot neighbours --summarize results/<run_dir>      # reads <run_dir>/experiments.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smcbot.config import PRESETS, Config

FLOATS = ["min_sl_pips", "max_spread_frac", "grade_po3_pips", "grade_delay_min", "af_body_k",
          "af_reclaim_body_k", "af_adr_min", "af_tp_r", "af_min_sl_pips"]
INTS = ["cont_window_bars", "af_fade_bars"]
TP_MODES = ["fixed2.5", "fixed3", "fixed3.5"]   # the 2R partial stays below every choice


def generate(n: int, pct: float, seed: int, preset: str) -> dict:
    base = Config().replace(**PRESETS[preset])
    rng = np.random.default_rng(seed)
    variants = [{"name": preset.upper(), "group": "control", "why": f"{preset} as published", "params": {}}]
    for i in range(1, n + 1):
        p = {k: round(float(rng.uniform(getattr(base, k) * (1 - pct), getattr(base, k) * (1 + pct))), 4)
             for k in FLOATS}
        p.update({k: max(1, int(round(rng.uniform(getattr(base, k) * (1 - pct), getattr(base, k) * (1 + pct)))))
                  for k in INTS})
        p["tp_mode"] = str(rng.choice(TP_MODES))
        Config().replace(**p)                    # raises on unknown keys
        variants.append({"name": f"NB_{i:03d}", "group": "nb", "why": f"random +-{pct:.0%} neighbour", "params": p})
    return {"_about": f"{n} random +-{pct:.0%} neighbours of {preset} (seed {seed}); summarise with "
                      "`neighbours --summarize <run_dir>`",
            "preset": preset, "base": {}, "modes": "c", "variants": variants}


def verdicts(nb: pd.DataFrame, ctl: pd.Series, tot: str, dd: str) -> list[tuple[str, bool]]:
    v5, rank = ctl[tot], (nb[tot] <= ctl[tot]).mean()
    out = [(">=80% neighbours w_totalR>0", (nb[tot] > 0).mean() >= 0.8),
           ("median w_totalR >= 70% of V5", nb[tot].median() >= 0.7 * v5)]
    if "OOS_totalR" in nb:
        out.append((">=70% OOS positive", (nb["OOS_totalR"] > 0).mean() >= 0.7))
    out += [("p95 w_maxDD_R <= 2x V5", nb[dd].abs().quantile(0.95) <= 2 * abs(ctl[dd])),
            ("V5 below 90th percentile", rank < 0.9)]
    return out


def summarize(run_dir: str) -> int:
    e = pd.read_csv(Path(run_dir) / "experiments.csv")
    if "mode" in e and (e["mode"] == "c").any():
        e = e[e["mode"] == "c"]
    tot, dd = ("w_totalR", "w_maxDD_R") if "w_totalR" in e else ("totalR", "maxDD_R")
    nb, ctl = e[e["variant"].str.startswith("NB_")], e[~e["variant"].str.startswith("NB_")].iloc[0]   # control = the preset row
    print(f"neighbours: {len(nb)}")
    for col in [tot, "OOS_totalR", "y2024", "y2025", "y2026"]:
        if col in nb:
            print(f"share {col} > 0: {(nb[col] > 0).mean():.1%}")
    for col in [tot, dd]:
        q = nb[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2)
        print(f"{col:10s} " + "  ".join(f"p{int(k * 100)}={x}" for k, x in q.items()) + f"  | V5={ctl[col]:.2f}")
    print(f"V5 percentile rank on {tot}: {(nb[tot] <= ctl[tot]).mean():.1%}")
    for name, ok in verdicts(nb, ctl, tot, dd):
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preset", default="v6")
    ap.add_argument("--out", default="results/neighbours.json")
    ap.add_argument("--summarize", metavar="RUN_DIR", help="summarise <RUN_DIR>/experiments.csv instead")
    a = ap.parse_args(argv)
    if a.summarize:
        return summarize(a.summarize)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(generate(a.n, a.pct, a.seed, a.preset), indent=1), encoding="utf-8")
    print(f"{a.n} neighbours -> {out}")
    return 0
