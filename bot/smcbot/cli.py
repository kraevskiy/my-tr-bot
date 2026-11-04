"""Single entry point:  python -m smcbot <command> [options]   (or `smcbot <command>` after pip install).

Commands
  backtest      one strategy run on your data            -> results/summary.txt, signals.csv, trades_<mode>.csv
  experiments   many variants from experiments.json       -> results/<run_id>/experiments.csv, report.md, ...
  analyze       deep HTML report for a run / variant      -> <run>/<variant>/analysis_<mode>.html
  neighbours    random +-pct neighbours of a preset + summary -> experiments JSON / robustness verdicts
  news          refresh the ForexFactory calendar         -> data/news/ff_calendar.csv
  sessions      Asia/Frankfurt/London/NY study            -> results/sessions_<SYMBOL>.csv
  presets       list strategy presets and what they change
  config        print the effective configuration (preset + .env overrides)
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import sys

COMMANDS = {
    "backtest": "smcbot.apps.backtest",
    "experiments": "smcbot.apps.experiments",
    "analyze": "smcbot.apps.analyze",
    "neighbours": "smcbot.apps.neighbours",
    "news": "smcbot.apps.news",
    "sessions": "smcbot.apps.sessions",
}


def _presets() -> int:
    from smcbot.config import PRESETS
    for name, kw in PRESETS.items():
        print(f"{name:9s} {json.dumps(kw, ensure_ascii=False) if kw else '(course defaults)'}")
    return 0


def _config() -> int:
    from smcbot.settings import load_settings
    s, cfg = load_settings()
    print(json.dumps({"settings": dataclasses.asdict(s),
                      "config": {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in dataclasses.asdict(cfg).items()}},
                     indent=1, ensure_ascii=False, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "presets":
        return _presets()
    if cmd == "config":
        return _config()
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}")
        return 2
    return int(importlib.import_module(COMMANDS[cmd]).main(rest) or 0)
