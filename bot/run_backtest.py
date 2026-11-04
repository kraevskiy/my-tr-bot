"""Kept for compatibility: same as `python -m smcbot backtest`."""
import sys

from smcbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["backtest", *sys.argv[1:]]))
