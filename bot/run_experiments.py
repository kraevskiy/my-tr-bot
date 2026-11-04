"""Kept for compatibility: same as `python -m smcbot experiments`."""
import sys

from smcbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["experiments", *sys.argv[1:]]))
