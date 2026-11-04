"""Kept for compatibility: same as `python -m smcbot analyze`."""
import sys

from smcbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["analyze", *sys.argv[1:]]))
