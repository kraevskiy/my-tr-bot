"""Kept for compatibility: same as `python -m smcbot news`."""
import sys

from smcbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["news", *sys.argv[1:]]))
