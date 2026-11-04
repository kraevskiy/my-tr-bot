"""Kept for compatibility: same as `python -m smcbot sessions`."""
import sys

from smcbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["sessions", *sys.argv[1:]]))
