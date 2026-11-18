"""Compatibility wrapper for the comparison pipeline.

Prefer the packaged CLI::

    uv run scanguard train
    uv run scanguard train --full
"""

import sys

from scanguard.training.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
