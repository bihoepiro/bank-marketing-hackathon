"""Train the STANDARD model: bank-full.csv, 17 variables.

This is the primary, required dataset (standardized by the professor for
the whole class). Run with:

    python src/train.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from train_common import run_training  # noqa: E402

if __name__ == "__main__":
    run_training(variant="standard")
