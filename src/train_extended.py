"""Train the EXTENDED model: bank-additional-full.csv, 20 variables.

Optional bonus: comparing this dataset against the standard bank-full.csv
(17 variables) and deploying both models in parallel on GCP is worth +4
participation points per the professor's guidance. Run with:

    python src/train_extended.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from train_common import run_training  # noqa: E402

if __name__ == "__main__":
    run_training(variant="extended")
