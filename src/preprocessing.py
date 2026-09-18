"""Feature definitions and preprocessing pipeline for the Bank Marketing model.

Two dataset variants are supported, per the professor's guidance:

- "standard": bank-full.csv, the 17-variable dataset (16 features + `y`) the
  class was asked to standardize on. This is the required, primary variant.
- "extended": bank-additional-full.csv, the 21-variable dataset (20 features
  + `y`, adds social/economic context attributes). Optional bonus: teams
  that implement and compare both, with both deployed in parallel on GCP,
  get +4 participation points.

`duration` is intentionally excluded from both feature sets: it is only
known after the sales call ends, so using it would leak information that is
not available at prediction time (see README section 3 "Data leakage").
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "y"

# Excluded on purpose in both variants: only known after the call happens.
LEAKAGE_FEATURES = ["duration"]

# --- "standard" variant: bank-full.csv (17 variables) ---
STANDARD_NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
STANDARD_CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
STANDARD_FEATURES = STANDARD_NUMERIC_FEATURES + STANDARD_CATEGORICAL_FEATURES

# --- "extended" variant: bank-additional-full.csv (20 variables, bonus) ---
EXTENDED_NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
EXTENDED_CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
EXTENDED_FEATURES = EXTENDED_NUMERIC_FEATURES + EXTENDED_CATEGORICAL_FEATURES

# Hard guarantee required by the hackathon: duration must never reach any model.
assert "duration" not in STANDARD_FEATURES, "duration must be excluded (data leakage)"
assert "duration" not in EXTENDED_FEATURES, "duration must be excluded (data leakage)"

VARIANTS = {
    "standard": {
        "label": "standard (bank-full.csv, 17 variables)",
        "data_file": "bank-full.csv",
        "csv_sep": ";",
        "numeric": STANDARD_NUMERIC_FEATURES,
        "categorical": STANDARD_CATEGORICAL_FEATURES,
        "features": STANDARD_FEATURES,
    },
    "extended": {
        "label": "extended (bank-additional-full.csv, 20 variables, bonus)",
        "data_file": "bank-additional-full.csv",
        "csv_sep": ";",
        "numeric": EXTENDED_NUMERIC_FEATURES,
        "categorical": EXTENDED_CATEGORICAL_FEATURES,
        "features": EXTENDED_FEATURES,
    },
}


def build_preprocessor(variant: str = "standard") -> ColumnTransformer:
    """ColumnTransformer shared by every candidate model for a given variant.

    Scaling numeric features is unnecessary for the tree model but harmless,
    and lets both candidates reuse the exact same preprocessing step.
    """
    cfg = VARIANTS[variant]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["categorical"]),
        ]
    )


def build_pipeline(model, variant: str = "standard") -> Pipeline:
    """Wrap a classifier together with the shared preprocessor.

    Serializing this whole Pipeline means the API never has to reimplement
    the encoding/scaling logic used at training time.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(variant)),
            ("model", model),
        ]
    )
