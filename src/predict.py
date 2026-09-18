"""Reusable prediction pipeline for the Bank Marketing model.

This module is the single source of truth for turning a raw feature
dictionary into a prediction, for any of the 2x2 = 4 model/dataset
combinations: variant in {"standard" (bank-full.csv, 17 vars, required),
"extended" (bank-additional-full.csv, 20 vars, bonus)} x model_key in
{"logistic_regression", "random_forest"}. It is imported directly by the
FastAPI app (api/main.py) -- the model logic is never duplicated inside the
API layer.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import VARIANTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

_models = {}
_model_info = {}


def _key(variant: str, model_key: str) -> str:
    return f"{variant}/{model_key}"


def load_model(variant: str, model_key: str):
    """Load a serialized Pipeline once per (variant, model_key) and cache it."""
    k = _key(variant, model_key)
    if k not in _models:
        _models[k] = joblib.load(MODEL_DIR / variant / model_key / "model.joblib")
    return _models[k]


def load_model_info(variant: str, model_key: str) -> dict:
    k = _key(variant, model_key)
    if k not in _model_info:
        with open(MODEL_DIR / variant / model_key / "model_info.json") as f:
            _model_info[k] = json.load(f)
    return _model_info[k]


def validate_input(input_dict: dict, variant: str = "standard") -> None:
    features = VARIANTS[variant]["features"]
    missing = [f for f in features if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if "duration" in input_dict:
        raise ValueError(
            "Field 'duration' is not accepted: it is excluded from this model "
            "because it causes data leakage."
        )


def predict(input_dict: dict, variant: str = "standard", model_key: str = "random_forest") -> dict:
    """Predict term-deposit subscription for a single client.

    Args:
        input_dict: dict with exactly the variant's feature keys.
        variant: "standard" (bank-full, 17 vars) or "extended"
            (bank-additional-full, 20 vars).
        model_key: "logistic_regression" or "random_forest".

    Returns:
        {
            "prediction": "yes" | "no",
            "probabilities": {"no": float, "yes": float}
        }
    """
    validate_input(input_dict, variant)

    model = load_model(variant, model_key)
    info = load_model_info(variant, model_key)
    features = VARIANTS[variant]["features"]

    row = {feature: input_dict[feature] for feature in features}
    X = pd.DataFrame([row], columns=features)

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    yes_idx = classes.index("yes")

    threshold = info["decision_threshold"]
    prediction = "yes" if proba[yes_idx] >= threshold else "no"

    return {
        "prediction": prediction,
        "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    standard_example = {
        "age": 41, "balance": 1200, "day": 15, "campaign": 2, "pdays": -1, "previous": 0,
        "job": "technician", "marital": "married", "education": "secondary",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "poutcome": "unknown",
    }
    extended_example = {
        "age": 41, "campaign": 2, "pdays": 999, "previous": 0,
        "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
        "euribor3m": 4.857, "nr.employed": 5191.0,
        "job": "technician", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "poutcome": "nonexistent",
    }

    for variant, example in [("standard", standard_example), ("extended", extended_example)]:
        for model_key in ["logistic_regression", "random_forest"]:
            print(f"{variant}/{model_key}:", predict(example, variant=variant, model_key=model_key))
