"""Shared training routine used by both src/train.py (standard, 17-variable
bank-full.csv, required) and src/train_extended.py (extended, 20-variable
bank-additional-full.csv, optional bonus comparison).

Flow, identical for both variants:
1. Load the raw dataset and do light cleaning (drop exact duplicates).
2. Split into train/test (stratified) using only the variant's features
   (never `duration`).
3. Train two candidate models (Logistic Regression, Random Forest), each with
   a very small hyperparameter search -- this is a hackathon, not a Kaggle
   competition, so the search is intentionally shallow.
4. Tune a decision threshold per candidate on a validation split carved out
   of the training data (see `tune_threshold`) -- with an ~88%/12% class
   imbalance, the default 0.5 cutoff is not where F1 is maximized, so this
   is a standard, cheap correction for imbalanced classification.
5. Compare both candidates (using their tuned threshold) with F1 and
   Balanced Accuracy on the held-out test set.
6. Retrain BOTH candidates on ALL available data (train+test) and serialize
   each Pipeline (preprocessing + model) to
   model/<variant>/<model_key>/model.joblib, model_key in
   {"logistic_regression", "random_forest"}. The hackathon requires training
   and comparing at least 2 models -- rather than deploying only the winner,
   the API exposes BOTH per variant so any of the 2x2 = 4 model/dataset
   combinations can be queried directly (see api/main.py).
7. Save each model's reference metrics and metadata to
   model/<variant>/<model_key>/metrics.json and model_info.json, plus an
   overall model/<variant>/comparison.json documenting which one would be
   recommended for production and why.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from preprocessing import TARGET, VARIANTS, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

RANDOM_STATE = 42
POS_LABEL = "yes"
MODEL_KEYS = ["logistic_regression", "random_forest"]


def load_data(variant: str) -> pd.DataFrame:
    cfg = VARIANTS[variant]
    df = pd.read_csv(DATA_DIR / cfg["data_file"], sep=cfg["csv_sep"])
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    print(f"[{variant}] Loaded {n_before} rows, dropped {n_dropped} exact duplicates -> {len(df)} rows")
    return df


def yes_proba(model, X) -> np.ndarray:
    classes = list(model.classes_)
    yes_idx = classes.index(POS_LABEL)
    return model.predict_proba(X)[:, yes_idx]


def tune_threshold(pipeline, X_val, y_val) -> float:
    """Find the probability cutoff that maximizes F1 for the "yes" class.

    With ~11-12% positives, the default 0.5 cutoff under-predicts "yes";
    scanning the precision-recall curve on a held-out validation split
    (never seen during fitting) gives an honest, principled threshold
    instead of guessing.
    """
    proba = yes_proba(pipeline, X_val)
    precision, recall, thresholds = precision_recall_curve(
        (y_val == POS_LABEL).astype(int), proba
    )
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_scores[:-1]))  # last point has no threshold
    return float(thresholds[best_idx])


def evaluate(model, X_test, y_test, threshold: float) -> dict:
    proba = yes_proba(model, X_test)
    y_pred = np.where(proba >= threshold, POS_LABEL, "no")

    return {
        "decision_threshold": threshold,
        "f1_score": float(f1_score(y_test, y_pred, pos_label=POS_LABEL)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=POS_LABEL)),
        "recall": float(recall_score(y_test, y_pred, pos_label=POS_LABEL)),
        "roc_auc": float(roc_auc_score((y_test == POS_LABEL).astype(int), proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["no", "yes"]).tolist(),
        "confusion_matrix_labels": ["no", "yes"],
    }


def build_model(model_key: str, params: dict = None):
    params = params or {}
    if model_key == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=params.get("model__C", 1.0),
        )
    # min_samples_leaf=10 keeps trees (and the serialized model) small --
    # without it, class_weight="balanced" lets trees grow very deep/large
    # (an early pass produced a 144MB model.joblib) for no real F1 benefit.
    # max_depth is also capped at 12: deeper values kept doubling the
    # serialized size while F1/Balanced Accuracy barely moved.
    return RandomForestClassifier(
        class_weight="balanced",
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=params.get("model__n_estimators", 300),
        max_depth=params.get("model__max_depth", 12),
    )


def run_training(variant: str):
    cfg = VARIANTS[variant]
    features = cfg["features"]
    print(f"\n{'='*70}\nTraining variant: {variant} -- {cfg['label']}\n{'='*70}")

    df = load_data(variant)

    # duration must never reach training: hard, explicit, verifiable check.
    assert "duration" not in features, "duration leaked into features!"
    print(f"Verified: duration NOT IN final_features ({len(features)} features used)")

    X = df[features]
    y = df[TARGET]

    print("\nClass distribution:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Carved out of the training data only -- never touches X_test -- so the
    # threshold is tuned honestly and test metrics stay a fair estimate.
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    grids = {}

    # --- Candidate 1: Logistic Regression (small grid) ---
    print("\n=== Training Logistic Regression (small grid) ===")
    lr_grid = GridSearchCV(
        build_pipeline(build_model("logistic_regression"), variant=variant),
        param_grid={"model__C": [0.1, 1.0, 3.0]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    print("Best LR params:", lr_grid.best_params_)
    grids["logistic_regression"] = lr_grid

    # --- Candidate 2: Random Forest (small grid) ---
    print("\n=== Training Random Forest (small grid) ===")
    rf_grid = GridSearchCV(
        build_pipeline(build_model("random_forest"), variant=variant),
        param_grid={
            "model__n_estimators": [200, 300],
            "model__max_depth": [8, 10, 12],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print("Best RF params:", rf_grid.best_params_)
    grids["random_forest"] = rf_grid

    # --- Tune a decision threshold per candidate, then evaluate on test ---
    print("\n=== Threshold tuning (validation split) + evaluation on held-out test set ===")
    comparison = {}
    thresholds = {}
    for model_key, grid in grids.items():
        # Refit the same hyperparameters on the smaller X_tr2 so X_val is
        # unseen, purely to pick a threshold -- the model used for the
        # reported test metrics is refit on the full X_train right after.
        val_model = clone(grid.best_estimator_)
        val_model.fit(X_tr2, y_tr2)
        threshold = tune_threshold(val_model, X_val, y_val)
        thresholds[model_key] = threshold

        test_model = clone(grid.best_estimator_)
        test_model.fit(X_train, y_train)
        metrics = evaluate(test_model, X_test, y_test, threshold)
        comparison[model_key] = metrics
        print(
            f"{model_key}: threshold={threshold:.3f}  F1={metrics['f1_score']:.4f}  "
            f"BalancedAcc={metrics['balanced_accuracy']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # Which one WOULD be recommended for production, if only one had to be
    # picked: prioritize F1, tie-break with balanced accuracy. If scores are
    # close (<0.01), prefer Logistic Regression for simplicity. This is
    # documentation/comparison only -- the hackathon requires training and
    # comparing >= 2 models; BOTH are retrained on full data and deployed
    # below so each can be queried directly via the API.
    recommended = max(comparison, key=lambda n: comparison[n]["f1_score"])
    f1_diff = abs(comparison["logistic_regression"]["f1_score"] - comparison["random_forest"]["f1_score"])
    if f1_diff < 0.01:
        recommended = "logistic_regression"
        print("\nF1 scores within 0.01 of each other -> logistic_regression recommended for simplicity")
    print(f"\nRecommended model (for reference): {recommended}")

    out_dir = MODEL_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Retrain BOTH candidates on ALL available data and save each ---
    for model_key, grid in grids.items():
        print(f"\n=== Retraining {model_key} on full dataset (train+test) ===")
        final_pipeline = build_pipeline(build_model(model_key, grid.best_params_), variant=variant)
        final_pipeline.fit(X, y)

        model_dir = out_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, model_dir / "model.joblib")
        print(f"Saved final model to {model_dir / 'model.joblib'}")

        reference_metrics = comparison[model_key]
        metrics_output = {
            "variant": variant,
            "dataset": cfg["label"],
            "model": model_key,
            "best_hyperparameters": grid.best_params_,
            "decision_threshold": thresholds[model_key],
            "reference_metrics_holdout_test": reference_metrics,
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_total_retrain": len(X),
        }
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics_output, f, indent=2)

        model_info = {
            "model": model_key,
            "variant": variant,
            "dataset": cfg["label"],
            "version": "1.0",
            "target": TARGET,
            "classes": list(final_pipeline.classes_),
            "duration_used": False,
            "features": features,
            "decision_threshold": thresholds[model_key],
            "f1_score": reference_metrics["f1_score"],
            "balanced_accuracy": reference_metrics["balanced_accuracy"],
            "recommended_for_production": model_key == recommended,
        }
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        print(f"Saved metrics + model info to {model_dir}")

    comparison_output = {
        "variant": variant,
        "dataset": cfg["label"],
        "recommended_model": recommended,
        "all_candidates": comparison,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total_retrain": len(X),
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison_output, f, indent=2)
    print(f"\nSaved comparison to {out_dir / 'comparison.json'}")

    return comparison_output
