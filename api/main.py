"""FastAPI service for the Bank Marketing term-deposit prediction model.

All 4 models are trained offline (src/train.py / src/train_extended.py ->
model/<variant>/<model_key>/model.joblib). This API only loads those
already-trained models and serves predictions; it never trains or retrains
anything.

The hackathon requires training and comparing at least 2 models (Part 1).
Rather than deploying only the winner, this API loads and serves BOTH
trained models for BOTH dataset variants -- 2x2 = 4 combinations, all
running in parallel inside this one process:

- POST /predict                              -> standard + random_forest
                                                 (required endpoint; the
                                                 recommended model for the
                                                 required, standardized
                                                 dataset)
- POST /predict/standard/logistic_regression -> standard + logistic_regression
- POST /predict/extended                     -> extended + random_forest
                                                 (bonus dataset, recommended
                                                 model)
- POST /predict/extended/logistic_regression -> extended + logistic_regression

Mirrored for introspection: GET /model-info, /model-info/standard/logistic_regression,
/model-info/extended, /model-info/extended/logistic_regression.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from predict import predict as run_predict  # noqa: E402
from predict import load_model  # noqa: E402
from preprocessing import VARIANTS  # noqa: E402
from train_common import MODEL_KEYS  # noqa: E402

MODEL_DIR = ROOT / "model"
model_info_cache: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all 4 models once at startup (not per-request) so all of them
    # are served in parallel by this single process.
    for variant in VARIANTS:
        for model_key in MODEL_KEYS:
            load_model(variant, model_key)
            info_path = MODEL_DIR / variant / model_key / "model_info.json"
            with open(info_path) as f:
                model_info_cache[f"{variant}/{model_key}"] = json.load(f)
    yield


app = FastAPI(
    title="Bank Marketing Prediction API",
    description=(
        "Predicts whether a client will subscribe to a term deposit. "
        "Serves 4 models in parallel (2 datasets x 2 algorithms): "
        "POST /predict (standard + random_forest, required), "
        "POST /predict/standard/logistic_regression, "
        "POST /predict/extended (extended + random_forest, bonus), "
        "POST /predict/extended/logistic_regression."
    ),
    version="1.0",
    lifespan=lifespan,
)


# --- Pydantic input schemas, one per dataset variant (shared across both models of that variant) ---

class PredictionInputStandard(BaseModel):
    """Input schema for standard-variant models (bank-full.csv, 17 variables)."""

    age: int = Field(..., ge=17, le=100, description="Client age in years")
    balance: int = Field(..., description="Average yearly balance, in euros")
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., description="Days since last contact from a previous campaign (-1 = never contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")

    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["no", "yes"]
    housing: Literal["no", "yes"]
    loan: Literal["no", "yes"]
    contact: Literal["cellular", "telephone", "unknown"]
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    poutcome: Literal["failure", "other", "success", "unknown"]

    model_config = {
        "extra": "forbid",  # rejects unknown fields, e.g. "duration" (leakage)
        "json_schema_extra": {
            "example": {
                "age": 41, "balance": 1200, "day": 15, "campaign": 2, "pdays": -1, "previous": 0,
                "job": "technician", "marital": "married", "education": "secondary",
                "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                "month": "may", "poutcome": "unknown",
            }
        },
    }


class PredictionInputExtended(BaseModel):
    """Input schema for extended-variant models (bank-additional-full.csv, 20 variables)."""

    age: int = Field(..., ge=17, le=100, description="Client age in years")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., description="Days since last contact from a previous campaign (999 = never contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    emp_var_rate: float = Field(..., alias="emp.var.rate", description="Employment variation rate (quarterly)")
    cons_price_idx: float = Field(..., alias="cons.price.idx", description="Consumer price index (monthly)")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", description="Consumer confidence index (monthly)")
    euribor3m: float = Field(..., description="Euribor 3-month rate (daily)")
    nr_employed: float = Field(..., alias="nr.employed", description="Number of employees (quarterly)")

    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ]
    default: Literal["no", "unknown", "yes"]
    housing: Literal["no", "unknown", "yes"]
    loan: Literal["no", "unknown", "yes"]
    contact: Literal["cellular", "telephone"]
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    poutcome: Literal["failure", "nonexistent", "success"]

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",  # rejects unknown fields, e.g. "duration" (leakage)
        "json_schema_extra": {
            "example": {
                "age": 41, "campaign": 2, "pdays": 999, "previous": 0,
                "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
                "euribor3m": 4.857, "nr.employed": 5191.0,
                "job": "technician", "marital": "married", "education": "university.degree",
                "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                "month": "may", "day_of_week": "mon", "poutcome": "nonexistent",
            }
        },
    }


class PredictionOutput(BaseModel):
    prediction: str
    probabilities: dict[str, float]


@app.get("/")
def root():
    return {
        "service": "Bank Marketing Prediction API",
        "status": "running",
        "models": {
            "standard + random_forest (recommended)": "POST /predict",
            "standard + logistic_regression": "POST /predict/standard/logistic_regression",
            "extended + random_forest (recommended, bonus)": "POST /predict/extended",
            "extended + logistic_regression (bonus)": "POST /predict/extended/logistic_regression",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return model_info_cache["standard/random_forest"]


@app.get("/model-info/standard/logistic_regression")
def model_info_standard_lr():
    return model_info_cache["standard/logistic_regression"]


@app.get("/model-info/extended")
def model_info_extended():
    return model_info_cache["extended/random_forest"]


@app.get("/model-info/extended/logistic_regression")
def model_info_extended_lr():
    return model_info_cache["extended/logistic_regression"]


def _run_predict(payload: BaseModel, variant: str, model_key: str) -> dict:
    input_dict = payload.model_dump(by_alias=True)
    assert "duration" not in VARIANTS[variant]["features"]  # defense in depth

    try:
        return run_predict(input_dict, variant=variant, model_key=model_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict", response_model=PredictionOutput)
def predict_standard_rf(payload: PredictionInputStandard):
    """Required endpoint. standard dataset (17 vars) + random_forest (recommended)."""
    return _run_predict(payload, "standard", "random_forest")


@app.post("/predict/standard/logistic_regression", response_model=PredictionOutput)
def predict_standard_lr(payload: PredictionInputStandard):
    """standard dataset (17 vars) + logistic_regression."""
    return _run_predict(payload, "standard", "logistic_regression")


@app.post("/predict/extended", response_model=PredictionOutput)
def predict_extended_rf(payload: PredictionInputExtended):
    """Bonus. extended dataset (20 vars) + random_forest (recommended)."""
    return _run_predict(payload, "extended", "random_forest")


@app.post("/predict/extended/logistic_regression", response_model=PredictionOutput)
def predict_extended_lr(payload: PredictionInputExtended):
    """Bonus. extended dataset (20 vars) + logistic_regression."""
    return _run_predict(payload, "extended", "logistic_regression")
