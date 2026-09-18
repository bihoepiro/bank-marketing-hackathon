"""Automated test client for the Bank Marketing Prediction API.

All 4 trained models run in parallel behind the SAME API (2 datasets x 2
algorithms):
    POST /predict                              -> standard + random_forest (required)
    POST /predict/standard/logistic_regression -> standard + logistic_regression
    POST /predict/extended                     -> extended + random_forest (bonus)
    POST /predict/extended/logistic_regression -> extended + logistic_regression

This script sends several distinct client profiles to each endpoint and
prints the result. Works against a local server or a deployed Cloud Run URL.

Usage:
    python client/test_api.py --url http://127.0.0.1:8000                    # tests all 4
    python client/test_api.py --url https://<cloud-run-url>                  # tests all 4
    python client/test_api.py --url http://127.0.0.1:8000 --model standard-rf
    python client/test_api.py --url http://127.0.0.1:8000 --model standard-lr
    python client/test_api.py --url http://127.0.0.1:8000 --model extended-rf
    python client/test_api.py --url http://127.0.0.1:8000 --model extended-lr
"""

import argparse
import sys

import requests

# Six distinct client profiles for the "standard" variant (bank-full.csv, 17 vars).
STANDARD_CASES = [
    {
        "name": "Retired client, positive balance, previous campaign success",
        "payload": {
            "age": 65, "balance": 3200, "day": 12, "campaign": 1, "pdays": 90, "previous": 2,
            "job": "retired", "marital": "married", "education": "primary",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "mar", "poutcome": "success",
        },
    },
    {
        "name": "Young student, never contacted before",
        "payload": {
            "age": 21, "balance": 150, "day": 5, "campaign": 3, "pdays": -1, "previous": 0,
            "job": "student", "marital": "single", "education": "secondary",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "jul", "poutcome": "unknown",
        },
    },
    {
        "name": "Blue-collar worker, negative balance, many contacts",
        "payload": {
            "age": 45, "balance": -350, "day": 20, "campaign": 8, "pdays": -1, "previous": 0,
            "job": "blue-collar", "marital": "married", "education": "primary",
            "default": "no", "housing": "yes", "loan": "yes", "contact": "unknown",
            "month": "may", "poutcome": "unknown",
        },
    },
    {
        "name": "Management profile, previous campaign failure",
        "payload": {
            "age": 38, "balance": 2100, "day": 8, "campaign": 2, "pdays": 180, "previous": 3,
            "job": "management", "marital": "divorced", "education": "tertiary",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "apr", "poutcome": "failure",
        },
    },
    {
        "name": "Self-employed, has credit in default",
        "payload": {
            "age": 52, "balance": 50, "day": 28, "campaign": 1, "pdays": -1, "previous": 0,
            "job": "self-employed", "marital": "married", "education": "tertiary",
            "default": "yes", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "jun", "poutcome": "unknown",
        },
    },
    {
        "name": "Entrepreneur, illiterate-equivalent (primary) education",
        "payload": {
            "age": 58, "balance": 4200, "day": 3, "campaign": 4, "pdays": -1, "previous": 0,
            "job": "entrepreneur", "marital": "married", "education": "primary",
            "default": "no", "housing": "yes", "loan": "no", "contact": "telephone",
            "month": "aug", "poutcome": "unknown",
        },
    },
]

# Six distinct client profiles for the "extended" variant (bank-additional-full.csv, 20 vars).
EXTENDED_CASES = [
    {
        "name": "Retired client, previous campaign success",
        "payload": {
            "age": 65, "campaign": 1, "pdays": 3, "previous": 1,
            "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2,
            "euribor3m": 1.313, "nr.employed": 5099.1,
            "job": "retired", "marital": "married", "education": "basic.4y",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "mar", "day_of_week": "tue", "poutcome": "success",
        },
    },
    {
        "name": "Young student, never contacted before",
        "payload": {
            "age": 21, "campaign": 3, "pdays": 999, "previous": 0,
            "emp.var.rate": 1.4, "cons.price.idx": 93.918, "cons.conf.idx": -42.7,
            "euribor3m": 4.963, "nr.employed": 5228.1,
            "job": "student", "marital": "single", "education": "high.school",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "jul", "day_of_week": "thu", "poutcome": "nonexistent",
        },
    },
    {
        "name": "Blue-collar worker, high campaign contacts",
        "payload": {
            "age": 45, "campaign": 8, "pdays": 999, "previous": 0,
            "emp.var.rate": 1.4, "cons.price.idx": 93.444, "cons.conf.idx": -36.1,
            "euribor3m": 4.964, "nr.employed": 5228.1,
            "job": "blue-collar", "marital": "married", "education": "basic.9y",
            "default": "unknown", "housing": "yes", "loan": "yes", "contact": "telephone",
            "month": "may", "day_of_week": "mon", "poutcome": "nonexistent",
        },
    },
    {
        "name": "Management profile, previous campaign failure",
        "payload": {
            "age": 38, "campaign": 2, "pdays": 6, "previous": 2,
            "emp.var.rate": -0.1, "cons.price.idx": 93.2, "cons.conf.idx": -42.0,
            "euribor3m": 4.153, "nr.employed": 5195.8,
            "job": "management", "marital": "divorced", "education": "university.degree",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "apr", "day_of_week": "wed", "poutcome": "failure",
        },
    },
    {
        "name": "Self-employed, unknown housing/loan status",
        "payload": {
            "age": 52, "campaign": 1, "pdays": 999, "previous": 0,
            "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
            "euribor3m": 4.857, "nr.employed": 5191.0,
            "job": "self-employed", "marital": "married", "education": "professional.course",
            "default": "no", "housing": "unknown", "loan": "unknown", "contact": "cellular",
            "month": "jun", "day_of_week": "fri", "poutcome": "nonexistent",
        },
    },
    {
        "name": "Entrepreneur, illiterate education, no prior contact",
        "payload": {
            "age": 58, "campaign": 4, "pdays": 999, "previous": 0,
            "emp.var.rate": 1.4, "cons.price.idx": 94.465, "cons.conf.idx": -41.8,
            "euribor3m": 4.959, "nr.employed": 5228.1,
            "job": "entrepreneur", "marital": "married", "education": "illiterate",
            "default": "no", "housing": "yes", "loan": "no", "contact": "telephone",
            "month": "aug", "day_of_week": "tue", "poutcome": "nonexistent",
        },
    },
]

TEST_CASES_BY_VARIANT = {"standard": STANDARD_CASES, "extended": EXTENDED_CASES}
# The 4 deployed model/dataset combinations: (variant, model_key, endpoint path).
# Payload shape only depends on the variant, so both model_keys of a variant
# reuse the same TEST_CASES_BY_VARIANT[variant] list.
COMBOS = {
    "standard-rf": ("standard", "random_forest", "/predict"),
    "standard-lr": ("standard", "logistic_regression", "/predict/standard/logistic_regression"),
    "extended-rf": ("extended", "random_forest", "/predict/extended"),
    "extended-lr": ("extended", "logistic_regression", "/predict/extended/logistic_regression"),
}


def run_combo(base_url: str, combo_name: str) -> int:
    variant, model_key, endpoint = COMBOS[combo_name]
    cases = TEST_CASES_BY_VARIANT[variant]
    print(f"\n=== {variant} + {model_key}  (POST {endpoint}) ===\n")

    passed = 0
    for i, case in enumerate(cases, start=1):
        print(f"Test {i}: {case['name']}")
        try:
            r = requests.post(f"{base_url}{endpoint}", json=case["payload"], timeout=15)
            print(f"Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"Prediction: {data['prediction']}")
                print(f"Probabilities: {data['probabilities']}")
                passed += 1
            else:
                print(f"Error response: {r.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        print("-" * 60)

    print(f"\n{passed}/{len(cases)} test cases returned HTTP 200 with a prediction ({combo_name}).")
    return passed


def run_tests(base_url: str, combos: list[str]):
    print(f"Testing API at: {base_url}\n")

    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check: {r.status_code} {r.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Could not reach API at {base_url}: {e}")
        sys.exit(1)

    total_passed = 0
    total_cases = 0
    for combo_name in combos:
        total_passed += run_combo(base_url, combo_name)
        variant, _, _ = COMBOS[combo_name]
        total_cases += len(TEST_CASES_BY_VARIANT[variant])

    print(f"\nTOTAL: {total_passed}/{total_cases} test cases returned HTTP 200 with a prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Bank Marketing Prediction API")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the API (default: local server on port 8000)",
    )
    parser.add_argument(
        "--model",
        choices=list(COMBOS) + ["all"],
        default="all",
        help="Which of the 4 deployed model/dataset combinations to test (default: all)",
    )
    args = parser.parse_args()
    combos = list(COMBOS) if args.model == "all" else [args.model]
    run_tests(args.url.rstrip("/"), combos)
