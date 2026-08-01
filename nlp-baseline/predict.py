"""
predict.py
==========

Load a trained baseline pipeline (produced by train_baseline.py) and run it
on a single piece of ad-hoc text.

Usage
-----
    python predict.py --model results/tfidf_logistic_regression.joblib --text "כמה זה עולה?"

Prints the predicted class followed by the model's probability for every
class it was trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import joblib

# Reuse the exact same text-cleaning logic used at training time, so
# predictions are made on text that looks like what the model was trained on.
# Default cleaning matches the settings shipped in config_messages.json /
# config_csv.json; edit DEFAULT_CLEANING_CONFIG below if your final config
# changes these values.
from train_baseline import clean_text

DEFAULT_CLEANING_CONFIG = {
    "replace_urls": True,
    "replace_emails": True,
    "replace_phones": True,
    "replace_numbers": False,
    "lowercase_english": True,
    "remove_non_text_symbols": True,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained TF-IDF + Logistic Regression baseline on a single text."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to the saved .joblib pipeline.")
    parser.add_argument("--text", required=True, type=str, help="Raw text to classify.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.model.exists():
        print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
        return 1

    try:
        pipeline = joblib.load(args.model)
    except Exception as exc:
        print(f"[ERROR] Failed to load model from {args.model}: {exc}", file=sys.stderr)
        return 1

    cleaned = clean_text(args.text, DEFAULT_CLEANING_CONFIG)
    if not cleaned:
        print("[ERROR] Input text is empty after cleaning; nothing to predict.", file=sys.stderr)
        return 1

    prediction = pipeline.predict([cleaned])[0]
    print(f"Predicted class: {prediction}")

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([cleaned])[0]
        print("Class probabilities:")
        for class_label, probability in sorted(
            zip(pipeline.classes_, probabilities), key=lambda pair: pair[1], reverse=True
        ):
            print(f"  {class_label}: {probability:.4f}")
    else:
        print("Class probabilities: not available for this model/solver.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
