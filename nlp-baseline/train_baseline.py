"""
train_baseline.py
==================

Reusable baseline NLP pipeline: text cleaning -> TF-IDF -> Logistic Regression.

Designed to be run once, and re-run unchanged, against different corpora that
share one of two shapes (conversation-based or flat-text) and one of three
file formats (JSONL, JSON, CSV). All corpus-specific behavior (which fields to
read, how to clean text, which labels map to which class, and all model
hyper-parameters) lives in an external JSON config file so this script never
needs to be edited when a new corpus arrives -- only the config does.

Usage
-----
    python train_baseline.py --input data/corpus.jsonl --config config_messages.json --output results
    python train_baseline.py --input data/corpus.csv   --config config_csv.json      --output results

See README.md for a full walkthrough.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_baseline")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class BaselineError(RuntimeError):
    """Raised for any recoverable, user-facing pipeline error."""


# ---------------------------------------------------------------------------
# Corpus loading (JSONL / JSON / CSV -> list of dict records)
# ---------------------------------------------------------------------------

def load_corpus(input_path: Path) -> List[Dict[str, Any]]:
    """Load a corpus file and return a flat list of raw record dicts.

    Supports three formats, detected from the file extension:
      * .jsonl -- one JSON object per line.
      * .json  -- either a JSON array of objects, or a JSON object that
                  wraps the array under a common key ("data", "records",
                  "conversations", "items"), or a single record object.
      * .csv   -- a standard CSV file with a header row.
    """
    if not input_path.exists():
        raise BaselineError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(input_path)
    if suffix == ".json":
        return _load_json(input_path)
    if suffix == ".csv":
        return _load_csv(input_path)
    raise BaselineError(
        f"Unsupported input file extension '{suffix}'. Expected .jsonl, .json, or .csv."
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BaselineError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    if not records:
        raise BaselineError(f"No records found in {path}")
    return records


def _load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "conversations", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # Fall back to treating the whole object as a single record.
        return [payload]
    raise BaselineError(
        f"Unsupported JSON structure in {path}: expected a list or object at the top level."
    )


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # pandas raises several exception types
        raise BaselineError(f"Failed to read CSV file {path}: {exc}") from exc
    if df.empty:
        raise BaselineError(f"No records found in {path}")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and minimally validate the JSON config file."""
    if not config_path.exists():
        raise BaselineError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"Invalid JSON in config file {config_path}: {exc}") from exc

    text_mode = config.get("text_mode")
    if text_mode not in ("field", "messages"):
        raise BaselineError(
            "Config 'text_mode' must be either 'field' or 'messages' "
            f"(got: {text_mode!r})."
        )
    if text_mode == "field" and not config.get("text_field"):
        raise BaselineError("Config 'text_mode' is 'field' but 'text_field' is missing.")
    if text_mode == "messages" and not config.get("messages_field"):
        raise BaselineError("Config 'text_mode' is 'messages' but 'messages_field' is missing.")
    if not config.get("label_field"):
        raise BaselineError("Config is missing required key 'label_field'.")

    return config


# ---------------------------------------------------------------------------
# Nested field access (dot notation)
# ---------------------------------------------------------------------------

def get_nested_value(record: Mapping[str, Any], dotted_path: Optional[str]) -> Any:
    """Fetch a possibly-nested value from a record using dot notation.

    Example: get_nested_value(record, "customer.contact.email") walks
    record["customer"]["contact"]["email"], returning None if any hop
    along the way is missing or is not a dict.
    """
    if not dotted_path:
        return None
    current: Any = record
    for part in dotted_path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(record: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Extract the raw (uncleaned) text for a single record.

    Two modes, selected by config["text_mode"]:
      * "field"    -- a single free-text field (flat corpus).
      * "messages" -- a list of {role, text} messages that get filtered by
                       role and joined with a separator (conversation corpus).
    """
    text_mode = config.get("text_mode")

    if text_mode == "field":
        value = get_nested_value(record, config.get("text_field"))
        return str(value) if value is not None else ""

    if text_mode == "messages":
        messages = get_nested_value(record, config.get("messages_field"))
        if not isinstance(messages, list):
            return ""

        role_field = config.get("role_field", "role")
        message_text_field = config.get("message_text_field", "text")
        included_roles = config.get("included_roles")
        separator = config.get("message_separator", " [SEP] ")

        pieces: List[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = get_nested_value(message, role_field)
            # If included_roles is empty/omitted, include every role.
            if included_roles and role not in included_roles:
                continue
            text = get_nested_value(message, message_text_field)
            if text:
                pieces.append(str(text))
        return separator.join(pieces)

    raise BaselineError(f"Unknown text_mode: {text_mode!r}")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Sequences of 7+ digits, optionally separated by spaces/dashes/dots/parens --
# matches phone numbers (Israeli and international) without swallowing short
# standalone numbers such as prices or quantities.
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\-.\s()]{5,}\d)(?!\w)")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Keep: word characters (Unicode-aware, so Hebrew letters are preserved),
# whitespace, basic punctuation, and the angle brackets used by placeholder
# tokens such as <URL>/<EMAIL>/<PHONE>/<NUM>.
_NON_TEXT_RE = re.compile(r"[^\w\s.,!?;:'\"\-()<>\[\]]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, cleaning_config: Mapping[str, Any]) -> str:
    """Clean a single piece of text according to the config's cleaning rules.

    Hebrew (and any other non-Latin script) text is preserved throughout:
    `str.lower()` in Python only affects cased (Latin-style) characters, and
    the "keep" regex used for symbol removal is Unicode-aware so Hebrew
    letters are never stripped.
    """
    if not text:
        return ""

    cleaned = text

    if cleaning_config.get("replace_phones", True):
        cleaned = _PHONE_RE.sub(" <PHONE> ", cleaned)
    if cleaning_config.get("replace_urls", True):
        cleaned = _URL_RE.sub(" <URL> ", cleaned)
    if cleaning_config.get("replace_emails", True):
        cleaned = _EMAIL_RE.sub(" <EMAIL> ", cleaned)
    if cleaning_config.get("replace_numbers", False):
        cleaned = _NUMBER_RE.sub(" <NUM> ", cleaned)
    if cleaning_config.get("remove_non_text_symbols", True):
        cleaned = _NON_TEXT_RE.sub(" ", cleaned)
    if cleaning_config.get("lowercase_english", True):
        cleaned = cleaned.lower()

    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def invert_label_mapping(label_mapping: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    """Turn {mapped_label: [raw_label, ...]} into {raw_label: mapped_label}."""
    inverted: Dict[str, str] = {}
    for mapped_label, raw_labels in label_mapping.items():
        for raw_label in raw_labels:
            if raw_label in inverted:
                raise BaselineError(
                    f"Raw label '{raw_label}' appears under more than one mapped "
                    f"label in label_mapping ('{inverted[raw_label]}' and '{mapped_label}')."
                )
            inverted[raw_label] = mapped_label
    return inverted


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------

def build_dataset(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> pd.DataFrame:
    """Turn raw records into a DataFrame with columns:
    record_id, raw_text, cleaned_text, raw_label, label.

    Records whose raw label is not present in label_mapping (when a mapping
    is configured) are dropped. Classes left with fewer than
    minimum_examples_per_class examples are also dropped, with a warning.
    """
    id_field = config.get("id_field")
    label_field = config["label_field"]
    cleaning_config = config.get("cleaning", {})
    label_mapping = config.get("label_mapping")
    raw_to_mapped = invert_label_mapping(label_mapping) if label_mapping else None

    rows: List[Dict[str, Any]] = []
    skipped_unmapped = 0
    skipped_empty_text = 0
    skipped_missing_label = 0

    for index, record in enumerate(records):
        raw_label = get_nested_value(record, label_field)
        if raw_label is None:
            skipped_missing_label += 1
            continue
        raw_label = str(raw_label)

        if raw_to_mapped is not None:
            if raw_label not in raw_to_mapped:
                skipped_unmapped += 1
                continue
            mapped_label = raw_to_mapped[raw_label]
        else:
            mapped_label = raw_label

        raw_text = extract_text(record, config)
        cleaned = clean_text(raw_text, cleaning_config)
        if not cleaned:
            skipped_empty_text += 1
            continue

        record_id = get_nested_value(record, id_field) if id_field else None
        if record_id is None:
            record_id = f"row_{index}"

        rows.append(
            {
                "record_id": record_id,
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "raw_label": raw_label,
                "label": mapped_label,
            }
        )

    if skipped_unmapped:
        logger.info("Ignored %d record(s) with a label not present in label_mapping.", skipped_unmapped)
    if skipped_missing_label:
        logger.info("Ignored %d record(s) with a missing label field.", skipped_missing_label)
    if skipped_empty_text:
        logger.info("Ignored %d record(s) with empty text after extraction/cleaning.", skipped_empty_text)

    df = pd.DataFrame(rows, columns=["record_id", "raw_text", "cleaned_text", "raw_label", "label"])
    if df.empty:
        raise BaselineError(
            "No usable records remain after extraction and label filtering. "
            "Check text_mode/label_field/label_mapping in the config against the actual corpus."
        )

    min_per_class = int(config.get("minimum_examples_per_class", 2))
    counts = df["label"].value_counts()
    small_classes = counts[counts < min_per_class].index.tolist()
    if small_classes:
        logger.warning(
            "Dropping class(es) with fewer than minimum_examples_per_class=%d example(s): %s",
            min_per_class,
            small_classes,
        )
        df = df[~df["label"].isin(small_classes)].reset_index(drop=True)

    if df["label"].nunique() < 2:
        raise BaselineError(
            "Fewer than 2 classes remain after filtering. Need at least 2 classes to train a classifier. "
            f"Remaining class counts: {df['label'].value_counts().to_dict()}"
        )

    return df


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_pipeline(tfidf_config: Mapping[str, Any], model_config: Mapping[str, Any]) -> Pipeline:
    """Build an (unfit) TF-IDF + Logistic Regression sklearn Pipeline."""
    ngram_range = tfidf_config.get("ngram_range", [1, 1])
    vectorizer = TfidfVectorizer(
        analyzer=tfidf_config.get("analyzer", "word"),
        ngram_range=tuple(ngram_range),
        min_df=tfidf_config.get("min_df", 1),
        max_df=tfidf_config.get("max_df", 1.0),
        max_features=tfidf_config.get("max_features"),
        sublinear_tf=tfidf_config.get("sublinear_tf", False),
    )
    classifier = LogisticRegression(
        max_iter=model_config.get("max_iter", 1000),
        class_weight=model_config.get("class_weight"),
        C=model_config.get("C", 1.0),
        solver=model_config.get("solver", "lbfgs"),
        random_state=model_config.get("random_state", 42),
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Evaluation + saving
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    """Compute the metrics required by the Definition of Done.

    Both binary and multiclass labels are supported: macro/weighted
    averaging is used throughout since it is well-defined for either case
    (plain "precision"/"recall"/"f1" below use macro averaging as the
    baseline default; "*_weighted" variants are also provided).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_outputs(
    output_dir: Path,
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    y_test: Sequence[str],
    y_pred: Sequence[str],
    class_labels: Sequence[str],
    test_df: pd.DataFrame,
    probabilities: Optional[Any],
) -> None:
    """Write every artifact required by the Definition of Done to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trained pipeline.
    joblib.dump(pipeline, output_dir / "tfidf_logistic_regression.joblib")

    # 2. Metrics.
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    # 3. Classification report (text).
    report_text = classification_report(y_test, y_pred, labels=class_labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    # 4. Confusion matrix (CSV with labeled rows/columns).
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in class_labels], columns=[f"pred_{c}" for c in class_labels])
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")

    # 5. Per-record test predictions, including per-class probabilities.
    predictions_df = test_df[["record_id", "cleaned_text", "label"]].copy()
    predictions_df = predictions_df.rename(columns={"label": "actual_label"})
    predictions_df["predicted_label"] = list(y_pred)
    if probabilities is not None:
        for i, class_label in enumerate(class_labels):
            predictions_df[f"probability_{class_label}"] = probabilities[:, i]
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    logger.info("Saved all outputs to %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression baseline classifier."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the corpus file (.jsonl, .json, or .csv).")
    parser.add_argument("--config", required=True, type=Path, help="Path to the JSON config file.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write results into.")
    parser.add_argument(
        "--split-ids",
        type=Path,
        default=None,
        help=(
            "Optional path to a split_ids.json produced by the AlephBERT training pipeline. "
            "When provided, the baseline uses the same train_ids and test_ids for a "
            "scientifically fair comparison. val_ids are excluded from training so both "
            "experiments see exactly the same training set. "
            "If omitted, an independent stratified random split is used (original behavior)."
        ),
    )
    return parser.parse_args(argv)


def _load_split_ids_from_file(split_ids_path: Path) -> tuple:
    """Load train_ids and test_ids from a split_ids.json file.

    val_ids are excluded from training so the baseline trains on exactly the
    same conversations as AlephBERT. This ensures a fair comparison.
    """
    if not split_ids_path.exists():
        raise BaselineError(f"split-ids file not found: {split_ids_path}")
    try:
        with split_ids_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"Invalid JSON in split-ids file: {exc}") from exc
    train_ids = set(payload.get("train_ids", []))
    test_ids = set(payload.get("test_ids", []))
    val_ids = set(payload.get("val_ids", []))
    if not train_ids or not test_ids:
        raise BaselineError("split-ids file must contain non-empty 'train_ids' and 'test_ids'.")
    logger.info(
        "Loaded split IDs: train=%d, val=%d (excluded), test=%d",
        len(train_ids), len(val_ids), len(test_ids),
    )
    return train_ids, test_ids


def run(input_path: Path, config_path: Path, output_dir: Path, split_ids_path: Optional[Path] = None) -> None:
    config = load_config(config_path)
    records = load_corpus(input_path)
    logger.info("Loaded %d raw record(s) from %s", len(records), input_path)

    df = build_dataset(records, config)
    logger.info("Built dataset with %d usable record(s) across %d class(es).", len(df), df["label"].nunique())
    logger.info("Class distribution: %s", df["label"].value_counts().to_dict())

    if split_ids_path is not None:
        # Use shared split IDs for fair comparison with AlephBERT
        train_ids, test_ids = _load_split_ids_from_file(split_ids_path)
        id_field = config.get("id_field", "record_id")
        train_df = df[df[id_field].astype(str).isin(train_ids)].reset_index(drop=True)
        test_df = df[df[id_field].astype(str).isin(test_ids)].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise BaselineError(
                f"After applying split IDs, train={len(train_df)} test={len(test_df)}. "
                "Check that id_field in config matches the conversation_id field in the corpus."
            )
        logger.info(
            "Using shared split: train=%d, test=%d (val excluded from training)",
            len(train_df), len(test_df),
        )
    else:
        # Original behavior: independent stratified random split
        test_size = float(config.get("test_size", 0.2))
        random_state = int(config.get("random_state", 42))
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df["label"],
            )
        except ValueError as exc:
            raise BaselineError(
                "Stratified train/test split failed -- this usually means one or more "
                "classes have too few examples for the chosen test_size. Add more data, "
                "raise minimum_examples_per_class filtering, or adjust test_size. "
                f"Original error: {exc}"
            ) from exc

    pipeline = build_pipeline(config.get("tfidf", {}), config.get("model", {}))
    pipeline.fit(train_df["cleaned_text"], train_df["label"])

    y_pred = pipeline.predict(test_df["cleaned_text"])
    class_labels = sorted(df["label"].unique().tolist())

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            raw_proba = pipeline.predict_proba(test_df["cleaned_text"])
            # Reorder columns to match `class_labels` (pipeline.classes_ order
            # may differ, e.g. alphabetical vs. insertion order).
            proba_classes = list(pipeline.classes_)
            column_order = [proba_classes.index(c) for c in class_labels]
            probabilities = raw_proba[:, column_order]
        except Exception:  # pragma: no cover - defensive; not all solvers support proba
            probabilities = None

    metrics = evaluate_predictions(test_df["label"], y_pred)
    metrics["n_train"] = len(train_df)
    metrics["n_test"] = len(test_df)
    metrics["classes"] = class_labels
    logger.info("Test accuracy: %.4f | macro F1: %.4f | weighted F1: %.4f", metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])

    save_outputs(
        output_dir=output_dir,
        pipeline=pipeline,
        metrics=metrics,
        y_test=test_df["label"],
        y_pred=y_pred,
        class_labels=class_labels,
        test_df=test_df,
        probabilities=probabilities,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.input, args.config, args.output, split_ids_path=args.split_ids)
    except BaselineError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure surface
        logger.error("Unexpected error: %s", exc)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
