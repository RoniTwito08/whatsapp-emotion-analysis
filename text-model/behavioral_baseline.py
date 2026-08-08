"""behavioral_baseline.py

Model C — Behavioral Features Only classifier.

Trains Logistic Regression (primary) and Random Forest (secondary) on
conversation-level behavioral features extracted from the messages array.
Uses NO text content whatsoever — only message structure, counts, lengths,
timing, and boolean flags derived from Hebrew regex patterns.

The exact same train/val/test split from AlephBERT is used for a fair
comparison. Split IDs are loaded from text-model/outputs/split_ids.json.

Usage:
    python behavioral_baseline.py --config config.json
    python behavioral_baseline.py --config config.json --experiment-name behavioral_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from behavioral_features_design import (
    FORBIDDEN_FIELDS,
    extract_behavioral_features,
    feature_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("behavioral_baseline")


ALEPHBERT_RESULTS = {
    "accuracy": 0.9695,
    "macro_f1": 0.9695,
    "weighted_f1": 0.9695,
    "losing_interest_precision": 0.9821,
    "losing_interest_recall": 0.9563,
    "losing_interest_f1": 0.9690,
    "interested_precision": 0.9569,
    "interested_recall": 0.9826,
    "interested_f1": 0.9696,
    "note": "best checkpoint epoch 4 / 5",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Model C: behavioral features only classifier."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json")
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _invert_label_mapping(label_mapping: Dict[str, List[str]]) -> Dict[str, str]:
    inverted: Dict[str, str] = {}
    for mapped, raws in label_mapping.items():
        for raw in raws:
            inverted[raw] = mapped
    return inverted


def load_data(
    config_path: Path,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load raw corpus records and return (records, id_to_binary_label)."""
    data_cfg = config["data"]
    input_path = Path(data_cfg["input_path"])
    if not input_path.is_absolute():
        input_path = config_path.resolve().parent / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if isinstance(records, dict):
        for key in ("data", "records", "conversations", "items"):
            if key in records:
                records = records[key]
                break

    label_mapping = data_cfg["label_mapping"]
    raw_to_binary = _invert_label_mapping(label_mapping)
    id_field = data_cfg.get("id_field", "conversation_id")
    label_field = data_cfg.get("label_field", "final_outcome")

    id_to_label: Dict[str, str] = {}
    for rec in records:
        cid = str(rec.get(id_field, ""))
        raw = str(rec.get(label_field, ""))
        if cid and raw in raw_to_binary:
            id_to_label[cid] = raw_to_binary[raw]

    logger.info("Loaded %d records; %d with valid labels.", len(records), len(id_to_label))
    return records, id_to_label


def load_split_ids(split_ids_path: Path) -> Tuple[List[str], List[str], List[str]]:
    if not split_ids_path.exists():
        raise FileNotFoundError(f"split_ids.json not found: {split_ids_path}")
    with split_ids_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["train_ids"], payload["val_ids"], payload["test_ids"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    records: List[Dict[str, Any]],
    id_to_label: Dict[str, str],
    split_ids: List[str],
    config: Dict[str, Any],
    label_to_id: Dict[str, int],
    feat_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features for all conversations in split_ids.

    Args:
        feat_names: Optional explicit feature list. Defaults to all features
            from feature_names(). Pass pure_behavioral_feature_names() for the
            structural-only experiment.

    Returns:
        X: float array [n_conversations, n_features]
        y: int array  [n_conversations]
        kept_ids: list of conversation IDs in the order they appear in X/y
    """
    id_field = config["data"].get("id_field", "conversation_id")
    split_id_set = set(split_ids)
    id_to_record: Dict[str, Dict[str, Any]] = {
        str(r.get(id_field, "")): r for r in records
    }

    if feat_names is None:
        feat_names = feature_names()
    rows: List[List[float]] = []
    labels: List[int] = []
    kept_ids: List[str] = []
    missing = 0

    for cid in split_ids:
        if cid not in id_to_label:
            missing += 1
            continue
        rec = id_to_record.get(cid)
        if rec is None:
            missing += 1
            continue
        feats = extract_behavioral_features(rec)
        if not feats:
            missing += 1
            continue
        rows.append([feats.get(fn, 0.0) for fn in feat_names])
        labels.append(label_to_id[id_to_label[cid]])
        kept_ids.append(cid)

    if missing:
        logger.warning("Skipped %d records (missing label/features).", missing)
    if not rows:
        raise ValueError(f"No valid features extracted for split with {len(split_ids)} IDs.")

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)
    logger.info("Built feature matrix: shape=%s, labels=%s", X.shape, dict(zip(*np.unique(y, return_counts=True))))
    return X, y, kept_ids


# ---------------------------------------------------------------------------
# Model building and evaluation
# ---------------------------------------------------------------------------

def build_lr_pipeline(lr_cfg: Dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(lr_cfg.get("C", 1.0)),
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            solver=lr_cfg.get("solver", "lbfgs"),
            class_weight=lr_cfg.get("class_weight"),
            random_state=int(lr_cfg.get("random_state", 42)),
        )),
    ])


def build_rf_pipeline(rf_cfg: Dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("classifier", RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
            class_weight=rf_cfg.get("class_weight"),
            random_state=int(rf_cfg.get("random_state", 42)),
            n_jobs=rf_cfg.get("n_jobs", -1),
        )),
    ])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class_id: int,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_class_id, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_class_id, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_class_id, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    positive_class_id: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X) if hasattr(pipeline, "predict_proba") else None
    metrics = compute_metrics(y, y_pred, positive_class_id)
    return metrics, y_pred, proba


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------

def save_feature_analysis(
    output_dir: Path,
    X_train: np.ndarray,
    feat_names: List[str],
    pipeline: Pipeline,
    model_name: str,
    label_to_id: Dict[str, int],
) -> None:
    # Feature statistics
    stats: Dict[str, Any] = {}
    for i, name in enumerate(feat_names):
        col = X_train[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p25": float(np.percentile(col, 25)),
            "p75": float(np.percentile(col, 75)),
        }
    with (output_dir / "feature_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    # Feature importance / coefficients
    id_to_label = {v: k for k, v in label_to_id.items()}
    classifier = pipeline.named_steps["classifier"]

    if model_name == "logistic_regression" and hasattr(classifier, "coef_"):
        coef = classifier.coef_[0]
        rows = sorted(
            [{"feature": n, "coefficient": float(c), "abs_coefficient": abs(float(c))}
             for n, c in zip(feat_names, coef)],
            key=lambda r: r["abs_coefficient"],
            reverse=True,
        )
        with (output_dir / "lr_coefficients.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["feature", "coefficient", "abs_coefficient"])
            writer.writeheader()
            writer.writerows(rows)

    if model_name == "random_forest" and hasattr(classifier, "feature_importances_"):
        imps = classifier.feature_importances_
        rows = sorted(
            [{"feature": n, "importance": float(v)} for n, v in zip(feat_names, imps)],
            key=lambda r: r["importance"],
            reverse=True,
        )
        with (output_dir / "rf_feature_importances.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["feature", "importance"])
            writer.writeheader()
            writer.writerows(rows)


def save_predictions(
    output_dir: Path,
    kept_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: Optional[np.ndarray],
    label_to_id: Dict[str, int],
) -> None:
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]
    fieldnames = (
        ["conversation_id", "actual_label", "predicted_label"]
        + [f"probability_{c}" for c in class_labels]
    )
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, cid in enumerate(kept_ids):
            row: Dict[str, Any] = {
                "conversation_id": cid,
                "actual_label": id_to_label[int(y_true[i])],
                "predicted_label": id_to_label[int(y_pred[i])],
            }
            if proba is not None:
                for j, c in enumerate(class_labels):
                    row[f"probability_{c}"] = f"{proba[i, j]:.6f}"
            writer.writerow(row)


def save_test_results(
    output_dir: Path,
    test_metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_to_id: Dict[str, int],
) -> None:
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, ensure_ascii=False, indent=2)

    report = classification_report(
        y_true, y_pred, target_names=class_labels, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    cm_rows = [[""] + [f"pred_{c}" for c in class_labels]]
    for i, c in enumerate(class_labels):
        cm_rows.append([f"actual_{c}"] + [str(v) for v in cm[i]])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(cm_rows)


def print_comparison_table(
    behavioral_test: Dict[str, float],
    model_name: str,
    label_to_id: Dict[str, int],
) -> None:
    id_to_label = {v: k for k, v in label_to_id.items()}
    pos_class = "losing_interest"
    pos_id = label_to_id.get(pos_class, 1)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 70)
    print(f"{'Metric':<30} {'AlephBERT':>12} {'Behavioral (' + model_name + ')':>20}")
    print("-" * 70)

    metrics = [
        ("accuracy", "accuracy", "accuracy"),
        ("macro_f1", "macro_f1", "macro_f1"),
        ("weighted_f1", "weighted_f1", "weighted_f1"),
        (f"{pos_class} precision", "losing_interest_precision", "precision"),
        (f"{pos_class} recall", "losing_interest_recall", "recall"),
        (f"{pos_class} F1", "losing_interest_f1", "f1"),
    ]
    for label, ab_key, beh_key in metrics:
        ab_val = ALEPHBERT_RESULTS.get(ab_key, float("nan"))
        beh_val = behavioral_test.get(beh_key, float("nan"))
        print(f"  {label:<28} {ab_val:>12.4f} {beh_val:>20.4f}")

    print("-" * 70)
    print()
    print("NOTE: Both models use the same held-out test set (seed=42, 459 conversations).")
    print("AlephBERT: text only (customer messages, max_length=512).")
    print(f"Behavioral: {feature_names().__len__()} structural features only, no message text.")
    print()
    print("CAUTION: All conversations are synthetic. Performance may not")
    print("generalise to real WhatsApp data. Behavioral features are derived")
    print("from Hebrew regex patterns that may produce cleaner signals on")
    print("synthetic text than on organic customer conversations.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: Path, experiment_name_override: Optional[str] = None) -> None:
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    beh_cfg = config.get("behavioral_model", {})
    experiment_name = experiment_name_override or beh_cfg.get("experiment_name", "behavioral_baseline_v1")
    positive_class = beh_cfg.get("positive_class", "losing_interest")

    base_dir = config_path.resolve().parent
    output_dir = base_dir / "outputs" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Label mapping
    label_mapping = config["data"]["label_mapping"]
    label_to_id = {label: i for i, label in enumerate(sorted(label_mapping.keys()))}
    id_to_label = {v: k for k, v in label_to_id.items()}
    positive_class_id = label_to_id[positive_class]

    print("=" * 64)
    print("Behavioral Features Baseline (Model C)")
    print("=" * 64)
    print(f"  Experiment  : {experiment_name}")
    print(f"  Features    : {len(feature_names())} behavioral features (no text)")
    print(f"  Labels      : {label_to_id}")
    print(f"  Positive    : {positive_class} (id={positive_class_id})")
    print(f"  Output dir  : {output_dir}")
    print("=" * 64)

    # Load data and split IDs
    records, id_to_label_map = load_data(config_path, config)

    split_ids_path_str = config.get("split", {}).get("split_ids_output", "outputs/split_ids.json")
    split_ids_path = base_dir / split_ids_path_str
    train_ids, val_ids, test_ids = load_split_ids(split_ids_path)

    # Validate split IDs
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set), "train/val overlap in split_ids.json!"
    assert not (train_set & test_set), "train/test overlap in split_ids.json!"
    assert not (val_set & test_set), "val/test overlap in split_ids.json!"

    print(f"\nSplit IDs loaded from: {split_ids_path}")
    print(f"  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print()

    # Extract features
    feat_names = feature_names()
    logger.info("Extracting features for all splits...")

    X_train, y_train, train_kept = build_feature_matrix(records, id_to_label_map, train_ids, config, label_to_id)
    X_val, y_val, val_kept = build_feature_matrix(records, id_to_label_map, val_ids, config, label_to_id)
    X_test, y_test, test_kept = build_feature_matrix(records, id_to_label_map, test_ids, config, label_to_id)

    print(f"Feature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Save feature names
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
        json.dump(feat_names, fh, ensure_ascii=False, indent=2)

    # ---- Train models ----
    lr_cfg = beh_cfg.get("logistic_regression", {})
    rf_cfg = beh_cfg.get("random_forest", {})

    candidates: Dict[str, Pipeline] = {
        "logistic_regression": build_lr_pipeline(lr_cfg),
        "random_forest": build_rf_pipeline(rf_cfg),
    }

    val_results: Dict[str, Dict[str, float]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_name, pipeline in candidates.items():
        logger.info("Training %s...", model_name)
        pipeline.fit(X_train, y_train)
        val_metrics, _, _ = evaluate_pipeline(pipeline, X_val, y_val, positive_class_id)
        val_results[model_name] = val_metrics
        trained_pipelines[model_name] = pipeline
        print(f"  [{model_name}] val: acc={val_metrics['accuracy']:.4f} | "
              f"macro_f1={val_metrics['macro_f1']:.4f} | "
              f"f1={val_metrics['f1']:.4f} (positive={positive_class})")

    # ---- Model selection by val macro F1 ----
    best_name = max(val_results, key=lambda k: val_results[k]["macro_f1"])
    best_pipeline = trained_pipelines[best_name]
    print(f"\nBest model: {best_name} (val macro_f1={val_results[best_name]['macro_f1']:.4f})")

    # Save validation results
    with (output_dir / "val_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(val_results, fh, ensure_ascii=False, indent=2)

    # ---- Save trained model ----
    joblib.dump(best_pipeline, output_dir / "model.joblib")

    # ---- Feature analysis on training data ----
    save_feature_analysis(output_dir, X_train, feat_names, best_pipeline, best_name, label_to_id)

    # ---- Save split statistics ----
    with (output_dir / "split_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
            "source": str(split_ids_path),
        }, fh, indent=2)

    # ---- Save model config ----
    model_config = {
        "best_model": best_name,
        "feature_count": len(feat_names),
        "label_to_id": label_to_id,
        "positive_class": positive_class,
        "logistic_regression": lr_cfg,
        "random_forest": rf_cfg,
        "validation_results": val_results,
    }
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as fh:
        json.dump(model_config, fh, ensure_ascii=False, indent=2)

    # ---- Final test evaluation (once) ----
    logger.info("Evaluating best model on held-out test set...")
    test_metrics, test_preds, test_proba = evaluate_pipeline(
        best_pipeline, X_test, y_test, positive_class_id
    )
    test_metrics["best_model"] = best_name
    test_metrics["experiment_name"] = experiment_name

    print("\n" + "=" * 64)
    print(f"TEST SET RESULTS ({best_name})")
    print("=" * 64)
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")

    id_to_lbl = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_lbl[i] for i in range(len(label_to_id))]
    print()
    print(classification_report(y_test, test_preds, target_names=class_labels, zero_division=0))

    save_test_results(output_dir, test_metrics, y_test, test_preds, label_to_id)
    save_predictions(output_dir, test_kept, y_test, test_preds, test_proba, label_to_id)

    print(f"\nAll artifacts saved to: {output_dir.resolve()}")

    # ---- Comparison table ----
    print_comparison_table(test_metrics, best_name, label_to_id)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.config, experiment_name_override=args.experiment_name)
    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
