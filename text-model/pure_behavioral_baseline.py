"""pure_behavioral_baseline.py

Model C1 — Pure Structural / Timing Features Only.

Uses ZERO text-derived (lexical) signals. All features are observable from
message structure: counts, lengths, response delays, session breaks, and
who sends the last message. No regex, no character-pattern scanning.

Shares all infrastructure with behavioral_baseline.py; only the feature
set differs.

Usage:
    python pure_behavioral_baseline.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from behavioral_baseline import (
    ALEPHBERT_RESULTS,
    build_lr_pipeline,
    build_rf_pipeline,
    evaluate_pipeline,
    load_data,
    load_split_ids,
    save_feature_analysis,
    save_predictions,
    save_test_results,
)
from behavioral_features_design import (
    FORBIDDEN_FIELDS,
    LEXICAL_FEATURE_NAMES,
    pure_behavioral_feature_names,
)

# Import build_feature_matrix with explicit feat_names support
from behavioral_baseline import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("pure_behavioral")

BEHAVIORAL_LEXICAL_RESULTS = {
    "model": "Random Forest (43 features)",
    "accuracy": 0.8671,
    "macro_f1": 0.8669,
    "weighted_f1": 0.8669,
    "losing_interest_precision": 0.8962,
    "losing_interest_recall": 0.8297,
    "losing_interest_f1": 0.8617,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model C1: pure structural/timing features.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args(argv)


def print_three_way_comparison(
    pure_test: Dict[str, float],
    pure_model_name: str,
    label_to_id: Dict[str, int],
) -> None:
    pos = "losing_interest"
    print("\n" + "=" * 78)
    print("THREE-WAY MODEL COMPARISON — TEST SET (459 conversations, seed=42)")
    print("=" * 78)
    hdr = f"{'Metric':<32} {'AlephBERT':>12} {'Behavioral+Lex':>16} {'Pure Structural':>16}"
    print(hdr)
    print("-" * 78)

    rows = [
        ("accuracy",           "accuracy",                "accuracy",         "accuracy"),
        ("macro F1",           "macro_f1",                "macro_f1",         "macro_f1"),
        ("weighted F1",        "weighted_f1",             "weighted_f1",      "weighted_f1"),
        (f"{pos} precision",   "losing_interest_precision","losing_interest_precision","precision"),
        (f"{pos} recall",      "losing_interest_recall",  "losing_interest_recall",   "recall"),
        (f"{pos} F1",          "losing_interest_f1",      "losing_interest_f1",       "f1"),
    ]
    for label, ab_key, bl_key, p_key in rows:
        ab  = ALEPHBERT_RESULTS.get(ab_key, float("nan"))
        bl  = BEHAVIORAL_LEXICAL_RESULTS.get(bl_key, float("nan"))
        pur = pure_test.get(p_key, float("nan"))
        print(f"  {label:<30} {ab:>12.4f} {bl:>16.4f} {pur:>16.4f}")

    print("-" * 78)
    beh_macro = BEHAVIORAL_LEXICAL_RESULTS["macro_f1"]
    pur_macro  = pure_test.get("macro_f1", float("nan"))
    lex_gain   = beh_macro - pur_macro
    text_gain  = ALEPHBERT_RESULTS["macro_f1"] - beh_macro

    print(f"\n  Lexical signals add  : {lex_gain:+.4f} macro F1 (Behavioral+Lex − Pure Structural)")
    print(f"  Text understanding adds: {text_gain:+.4f} macro F1 (AlephBERT − Behavioral+Lex)")
    print()
    print(f"  Pure structural model  : {len(pure_behavioral_feature_names())} features, no text")
    print(f"  Behavioral+Lex model   : 43 features (includes regex signals)")
    print(f"  AlephBERT              : full text, max_length=512")
    print()
    print("  CAUTION: All data is synthetic. Patterns may be more stereotyped")
    print("  than in real WhatsApp conversations.")
    print("=" * 78)


def run(config_path: Path, experiment_name_override: Optional[str] = None) -> None:
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    experiment_name = experiment_name_override or "pure_behavioral_baseline_v1"
    positive_class = config.get("behavioral_model", {}).get("positive_class", "losing_interest")
    base_dir = config_path.resolve().parent
    output_dir = base_dir / "outputs" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_names = pure_behavioral_feature_names()
    label_mapping = config["data"]["label_mapping"]
    label_to_id = {lbl: i for i, lbl in enumerate(sorted(label_mapping.keys()))}
    positive_class_id = label_to_id[positive_class]

    print("=" * 64)
    print("Pure Structural Baseline (Model C1)")
    print("=" * 64)
    print(f"  Experiment  : {experiment_name}")
    print(f"  Features    : {len(feat_names)} pure structural (no text/regex)")
    print(f"  Labels      : {label_to_id}")
    print(f"  Positive    : {positive_class}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 64)

    records, id_to_label_map = load_data(config_path, config)

    split_ids_path = base_dir / config.get("split", {}).get("split_ids_output", "outputs/split_ids.json")
    train_ids, val_ids, test_ids = load_split_ids(split_ids_path)

    assert not (set(train_ids) & set(val_ids)),  "train/val overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap"
    assert not (set(val_ids) & set(test_ids)),   "val/test overlap"
    print(f"\nSplit IDs: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}\n")

    logger.info("Extracting pure structural features...")
    X_train, y_train, _ = build_feature_matrix(records, id_to_label_map, train_ids, config, label_to_id, feat_names)
    X_val,   y_val,   _ = build_feature_matrix(records, id_to_label_map, val_ids,   config, label_to_id, feat_names)
    X_test,  y_test,  test_kept = build_feature_matrix(records, id_to_label_map, test_ids, config, label_to_id, feat_names)
    print(f"Feature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    beh_cfg = config.get("behavioral_model", {})
    candidates = {
        "logistic_regression": build_lr_pipeline(beh_cfg.get("logistic_regression", {})),
        "random_forest":       build_rf_pipeline(beh_cfg.get("random_forest", {})),
    }

    val_results: Dict[str, Any] = {}
    trained: Dict[str, Any] = {}
    for name, pipe in candidates.items():
        logger.info("Training %s...", name)
        pipe.fit(X_train, y_train)
        metrics, _, _ = evaluate_pipeline(pipe, X_val, y_val, positive_class_id)
        val_results[name] = metrics
        trained[name] = pipe
        print(f"  [{name}] val: acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | f1={metrics['f1']:.4f}")

    best_name = max(val_results, key=lambda k: val_results[k]["macro_f1"])
    best_pipe  = trained[best_name]
    print(f"\nBest model: {best_name} (val macro_f1={val_results[best_name]['macro_f1']:.4f})")

    # Save artifacts
    import json as _json, joblib
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
        _json.dump(feat_names, fh, indent=2)
    with (output_dir / "val_metrics.json").open("w", encoding="utf-8") as fh:
        _json.dump(val_results, fh, indent=2)
    with (output_dir / "split_statistics.json").open("w", encoding="utf-8") as fh:
        _json.dump({"train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
                    "source": str(split_ids_path)}, fh, indent=2)
    joblib.dump(best_pipe, output_dir / "model.joblib")
    save_feature_analysis(output_dir, X_train, feat_names, best_pipe, best_name, label_to_id)

    model_cfg = {"best_model": best_name, "feature_count": len(feat_names),
                 "label_to_id": label_to_id, "positive_class": positive_class,
                 "validation_results": val_results,
                 "lexical_features_excluded": sorted(LEXICAL_FEATURE_NAMES)}
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as fh:
        _json.dump(model_cfg, fh, indent=2)

    # Final test evaluation
    logger.info("Evaluating on held-out test set...")
    test_metrics, test_preds, test_proba = evaluate_pipeline(best_pipe, X_test, y_test, positive_class_id)
    test_metrics["best_model"] = best_name
    test_metrics["experiment_name"] = experiment_name

    from sklearn.metrics import classification_report
    id_to_lbl = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_lbl[i] for i in range(len(label_to_id))]
    print("\n" + "=" * 64)
    print(f"TEST SET RESULTS ({best_name})")
    print("=" * 64)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
    print()
    print(classification_report(y_test, test_preds, target_names=class_labels, zero_division=0))

    save_test_results(output_dir, test_metrics, y_test, test_preds, label_to_id)
    save_predictions(output_dir, test_kept, y_test, test_preds, test_proba, label_to_id)
    print(f"\nAll artifacts saved to: {output_dir.resolve()}")

    print_three_way_comparison(test_metrics, best_name, label_to_id)


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
