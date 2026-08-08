"""ablation_continued_finetune.py

Controlled ablation: AlephBERT continued fine-tuning (text only, no behavioral features).

Isolates whether Fusion D2's +0.43pp improvement over AlephBERT is due to:
  (A) the 26 pure behavioral features, or
  (B) simply receiving additional gradient steps.

Starting point: SAME checkpoint as D2 used to initialize:
  text-model/outputs/alephbert_baseline_v1/best_model

Hyperparameters matched to D2 as closely as technically possible:
  - Same BERT learning rate: 2e-5
  - Same epochs: 5
  - Same batch size: 8
  - Same weight decay: 0.01
  - Same warmup ratio: 0.1
  - Same gradient clip: 1.0
  - Same random seed: 42
  - Same max_length: 512
  - Same split IDs (from split_ids.json)
  - Same attn_implementation="eager" (MPS constraint)

Unavoidable differences from Fusion D2:
  1. SINGLE LR for all params (BERT + classifier = both 2e-5).
     D2 used differential: BERT@2e-5 + FRESH_head@1e-3.
     Reason: the original Linear(768→2) classifier is already trained;
     using 1e-3 would over-perturb it, while the D2 head was randomly
     initialised. Using 2e-5 uniformly is the standard continued fine-tuning
     regime for an already-trained model.

  2. Head architecture: D2 had Linear(794→256)→ReLU→Dropout→Linear(256→2).
     This ablation has the original Linear(768→2). Capacity differs.

  3. No behavioral features (this is the controlled variable).

These differences mean the comparison is an approximation, not a perfect control.
A perfect control would require training D2 without behavioral features using the
EXACT SAME architecture — but that would mean the fusion head sees only a 768-dim
input instead of 794, which is a different network. The present ablation is the
closest achievable controlled comparison.

Usage:
    python ablation_continued_finetune.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_loader import create_dataset, get_device, load_config, load_corpus_records
from splitter import load_split_ids, print_split_statistics
from train import evaluate, save_test_results, set_full_seed, train_epoch, save_training_artifacts

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("ablation")

EXPERIMENT_NAME = "alephbert_continued_ablation_v1"

KNOWN_RESULTS = {
    "original_alephbert": {
        "accuracy": 0.9695, "macro_f1": 0.9695,
        "precision": 0.9821, "recall": 0.9563, "f1": 0.9690,
    },
    "fusion_d2": {
        "accuracy": 0.9739, "macro_f1": 0.9738,
        "precision": 0.9910, "recall": 0.9563, "f1": 0.9733,
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation: continued AlephBERT fine-tuning, text only."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def print_three_way_comparison(ablation_test: Dict[str, float]) -> None:
    pos = "losing_interest"
    orig = KNOWN_RESULTS["original_alephbert"]
    d2   = KNOWN_RESULTS["fusion_d2"]
    abl  = ablation_test

    print("\n" + "=" * 72)
    print("THREE-WAY COMPARISON — TEST SET (459 conversations, seed=42)")
    print("=" * 72)
    print(f"{'Metric':<30} {'Orig AlephBERT':>14} {'Continued Abl':>14} {'Fusion D2':>12}")
    print("-" * 72)

    rows = [
        ("accuracy",           "accuracy"),
        ("macro F1",           "macro_f1"),
        ("weighted F1",        "weighted_f1"),
        (f"{pos} F1",          "f1"),
        (f"{pos} precision",   "precision"),
        (f"{pos} recall",      "recall"),
    ]
    for label, key in rows:
        o = orig.get(key, float("nan"))
        a = abl.get(key, float("nan"))
        d = d2.get(key, float("nan"))
        print(f"  {label:<28} {o:>14.4f} {a:>14.4f} {d:>12.4f}")

    print("-" * 72)

    a_macro  = abl.get("macro_f1", float("nan"))
    o_macro  = orig["macro_f1"]
    d2_macro = d2["macro_f1"]

    d_abl_vs_orig = (a_macro - o_macro) * 100
    d_d2_vs_orig  = (d2_macro - o_macro) * 100
    d_d2_vs_abl   = (d2_macro - a_macro) * 100

    print(f"\n  Δ macro F1 (pp):")
    print(f"    Continued Ablation  − Original AlephBERT : {d_abl_vs_orig:+.2f}pp")
    print(f"    Fusion D2           − Original AlephBERT : {d_d2_vs_orig:+.2f}pp")
    print(f"    Fusion D2           − Continued Ablation : {d_d2_vs_abl:+.2f}pp")

    print()
    print("  Interpretation:")
    threshold = 1.0

    if abs(d_d2_vs_abl) < threshold:
        conclusion = (
            "Case A — Continued fine-tuning accounts for the improvement.\n"
            "  Fusion D2's gain over original AlephBERT cannot be attributed\n"
            "  to behavioral features; additional gradient steps explain the gap."
        )
    elif d_d2_vs_abl > threshold:
        conclusion = (
            "Case B — Evidence that behavioral features provide complementary signal.\n"
            "  Fusion D2 outperforms continued fine-tuning; the behavioral\n"
            "  features appear to add predictive information beyond text."
        )
    else:
        conclusion = (
            "Case C — Behavioral fusion provides no demonstrated benefit.\n"
            "  Continued fine-tuning matches or exceeds Fusion D2;\n"
            "  the behavioral features may slightly hurt generalization."
        )

    for line in conclusion.split("\n"):
        print(f"  {line}")

    print()
    print("  CAVEAT: Test set = 459 conversations. Differences below ~1pp should be")
    print("  interpreted cautiously without repeated-seed experiments or statistical")
    print("  significance testing (e.g. McNemar's test on prediction disagreements).")
    print()
    print("  Unavoidable differences from Fusion D2:")
    print("    - Single LR=2e-5 vs D2's differential BERT@2e-5 / fresh_head@1e-3")
    print("    - Head: Linear(768→2) vs D2's Linear(794→256)→ReLU→Dropout→Linear(256→2)")
    print("    - No behavioral features (controlled variable)")
    print("=" * 72)


def run(config_path: Path) -> None:
    config   = load_config(config_path)
    base_dir = config_path.resolve().parent
    device   = get_device()
    seed     = int(config.get("random_seed", 42))
    set_full_seed(seed)

    # ---- Ablation config: match D2 conditions ----
    fusion_cfg = config.get("fusion", {})
    d2_cfg     = fusion_cfg.get("finetuned", {})

    checkpoint_dir = base_dir / fusion_cfg.get("bert_checkpoint",
                                                "outputs/alephbert_baseline_v1/best_model")
    positive_class = fusion_cfg.get("positive_class", "losing_interest")
    epochs         = int(d2_cfg.get("epochs", 5))
    lr             = float(d2_cfg.get("learning_rate", 2e-5))
    batch_size     = int(d2_cfg.get("batch_size", 8))
    wd             = float(d2_cfg.get("weight_decay", 0.01))
    warmup_ratio   = float(d2_cfg.get("warmup_ratio", 0.1))
    grad_clip      = float(d2_cfg.get("gradient_clip", 1.0))

    output_dir = base_dir / "outputs" / EXPERIMENT_NAME

    # Guard: do not overwrite existing experiments
    protected = [
        base_dir / "outputs" / "alephbert_baseline_v1",
        base_dir / "outputs" / "fusion_alephbert_behavioral_v1",
        base_dir / "outputs" / "behavioral_baseline_v1",
        base_dir / "outputs" / "pure_behavioral_baseline_v1",
    ]
    for p in protected:
        if p == output_dir:
            raise RuntimeError(f"Output directory would overwrite protected path: {p}")

    label_mapping = config["data"]["label_mapping"]
    label_to_id   = {lbl: i for i, lbl in enumerate(sorted(label_mapping.keys()))}
    id_to_label   = {v: k for k, v in label_to_id.items()}

    print("=" * 64)
    print("Ablation: Continued AlephBERT Fine-Tuning (text only)")
    print("=" * 64)
    print(f"  Checkpoint    : {checkpoint_dir}")
    print(f"  Epochs        : {epochs}   (matches D2)")
    print(f"  LR            : {lr}  (BERT LR from D2; single for all params)")
    print(f"  Batch size    : {batch_size}  (matches D2)")
    print(f"  Seed          : {seed}")
    print(f"  Device        : {device}")
    print(f"  max_length    : {config['tokenizer']['max_length']}")
    print(f"  Output        : {output_dir}")
    print()
    print("  Unavoidable differences from D2:")
    print("    - Single LR vs differential BERT@2e-5 / head@1e-3")
    print("    - Head: Linear(768→2), not the fusion MLP")
    print("    - No behavioral features (controlled variable)")
    print("=" * 64)

    # ---- Load corpus and split IDs ----
    records = load_corpus_records(config, base_dir=base_dir)
    logger.info("Loaded %d records.", len(records))

    split_ids_path = base_dir / config.get("split", {}).get("split_ids_output",
                                                              "outputs/split_ids.json")
    train_ids, val_ids, test_ids = load_split_ids(split_ids_path)

    assert not (set(train_ids) & set(val_ids)),  "train/val overlap in split_ids"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap in split_ids"
    assert not (set(val_ids)   & set(test_ids)), "val/test overlap in split_ids"

    print("\nSplit statistics:")
    print_split_statistics(records, train_ids, val_ids, test_ids,
                           label_mapping,
                           id_field=config["data"].get("id_field", "conversation_id"),
                           label_field=config["data"].get("label_field", "final_outcome"))
    print()

    # ---- Load tokenizer and model ----
    logger.info("Loading tokenizer from %s ...", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    logger.info("Loading AlephBERT checkpoint from %s ...", checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir),
        num_labels=2,
        ignore_mismatched_sizes=False,
        attn_implementation="eager",   # required for MPS training with dropout
    )
    model.to(device)

    # Verify no behavioral features are accidentally attached
    assert not hasattr(model, "behavioral_head"), \
        "Model should not have a behavioral_head attribute"
    assert not hasattr(model, "beh"), \
        "Model should not have a beh attribute"

    # ---- Datasets and DataLoaders ----
    dataloader_cfg = config.get("dataloader", {})
    n_workers = int(dataloader_cfg.get("num_workers", 0))

    logger.info("Building datasets...")
    train_ds = create_dataset(config, tokenizer, base_dir=base_dir, subset_ids=set(train_ids))
    val_ds   = create_dataset(config, tokenizer, base_dir=base_dir, subset_ids=set(val_ids))
    test_ds  = create_dataset(config, tokenizer, base_dir=base_dir, subset_ids=set(test_ids))
    logger.info("Dataset sizes: train=%d  val=%d  test=%d",
                len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size,  shuffle=True,  num_workers=n_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=n_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False, num_workers=n_workers)

    # ---- Optimizer + scheduler (single LR for all params) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")
    print(f"(D2 had same total steps: {total_steps})")

    # ---- Training loop ----
    best_val_macro_f1 = -1.0
    best_epoch = -1
    training_history: List[Dict[str, Any]] = []
    best_model_dir = output_dir / "best_model"

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, label_to_id, positive_class)

        record: Dict[str, Any] = {
            "epoch": epoch, "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        training_history.append(record)
        is_best = val_metrics["macro_f1"] > best_val_macro_f1

        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.4f} | "
              f"macro_f1={val_metrics['macro_f1']:.4f} | "
              f"f1={val_metrics['f1']:.4f}  {'<-- best' if is_best else ''}")

        if is_best:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            logger.info("Saved best checkpoint (epoch %d, macro_f1=%.4f)",
                        epoch, best_val_macro_f1)

    print(f"\nBest epoch: {best_epoch}  Val macro_f1: {best_val_macro_f1:.4f}")

    # ---- Reload best checkpoint ----
    logger.info("Loading best checkpoint...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(best_model_dir),
        num_labels=2,
        attn_implementation="eager",
    )
    model.to(device)

    # ---- Final test evaluation (once) ----
    logger.info("Evaluating on held-out test set (once)...")
    test_metrics, test_probs, test_preds, test_rids = evaluate(
        model, test_loader, device, label_to_id, positive_class
    )
    test_metrics["best_epoch"] = best_epoch
    test_metrics["experiment"] = EXPERIMENT_NAME

    id_map = {rid: i for i, rid in enumerate(test_ds.record_ids)}
    y_true = test_ds.labels.numpy()[[id_map[rid] for rid in test_rids]]

    id_to_lbl = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_lbl[i] for i in range(len(label_to_id))]

    print("\n" + "=" * 64)
    print("TEST SET RESULTS (Continued AlephBERT Ablation)")
    print("=" * 64)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
    print()
    print(classification_report(y_true, test_preds, target_names=class_labels, zero_division=0))

    # ---- Save all artifacts ----
    ablation_config = {
        "experiment": EXPERIMENT_NAME,
        "starting_checkpoint": str(checkpoint_dir),
        "epochs": epochs, "best_epoch": best_epoch,
        "learning_rate": lr,
        "note": "single LR for all params; matches D2's bert_lr=2e-5",
        "batch_size": batch_size, "weight_decay": wd,
        "warmup_ratio": warmup_ratio, "gradient_clip": grad_clip,
        "seed": seed, "max_length": config["tokenizer"]["max_length"],
        "behavioral_features": "NONE — text only",
        "unavoidable_differences_from_d2": [
            "Single LR=2e-5 vs D2's differential BERT@2e-5 / fresh_head@1e-3",
            "Head architecture: Linear(768→2) vs D2's fusion MLP Linear(794→256→2)",
            "No behavioral features (controlled variable)",
        ],
    }
    split_stats = {
        "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
    }
    save_training_artifacts(
        output_dir, model, tokenizer, ablation_config, split_stats, training_history,
    )
    save_test_results(output_dir, test_metrics, y_true, test_preds, test_probs,
                      test_rids, label_to_id)
    print(f"\nAll artifacts saved to: {output_dir.resolve()}")

    # ---- Three-way comparison ----
    print_three_way_comparison(test_metrics)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.config)
    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
