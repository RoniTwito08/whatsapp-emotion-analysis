"""early_detection_e2.py

Experiment E2 — Prefix-Aware AlephBERT Training for Early Detection.

Research question:
    Can AlephBERT learn to predict the final interest outcome from incomplete
    Hebrew WhatsApp sales conversations if trained on partial conversation prefixes?

Approach:
    For every training conversation, generate four prefix examples (25/50/75/100%).
    All prefixes of a conversation inherit the same final binary label.
    Train AlephBERT initialized from the current best full-conversation checkpoint.

Data split integrity:
    Prefixes are generated AFTER assigning conversations to train/val/test.
    All prefixes from one conversation stay in the same split.
    The exact 459 test conversation IDs from E1 are reused unchanged.

Checkpoint selection:
    Avg macro F1 across the three early prefixes: {25%, 50%, 75%}.
    (100% is excluded from the selection metric so the model optimises for
     early detection, not for re-learning the full-conversation task.)

Starting point:
    alephbert_continued_ablation_v1/best_model — already fine-tuned, conservative LR=1e-5.

Usage:
    python early_detection_e2.py --config config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_loader import get_device, load_config
from early_detection_e1 import build_prefix_text, prefix_length
from splitter import load_split_ids
from train import evaluate, set_full_seed, train_epoch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("e2")

EXPERIMENT_NAME = "early_detection_e2"


# ---------------------------------------------------------------------------
# Prefix-augmented dataset
# ---------------------------------------------------------------------------

class PrefixAugmentedDataset(Dataset):
    """One entry per (conversation, fraction) pair.

    For N conversations and F fractions → N×F examples.
    All prefixes of one conversation have the same final label and the same
    conversation_id; they differ only in the text (prefix length).

    Tokenisation happens once at __init__ for efficiency.
    """

    def __init__(
        self,
        conversations: List[Dict[str, Any]],
        fractions: List[float],
        data_config: Dict[str, Any],
        tokenizer: Any,
        max_length: int,
        label_to_id: Dict[str, int],
    ) -> None:
        label_mapping = data_config.get("label_mapping", {})
        raw_to_binary: Dict[str, str] = {
            raw: mapped for mapped, raws in label_mapping.items() for raw in raws
        }
        id_field    = data_config.get("id_field", "conversation_id")
        label_field = data_config.get("label_field", "final_outcome")

        records: List[Tuple[str, float, int]] = []  # (conv_id, fraction, label_id)
        texts: List[str] = []

        for conv in conversations:
            raw_label = str(conv.get(label_field, ""))
            binary_label = raw_to_binary.get(raw_label)
            if binary_label is None:
                continue
            label_id = label_to_id[binary_label]
            cid = str(conv.get(id_field, ""))

            for frac in fractions:
                text, _, _ = build_prefix_text(conv, frac, data_config)
                texts.append(text)
                records.append((cid, frac, label_id))

        # Batch tokenise all examples at once
        if texts:
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            self.input_ids      = enc["input_ids"].long()
            self.attention_mask = enc["attention_mask"].long()
            self.token_type_ids = (
                enc["token_type_ids"].long() if "token_type_ids" in enc else None
            )
        else:
            self.input_ids      = torch.zeros(0, max_length, dtype=torch.long)
            self.attention_mask = torch.zeros(0, max_length, dtype=torch.long)
            self.token_type_ids = None

        self.labels     = torch.tensor([r[2] for r in records], dtype=torch.long)
        self.records    = records          # list of (conv_id, fraction, label_id)
        self.conv_ids   = [r[0] for r in records]
        self.fractions  = [r[1] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
            "record_id":      self.conv_ids[idx],
            "fraction":       self.fractions[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


# ---------------------------------------------------------------------------
# Per-fraction evaluation
# ---------------------------------------------------------------------------

def evaluate_per_fraction(
    model: nn.Module,
    conversations: List[Dict[str, Any]],
    fractions: List[float],
    data_config: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    label_to_id: Dict[str, int],
    positive_class: str,
    eval_batch_size: int = 32,
) -> Dict[float, Dict[str, Any]]:
    """Evaluate the model at each fraction independently.

    Returns dict: fraction → {metrics, probs, preds, record_ids}
    """
    results: Dict[float, Dict[str, Any]] = {}
    for frac in fractions:
        ds = PrefixAugmentedDataset(
            conversations, [frac], data_config, tokenizer, max_length, label_to_id
        )
        loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
        metrics, probs, preds, rids = evaluate(model, loader, device, label_to_id, positive_class)
        results[frac] = {
            "metrics": metrics, "probs": probs, "preds": preds, "record_ids": rids,
            "y_true": ds.labels.numpy(),
        }
    return results


def selection_metric(
    val_results: Dict[float, Dict[str, Any]],
    selection_fractions: List[float],
) -> float:
    """Avg macro F1 across the specified fractions (checkpoint selection metric)."""
    vals = [val_results[f]["metrics"]["macro_f1"] for f in selection_fractions if f in val_results]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_prefix_metrics(
    output_dir: Path,
    fraction: float,
    metrics: Dict[str, float],
    probs: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
    record_ids: List[str],
    label_to_id: Dict[str, int],
    conversations_lookup: Dict[str, Dict[str, Any]],
    data_config: Dict[str, Any],
) -> None:
    tag = f"{int(fraction * 100):03d}pct"
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]

    with (output_dir / f"test_metrics_{tag}.json").open("w", encoding="utf-8") as fh:
        json.dump({**metrics, "fraction": fraction}, fh, indent=2)

    report = classification_report(y_true, preds, target_names=class_labels, zero_division=0)
    (output_dir / f"classification_report_{tag}.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, preds, labels=list(range(len(class_labels))))
    cm_rows = [[""] + [f"pred_{c}" for c in class_labels]]
    for i, c in enumerate(class_labels):
        cm_rows.append([f"actual_{c}"] + [str(v) for v in cm[i]])
    with (output_dir / f"confusion_matrix_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(cm_rows)

    messages_field = data_config.get("messages_field", "messages")
    id_field       = data_config.get("id_field", "conversation_id")
    fieldnames = [
        "conversation_id", "fraction",
        "num_total_msgs_in_prefix", "num_customer_msgs_in_prefix",
        "actual_label", "predicted_label",
        "probability_interested", "probability_losing_interest",
        "correct",
    ]
    with (output_dir / f"predictions_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, rid in enumerate(record_ids):
            conv = conversations_lookup.get(rid, {})
            msgs = conv.get(messages_field, [])
            plen = prefix_length(len(msgs), fraction)
            prefix_msgs = msgs[:plen]
            n_cust = sum(
                1 for m in prefix_msgs
                if m.get(data_config.get("role_field", "role"))
                in data_config.get("included_roles", ["customer"])
            )
            writer.writerow({
                "conversation_id": rid,
                "fraction": fraction,
                "num_total_msgs_in_prefix": plen,
                "num_customer_msgs_in_prefix": n_cust,
                "actual_label": id_to_label[int(y_true[i])],
                "predicted_label": id_to_label[int(preds[i])],
                "probability_interested": f"{probs[i, label_to_id['interested']]:.6f}",
                "probability_losing_interest": f"{probs[i, label_to_id['losing_interest']]:.6f}",
                "correct": int(y_true[i] == preds[i]),
            })


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_e1_vs_e2_comparison(
    e1_path: Path,
    e2_test: Dict[float, Dict[str, Any]],
    fractions: List[float],
    label_to_id: Dict[str, int],
) -> None:
    try:
        e1 = json.loads(e1_path.read_text())
    except Exception:
        e1 = {}

    id_to_label = {v: k for k, v in label_to_id.items()}

    print("\n" + "=" * 88)
    print("E1 vs E2 COMPARISON — TEST SET (459 conversations)")
    print("=" * 88)
    print(f"  {'Prefix':>8}  {'E1 Mac.F1':>10}  {'E2 Mac.F1':>10}  {'Δ Mac.F1':>10}  "
          f"{'E2 LI F1':>10}  {'E2 LI Prec':>11}  {'E2 LI Rec':>11}")
    print("  " + "-" * 76)

    for frac in fractions:
        frac_str = str(int(frac * 100))
        e1_macro = e1.get(frac_str, {}).get("macro_f1", float("nan"))
        e2m = e2_test[frac]["metrics"]
        e2_macro = e2m["macro_f1"]
        delta    = (e2_macro - e1_macro) * 100 if not np.isnan(e1_macro) else float("nan")
        sign     = "+" if delta >= 0 else ""
        print(f"  {int(frac*100):>7}%  {e1_macro:>10.4f}  {e2_macro:>10.4f}  "
              f"{sign}{delta:>9.2f}pp  "
              f"{e2m['f1']:>10.4f}  {e2m['precision']:>11.4f}  {e2m['recall']:>11.4f}")

    # Predicted-class distribution per fraction
    print()
    print(f"  {'Prefix':>8}  {'% pred interested':>20}  {'% pred losing_interest':>23}")
    print("  " + "-" * 55)
    for frac in fractions:
        preds = e2_test[frac]["preds"]
        n = len(preds)
        n_int  = int((preds == label_to_id["interested"]).sum())
        n_li   = int((preds == label_to_id["losing_interest"]).sum())
        print(f"  {int(frac*100):>7}%  {n_int/n*100:>19.1f}%  {n_li/n*100:>22.1f}%")

    print("=" * 88)


def print_threshold_analysis(
    e2_test: Dict[float, Dict[str, Any]],
    fractions: List[float],
) -> None:
    print("\n  Earliest prefix reaching performance thresholds (E2):")
    checks = [
        ("Macro F1 ≥ 0.70", "macro_f1",  0.70),
        ("Macro F1 ≥ 0.80", "macro_f1",  0.80),
        ("Macro F1 ≥ 0.90", "macro_f1",  0.90),
        ("LI recall ≥ 0.90 AND prec ≥ 0.80", None, None),
    ]
    for label, key, thr in checks:
        if key is not None:
            earliest = next(
                (f"{int(f*100)}%" for f in fractions if e2_test[f]["metrics"].get(key, 0) >= thr),
                "not reached"
            )
        else:
            earliest = next(
                (f"{int(f*100)}%" for f in fractions
                 if (e2_test[f]["metrics"].get("recall", 0) >= 0.90 and
                     e2_test[f]["metrics"].get("precision", 0) >= 0.80)),
                "not reached"
            )
        print(f"    {label:<44}: {earliest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2: prefix-aware AlephBERT training.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def run(config_path: Path) -> None:
    config   = load_config(config_path)
    base_dir = config_path.resolve().parent
    device   = get_device()

    e2_cfg          = config.get("early_detection_e2", {})
    data_config     = config["data"]
    max_length      = int(config["tokenizer"]["max_length"])
    seed            = int(config.get("random_seed", 42))
    positive_class  = e2_cfg.get("positive_class", "losing_interest")
    training_fracs  = e2_cfg.get("training_fractions", [0.25, 0.50, 0.75, 1.0])
    eval_fracs      = e2_cfg.get("eval_fractions",     [0.25, 0.50, 0.75, 1.0])
    sel_fracs       = e2_cfg.get("checkpoint_selection_fractions", [0.25, 0.50, 0.75])
    epochs          = int(e2_cfg.get("epochs", 5))
    lr              = float(e2_cfg.get("learning_rate", 1e-5))
    batch_size      = int(e2_cfg.get("batch_size", 8))
    eval_bs         = int(e2_cfg.get("eval_batch_size", 32))
    wd              = float(e2_cfg.get("weight_decay", 0.01))
    warmup_ratio    = float(e2_cfg.get("warmup_ratio", 0.1))
    grad_clip       = float(e2_cfg.get("gradient_clip", 1.0))

    set_full_seed(seed)

    label_mapping = data_config["label_mapping"]
    label_to_id   = {lbl: i for i, lbl in enumerate(sorted(label_mapping.keys()))}

    output_dir = base_dir / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    # Guard: do not overwrite protected experiments
    protected = ["alephbert_baseline_v1", "alephbert_continued_ablation_v1",
                 "early_detection_e1", "fusion_alephbert_behavioral_v1",
                 "behavioral_baseline_v1", "pure_behavioral_baseline_v1"]
    for name in protected:
        if (base_dir / "outputs" / name) == output_dir:
            raise RuntimeError(f"Would overwrite protected output: {name}")

    print("=" * 64)
    print("Early Detection E2 — Prefix-Aware Training")
    print("=" * 64)

    # ---- Load corpus ----
    input_path = Path(data_config["input_path"])
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    with input_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    corpus = raw if isinstance(raw, list) else next(
        (raw[k] for k in ("data","records","conversations","items") if k in raw), [raw]
    )
    id_field = data_config.get("id_field", "conversation_id")
    conversations_lookup: Dict[str, Dict[str, Any]] = {
        str(c.get(id_field, "")): c for c in corpus
    }

    # ---- Load split IDs ----
    split_ids_path = base_dir / config.get("split", {}).get("split_ids_output",
                                                              "outputs/split_ids.json")
    train_ids, val_ids, test_ids = load_split_ids(split_ids_path)
    train_convs = [conversations_lookup[cid] for cid in train_ids if cid in conversations_lookup]
    val_convs   = [conversations_lookup[cid] for cid in val_ids   if cid in conversations_lookup]
    test_convs  = [conversations_lookup[cid] for cid in test_ids  if cid in conversations_lookup]

    n_train_prefix = len(train_convs) * len(training_fracs)
    n_val_prefix   = len(val_convs)   * len(eval_fracs)
    n_test_prefix  = len(test_convs)  * len(eval_fracs)
    print(f"  Train : {len(train_convs)} unique convs × {len(training_fracs)} fracs = {n_train_prefix} examples")
    print(f"  Val   : {len(val_convs)} unique convs × {len(eval_fracs)} fracs = {n_val_prefix} examples (eval per-fraction)")
    print(f"  Test  : {len(test_convs)} unique convs × {len(eval_fracs)} fracs = {n_test_prefix} examples (eval per-fraction)")
    print(f"  Checkpoint selection: avg macro F1 across {[int(f*100) for f in sel_fracs]}%")
    print(f"  Fractions for training: {[int(f*100) for f in training_fracs]}%")
    print(f"  LR={lr}  epochs={epochs}  batch={batch_size}  device={device}")
    print(f"  Starting from: {e2_cfg.get('starting_checkpoint')}")
    print("=" * 64)

    # ---- Save augmentation statistics ----
    aug_stats = {
        "unique_train_conversations": len(train_convs),
        "unique_val_conversations":   len(val_convs),
        "unique_test_conversations":  len(test_convs),
        "training_fractions":  training_fracs,
        "eval_fractions":      eval_fracs,
        "prefix_train_examples": n_train_prefix,
        "prefix_val_examples":   n_val_prefix,
        "prefix_test_examples":  n_test_prefix,
        "checkpoint_selection_fractions": sel_fracs,
        "checkpoint_selection_metric": "avg macro_f1 across selection fractions",
    }
    with (output_dir / "prefix_augmentation_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(aug_stats, fh, indent=2)

    # ---- Load tokenizer and model ----
    ckpt_dir = base_dir / e2_cfg.get("starting_checkpoint",
                                      "outputs/alephbert_continued_ablation_v1/best_model")
    logger.info("Loading tokenizer from %s ...", ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    logger.info("Loading model from %s ...", ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=2, attn_implementation="eager",
    )
    model.to(device)

    # ---- Build training dataset and DataLoader ----
    logger.info("Tokenising %d training prefix examples...", n_train_prefix)
    train_ds = PrefixAugmentedDataset(
        train_convs, training_fracs, data_config, tokenizer, max_length, label_to_id
    )
    logger.info("Training dataset built: %d examples.", len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"\nTotal training steps: {total_steps}  Warmup: {warmup_steps}")

    # ---- Training loop ----
    best_sel_metric = -1.0
    best_epoch      = -1
    training_history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip)

        # Per-fraction validation
        val_frac_results = evaluate_per_fraction(
            model, val_convs, eval_fracs, data_config, tokenizer,
            max_length, device, label_to_id, positive_class, eval_bs,
        )
        sel = selection_metric(val_frac_results, sel_fracs)
        is_best = sel > best_sel_metric

        record: Dict[str, Any] = {
            "epoch": epoch, "train_loss": train_loss,
            "val_selection_metric": sel,
            **{f"val_{int(f*100)}pct_{k}": v
               for f, res in val_frac_results.items()
               for k, v in res["metrics"].items()},
        }
        training_history.append(record)

        # Per-fraction print
        frac_strs = "  ".join(
            f"{int(f*100)}%: mF1={val_frac_results[f]['metrics']['macro_f1']:.4f}"
            for f in eval_fracs
        )
        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} | sel={sel:.4f} | "
              f"{frac_strs}  {'<-- best' if is_best else ''}")

        if is_best:
            best_sel_metric = sel
            best_epoch = epoch
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            logger.info("Saved best model (epoch %d, sel=%.4f)", epoch, sel)

    print(f"\nBest epoch: {best_epoch}  Selection metric: {best_sel_metric:.4f}")

    # ---- Save training history ----
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(training_history, fh, indent=2)

    # ---- Save experiment config ----
    exp_cfg_record = {
        "experiment": EXPERIMENT_NAME,
        "starting_checkpoint": str(ckpt_dir),
        "training_fractions": training_fracs,
        "eval_fractions": eval_fracs,
        "checkpoint_selection_fractions": sel_fracs,
        "epochs": epochs, "best_epoch": best_epoch,
        "learning_rate": lr, "batch_size": batch_size,
        "weight_decay": wd, "warmup_ratio": warmup_ratio,
        "seed": seed, "max_length": max_length,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as fh:
        json.dump(exp_cfg_record, fh, indent=2)

    # ---- Reload best checkpoint ----
    logger.info("Loading best checkpoint from epoch %d ...", best_epoch)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(best_model_dir), num_labels=2, attn_implementation="eager",
    )
    model.to(device)

    # ---- Final test evaluation (once per fraction) ----
    logger.info("Final test evaluation...")
    test_frac_results = evaluate_per_fraction(
        model, test_convs, eval_fracs, data_config, tokenizer,
        max_length, device, label_to_id, positive_class, eval_bs,
    )

    # Save per-fraction test results
    id_to_lbl = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_lbl[i] for i in range(len(label_to_id))]

    print("\n" + "=" * 64)
    print("TEST SET RESULTS PER PREFIX (E2)")
    print("=" * 64)
    for frac in eval_fracs:
        m = test_frac_results[frac]["metrics"]
        preds = test_frac_results[frac]["preds"]
        y_true = test_frac_results[frac]["y_true"]
        n = len(preds)
        n_int = int((preds == label_to_id["interested"]).sum())
        n_li  = int((preds == label_to_id["losing_interest"]).sum())

        print(f"\n  [{int(frac*100)}%] acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"f1={m['f1']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}")
        print(f"  Pred dist: interested={n_int/n*100:.1f}%  losing_interest={n_li/n*100:.1f}%")
        print(classification_report(y_true, preds, target_names=class_labels, zero_division=0))

        save_prefix_metrics(
            output_dir, frac, m,
            test_frac_results[frac]["probs"], preds, y_true,
            test_frac_results[frac]["record_ids"],
            label_to_id, conversations_lookup, data_config,
        )

    # Save comparison table JSON
    e1_path = base_dir / "outputs" / "early_detection_e1" / "comparison_table.json"
    e1_data = {}
    if e1_path.exists():
        e1_data = json.loads(e1_path.read_text())

    comparison = {}
    for frac in eval_fracs:
        frac_str = str(int(frac * 100))
        e1m = e1_data.get(frac_str, {})
        e2m = test_frac_results[frac]["metrics"]
        comparison[frac_str] = {
            "e1_macro_f1":   e1m.get("macro_f1", None),
            "e2_macro_f1":   e2m["macro_f1"],
            "e1_f1":         e1m.get("f1", None),
            "e2_f1":         e2m["f1"],
            "e2_precision":  e2m["precision"],
            "e2_recall":     e2m["recall"],
            "e2_accuracy":   e2m["accuracy"],
        }
    with (output_dir / "e1_vs_e2_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    print_e1_vs_e2_comparison(e1_path, test_frac_results, eval_fracs, label_to_id)
    print_threshold_analysis(test_frac_results, eval_fracs)
    print(f"\nAll artifacts saved to: {output_dir.resolve()}")


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
