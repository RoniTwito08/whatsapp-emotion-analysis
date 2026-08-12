"""early_detection_e3.py

Experiment E3 — Prefix-Aware AlephBERT + Behavioral Feature Fusion.

Research question:
    Do pure behavioral signals (message counts, lengths, delays, session
    structure) provide additional value during EARLY stages of a Hebrew
    WhatsApp sales conversation, when textual evidence is still incomplete?

Design:
    Same prefix augmentation as E2 (25 / 50 / 75 / 100%).
    For every (conversation, fraction) pair:
      - text input:       customer messages from the prefix ONLY
      - behavioral input: 26 pure non-lexical features from the prefix ONLY
    Features are concatenated after standardisation:
      concat(pooler_output[768], standardised_behavioral[26]) → MLP → 2

    The StandardScaler is fitted on training-prefix features only (no val/test
    information leaks into the scaler).

Leakage guarantee:
    For a given fraction f, only the first ceil(N * f) messages of a
    conversation are visible. Behavioural features are extracted from this
    sliced message list. No future message information enters the model.

Fair comparison with E2:
    - Same starting checkpoint (alephbert_continued_ablation_v1/best_model)
    - Same prefix fractions and training procedure
    - Same checkpoint-selection criterion: avg macro F1 across {25%, 50%, 75%}
    - Same number of epochs and seed

Usage:
    python early_detection_e3.py --config config.json
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from behavioral_features_design import extract_behavioral_features, pure_behavioral_feature_names
from data_loader import get_device, load_config
from early_detection_e1 import build_prefix_text, prefix_length
from early_detection_e2 import selection_metric, save_prefix_metrics
from fusion_train import FusionModel
from fusion_train import train_epoch as fusion_train_epoch
from fusion_train import evaluate as fusion_evaluate
from splitter import load_split_ids
from train import set_full_seed

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("e3")

EXPERIMENT_NAME = "early_detection_e3"
FRACTIONS = [0.25, 0.50, 0.75, 1.00]


# ---------------------------------------------------------------------------
# Prefix behavioral feature extraction (leakage-safe)
# ---------------------------------------------------------------------------

def extract_prefix_behavioral(
    conversation: Dict[str, Any],
    fraction: float,
    feat_names: List[str],
    data_config: Dict[str, Any],
) -> List[float]:
    """Extract behavioral features from the prefix slice ONLY.

    The conversation is sliced to the first ceil(N * fraction) messages
    before feature extraction, so no information from later messages can
    influence the feature values.

    Args:
        conversation: Full conversation dict from the corpus.
        fraction:     Fraction of total messages to include.
        feat_names:   Ordered list of pure behavioral feature names.
        data_config:  config['data'] section (for field names).

    Returns:
        List of floats in feat_names order (zeros for missing features).
    """
    messages_field = data_config.get("messages_field", "messages")
    all_msgs = conversation.get(messages_field, [])
    plen = prefix_length(len(all_msgs), fraction)

    # Create a shallow copy of the conversation with only the prefix messages.
    # This is the KEY leakage prevention: extract_behavioral_features sees
    # only messages[:plen], never any later messages.
    sliced = {**conversation, messages_field: all_msgs[:plen]}
    feats = extract_behavioral_features(sliced)
    return [float(feats.get(f, 0.0)) for f in feat_names]


# ---------------------------------------------------------------------------
# Prefix-fusion dataset
# ---------------------------------------------------------------------------

class PrefixFusionDataset(Dataset):
    """One entry per (conversation, fraction) pair with text + behavioral features.

    Text is tokenised from the prefix only.
    Behavioral features are extracted from the prefix only (no future leakage).
    Behavioral features are optionally standardised by a pre-fitted StandardScaler.
    """

    def __init__(
        self,
        conversations: List[Dict[str, Any]],
        fractions: List[float],
        data_config: Dict[str, Any],
        tokenizer: Any,
        max_length: int,
        label_to_id: Dict[str, int],
        feat_names: List[str],
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        label_mapping = data_config.get("label_mapping", {})
        raw_to_binary: Dict[str, str] = {
            raw: mapped for mapped, raws in label_mapping.items() for raw in raws
        }
        id_field    = data_config.get("id_field", "conversation_id")
        label_field = data_config.get("label_field", "final_outcome")

        records: List[Tuple[str, float, int]] = []
        texts: List[str] = []
        beh_rows: List[List[float]] = []

        for conv in conversations:
            raw_label = str(conv.get(label_field, ""))
            binary_label = raw_to_binary.get(raw_label)
            if binary_label is None:
                continue
            label_id = label_to_id[binary_label]
            cid = str(conv.get(id_field, ""))

            for frac in fractions:
                text, _, _ = build_prefix_text(conv, frac, data_config)
                beh_vec    = extract_prefix_behavioral(conv, frac, feat_names, data_config)
                texts.append(text)
                beh_rows.append(beh_vec)
                records.append((cid, frac, label_id))

        # Batch tokenise
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

        # Standardise behavioral features (transform only if scaler is provided)
        X_beh = np.array(beh_rows, dtype=np.float64)
        if scaler is not None and len(X_beh) > 0:
            X_beh = scaler.transform(X_beh)
        self.beh_tensor = torch.tensor(X_beh, dtype=torch.float32)

        self.labels    = torch.tensor([r[2] for r in records], dtype=torch.long)
        self.records   = records
        self.conv_ids  = [r[0] for r in records]
        self.fractions = [r[1] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "input_ids":           self.input_ids[idx],
            "attention_mask":      self.attention_mask[idx],
            "behavioral_features": self.beh_tensor[idx],
            "labels":              self.labels[idx],
            "record_id":           self.conv_ids[idx],
            "fraction":            self.fractions[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


# ---------------------------------------------------------------------------
# Per-fraction evaluation (fusion model)
# ---------------------------------------------------------------------------

def evaluate_e3_per_fraction(
    model: FusionModel,
    conversations: List[Dict[str, Any]],
    fractions: List[float],
    data_config: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    label_to_id: Dict[str, int],
    positive_class: str,
    feat_names: List[str],
    scaler: StandardScaler,
    eval_batch_size: int = 32,
) -> Dict[float, Dict[str, Any]]:
    """Evaluate FusionModel at each prefix fraction independently."""
    results: Dict[float, Dict[str, Any]] = {}
    for frac in fractions:
        ds = PrefixFusionDataset(
            conversations, [frac], data_config, tokenizer, max_length,
            label_to_id, feat_names, scaler=scaler,
        )
        loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
        metrics, probs, preds, rids = fusion_evaluate(
            model, loader, device, label_to_id, positive_class
        )
        results[frac] = {
            "metrics": metrics, "probs": probs, "preds": preds, "record_ids": rids,
            "y_true": ds.labels.numpy(),
        }
    return results


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def compute_error_analysis(
    e2_preds_path: Path,
    e3_preds: np.ndarray,
    e3_y_true: np.ndarray,
    e3_rids: List[str],
    label_to_id: Dict[str, int],
    fraction: float,
) -> Dict[str, Any]:
    """Compare E2 and E3 predictions for a given fraction.

    Loads E2 predictions from its saved CSV, aligns on conversation_id, and
    produces counts / ID lists for the four outcome categories.
    """
    if not e2_preds_path.exists():
        return {"error": f"E2 predictions not found at {e2_preds_path}"}

    id_to_label = {v: k for k, v in label_to_id.items()}
    # Load E2 predictions
    e2_by_cid: Dict[str, Dict[str, Any]] = {}
    with e2_preds_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            e2_by_cid[row["conversation_id"]] = {
                "actual": row["actual_label"],
                "predicted": row["predicted_label"],
                "correct": int(row["correct"]),
            }

    e2_right_e3_wrong: List[str] = []
    e2_wrong_e3_right: List[str] = []
    both_wrong:         List[str] = []
    both_right: int = 0

    for i, cid in enumerate(e3_rids):
        e3_correct = int(e3_y_true[i] == e3_preds[i])
        e2_info = e2_by_cid.get(cid)
        if e2_info is None:
            continue
        e2_correct = e2_info["correct"]

        if e2_correct and not e3_correct:
            e2_right_e3_wrong.append(cid)
        elif not e2_correct and e3_correct:
            e2_wrong_e3_right.append(cid)
        elif not e2_correct and not e3_correct:
            both_wrong.append(cid)
        else:
            both_right += 1

    return {
        "fraction": fraction,
        "both_right_count": both_right,
        "e2_right_e3_wrong": e2_right_e3_wrong,
        "e2_wrong_e3_right": e2_wrong_e3_right,
        "both_wrong": both_wrong,
        "n_e2_right_e3_wrong": len(e2_right_e3_wrong),
        "n_e2_wrong_e3_right": len(e2_wrong_e3_right),
        "n_both_wrong": len(both_wrong),
    }


# ---------------------------------------------------------------------------
# Comparison and summary
# ---------------------------------------------------------------------------

def print_e2_vs_e3(
    e2_metrics: Dict[str, float],
    e3_test: Dict[float, Dict[str, Any]],
    fractions: List[float],
) -> None:
    print("\n" + "=" * 86)
    print("E2 vs E3 COMPARISON — TEST SET (459 conversations)")
    print("  E2: prefix-aware AlephBERT, text only")
    print("  E3: prefix-aware AlephBERT + 26 pure behavioral features (prefix-sliced)")
    print("=" * 86)
    print(f"  {'Prefix':>8}  {'E2 Mac.F1':>10}  {'E3 Mac.F1':>10}  {'Δ Mac.F1':>10}  "
          f"{'E3 LI F1':>9}  {'E3 LI Prec':>11}  {'E3 LI Rec':>11}")
    print("  " + "-" * 74)

    comparison: Dict[str, Any] = {}
    for frac in fractions:
        frac_str = str(int(frac * 100))
        e2_macro = e2_metrics.get(frac_str, {}).get("macro_f1", float("nan"))
        e3m      = e3_test[frac]["metrics"]
        delta    = (e3m["macro_f1"] - e2_macro) * 100 if not np.isnan(e2_macro) else float("nan")
        sign     = "+" if delta >= 0 else ""
        print(f"  {int(frac*100):>7}%  {e2_macro:>10.4f}  {e3m['macro_f1']:>10.4f}  "
              f"{sign}{delta:>9.2f}pp  {e3m['f1']:>9.4f}  "
              f"{e3m['precision']:>11.4f}  {e3m['recall']:>11.4f}")
        comparison[frac_str] = {
            "e2_macro_f1": e2_macro,
            "e3_macro_f1": e3m["macro_f1"],
            "delta_pp": round((e3m["macro_f1"] - e2_macro) * 100, 3) if not np.isnan(e2_macro) else None,
            "e3_accuracy": e3m["accuracy"],
            "e3_f1": e3m["f1"],
            "e3_precision": e3m["precision"],
            "e3_recall": e3m["recall"],
        }

    print()
    print(f"  Prediction distribution at each prefix:")
    id_to_label = {}
    for frac in fractions:
        preds = e3_test[frac]["preds"]
        n = len(preds)
        # We know label_to_id: interested=0, losing_interest=1 (alphabetical)
        n_int = int((preds == 0).sum())
        n_li  = int((preds == 1).sum())
        print(f"    {int(frac*100):>3}%: interested={n_int/n*100:.1f}%  losing_interest={n_li/n*100:.1f}%")
    print("=" * 86)
    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3: prefix-aware AlephBERT + behavioral fusion.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def run(config_path: Path) -> None:
    config   = load_config(config_path)
    base_dir = config_path.resolve().parent
    device   = get_device()

    e3_cfg       = config.get("early_detection_e3", {})
    data_config  = config["data"]
    max_length   = int(config["tokenizer"]["max_length"])
    seed         = int(config.get("random_seed", 42))
    pos_class    = e3_cfg.get("positive_class", "losing_interest")
    train_fracs  = e3_cfg.get("training_fractions", FRACTIONS)
    eval_fracs   = e3_cfg.get("eval_fractions", FRACTIONS)
    sel_fracs    = e3_cfg.get("checkpoint_selection_fractions", [0.25, 0.50, 0.75])
    epochs       = int(e3_cfg.get("epochs", 5))
    bert_lr      = float(e3_cfg.get("bert_learning_rate", 1e-5))
    head_lr      = float(e3_cfg.get("head_learning_rate", 1e-3))
    batch_size   = int(e3_cfg.get("batch_size", 8))
    eval_bs      = int(e3_cfg.get("eval_batch_size", 32))
    wd           = float(e3_cfg.get("weight_decay", 0.01))
    warmup_r     = float(e3_cfg.get("warmup_ratio", 0.1))
    grad_clip    = float(e3_cfg.get("gradient_clip", 1.0))
    hidden_dim   = int(e3_cfg.get("head_hidden_dim", 256))
    dropout      = float(e3_cfg.get("head_dropout", 0.1))

    set_full_seed(seed)

    label_mapping = data_config["label_mapping"]
    label_to_id   = {lbl: i for i, lbl in enumerate(sorted(label_mapping.keys()))}
    feat_names    = pure_behavioral_feature_names()
    n_beh         = len(feat_names)

    output_dir = base_dir / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    protected = [
        "alephbert_baseline_v1", "alephbert_continued_ablation_v1",
        "early_detection_e1", "early_detection_e2",
        "fusion_alephbert_behavioral_v1", "behavioral_baseline_v1",
        "pure_behavioral_baseline_v1",
    ]
    for name in protected:
        if (base_dir / "outputs" / name) == output_dir:
            raise RuntimeError(f"Would overwrite protected output: {name}")

    print("=" * 64)
    print("Early Detection E3 — Prefix-Aware AlephBERT + Behavioral Fusion")
    print("=" * 64)

    # ---- Load corpus ----
    input_path = Path(data_config["input_path"])
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    with input_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    corpus = raw if isinstance(raw, list) else next(
        (raw[k] for k in ("data", "records", "conversations", "items") if k in raw), [raw]
    )
    id_field = data_config.get("id_field", "conversation_id")
    conversations_lookup: Dict[str, Dict[str, Any]] = {
        str(c.get(id_field, "")): c for c in corpus
    }

    # ---- Split IDs ----
    split_ids_path = base_dir / config.get("split", {}).get(
        "split_ids_output", "outputs/split_ids.json"
    )
    train_ids, val_ids, test_ids = load_split_ids(split_ids_path)
    train_convs = [conversations_lookup[cid] for cid in train_ids if cid in conversations_lookup]
    val_convs   = [conversations_lookup[cid] for cid in val_ids   if cid in conversations_lookup]
    test_convs  = [conversations_lookup[cid] for cid in test_ids  if cid in conversations_lookup]

    n_train = len(train_convs) * len(train_fracs)
    print(f"  Train: {len(train_convs)} convs × {len(train_fracs)} fracs = {n_train} examples")
    print(f"  Val  : {len(val_convs)} convs × {len(eval_fracs)} fracs (per-fraction eval)")
    print(f"  Test : {len(test_convs)} convs × {len(eval_fracs)} fracs (per-fraction eval)")
    print(f"  Behavioral features: {n_beh} pure non-lexical")
    print(f"  Concat dim: 768 + {n_beh} = {768 + n_beh}")
    print(f"  BERT LR: {bert_lr}  Head LR: {head_lr}  epochs: {epochs}")
    print(f"  Selection: avg macro F1 across {[int(f*100) for f in sel_fracs]}%")
    print(f"  Device: {device}  Seed: {seed}")
    print("=" * 64)

    # ---- Fit StandardScaler on training prefix behavioral features ----
    logger.info("Fitting StandardScaler on %d training prefix examples...", n_train)
    raw_to_binary: Dict[str, str] = {
        raw: mapped for mapped, raws in label_mapping.items() for raw in raws
    }
    label_field = data_config.get("label_field", "final_outcome")

    scaler_rows: List[List[float]] = []
    for conv in train_convs:
        if str(conv.get(label_field, "")) not in raw_to_binary:
            continue
        for frac in train_fracs:
            scaler_rows.append(extract_prefix_behavioral(conv, frac, feat_names, data_config))

    scaler = StandardScaler()
    scaler.fit(np.array(scaler_rows, dtype=np.float64))
    joblib.dump(scaler, output_dir / "behavioral_scaler.joblib")
    logger.info("Scaler fitted on %d examples.", len(scaler_rows))

    # ---- Load tokenizer + model ----
    ckpt_dir = base_dir / e3_cfg.get(
        "starting_checkpoint", "outputs/alephbert_continued_ablation_v1/best_model"
    )
    logger.info("Loading tokenizer from %s ...", ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    logger.info("Loading AlephBERT backbone from %s ...", ckpt_dir)
    base_clf = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=2, attn_implementation="eager",
    )
    model = FusionModel(base_clf.bert, n_behavioral=n_beh,
                        hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)

    n_bert_params = sum(p.numel() for p in model.bert.parameters())
    n_head_params = sum(p.numel() for p in model.head.parameters())
    logger.info("BERT params: %d  Head params: %d", n_bert_params, n_head_params)

    # ---- Build training dataset ----
    logger.info("Tokenising %d training examples...", n_train)
    train_ds = PrefixFusionDataset(
        train_convs, train_fracs, data_config, tokenizer, max_length,
        label_to_id, feat_names, scaler=scaler,
    )
    logger.info("Training dataset: %d examples.", len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # ---- Optimizer + scheduler (differential LR) ----
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": bert_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=wd)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_r)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"\nTotal steps: {total_steps}  Warmup: {warmup_steps}")

    # ---- Training loop ----
    best_sel_metric = -1.0
    best_epoch      = -1
    training_history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)
        train_loss = fusion_train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip)

        val_results = evaluate_e3_per_fraction(
            model, val_convs, eval_fracs, data_config, tokenizer,
            max_length, device, label_to_id, pos_class, feat_names, scaler, eval_bs,
        )
        sel = selection_metric(val_results, sel_fracs)
        is_best = sel > best_sel_metric

        record: Dict[str, Any] = {
            "epoch": epoch, "train_loss": train_loss,
            "val_selection_metric": sel,
            **{f"val_{int(f*100)}pct_{k}": v
               for f, res in val_results.items()
               for k, v in res["metrics"].items()},
        }
        training_history.append(record)

        frac_str = "  ".join(
            f"{int(f*100)}%: mF1={val_results[f]['metrics']['macro_f1']:.4f}"
            for f in eval_fracs
        )
        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} | sel={sel:.4f} | "
              f"{frac_str}  {'<-- best' if is_best else ''}")

        if is_best:
            best_sel_metric = sel
            best_epoch = epoch
            best_model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.head.state_dict(), best_model_dir / "head_state_dict.pt")
            torch.save(model.bert.state_dict(), best_model_dir / "bert_state_dict.pt")
            logger.info("Saved best model (epoch %d, sel=%.4f)", epoch, sel)

    print(f"\nBest epoch: {best_epoch}  Selection metric: {best_sel_metric:.4f}")

    # ---- Save training artifacts ----
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(training_history, fh, indent=2)

    exp_cfg_out = {
        "experiment": EXPERIMENT_NAME,
        "starting_checkpoint": str(ckpt_dir),
        "n_behavioral_features": n_beh,
        "behavioral_feature_names": feat_names,
        "concat_dim": 768 + n_beh,
        "training_fractions": train_fracs, "eval_fractions": eval_fracs,
        "checkpoint_selection_fractions": sel_fracs,
        "epochs": epochs, "best_epoch": best_epoch,
        "bert_learning_rate": bert_lr, "head_learning_rate": head_lr,
        "batch_size": batch_size, "weight_decay": wd,
        "warmup_ratio": warmup_r, "seed": seed, "max_length": max_length,
        "head_hidden_dim": hidden_dim, "head_dropout": dropout,
        "leakage_prevention": (
            "Behavioral features extracted from conversation[:prefix_length] only. "
            "StandardScaler fitted on training prefix features only."
        ),
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as fh:
        json.dump(exp_cfg_out, fh, indent=2)

    aug_stats = {
        "unique_train_conversations": len(train_convs),
        "unique_val_conversations": len(val_convs),
        "unique_test_conversations": len(test_convs),
        "training_fractions": train_fracs,
        "prefix_train_examples": n_train,
    }
    with (output_dir / "prefix_augmentation_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(aug_stats, fh, indent=2)

    # ---- Reload best checkpoint ----
    logger.info("Loading best checkpoint (epoch %d) ...", best_epoch)
    model.head.load_state_dict(
        torch.load(best_model_dir / "head_state_dict.pt", map_location=device, weights_only=True)
    )
    model.bert.load_state_dict(
        torch.load(best_model_dir / "bert_state_dict.pt", map_location=device, weights_only=True)
    )

    # ---- Final test evaluation (once per fraction) ----
    logger.info("Final test evaluation...")
    test_results = evaluate_e3_per_fraction(
        model, test_convs, eval_fracs, data_config, tokenizer,
        max_length, device, label_to_id, pos_class, feat_names, scaler, eval_bs,
    )

    id_to_lbl = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_lbl[i] for i in range(len(label_to_id))]

    print("\n" + "=" * 64)
    print("TEST SET RESULTS PER PREFIX (E3)")
    print("=" * 64)
    for frac in eval_fracs:
        m     = test_results[frac]["metrics"]
        preds = test_results[frac]["preds"]
        y_true = test_results[frac]["y_true"]
        n     = len(preds)
        n_int = int((preds == label_to_id.get("interested", 0)).sum())
        n_li  = int((preds == label_to_id.get("losing_interest", 1)).sum())
        print(f"\n  [{int(frac*100)}%] acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"f1={m['f1']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}")
        print(f"  Pred dist: interested={n_int/n*100:.1f}%  losing_interest={n_li/n*100:.1f}%")
        print(classification_report(y_true, preds, target_names=class_labels, zero_division=0))

        save_prefix_metrics(
            output_dir, frac, m,
            test_results[frac]["probs"], preds, y_true,
            test_results[frac]["record_ids"],
            label_to_id, conversations_lookup, data_config,
        )

    # ---- E2 vs E3 comparison ----
    e2_metrics: Dict[str, Dict] = {}
    for frac in eval_fracs:
        tag = f"{int(frac * 100):03d}pct"
        p = base_dir / "outputs" / "early_detection_e2" / f"test_metrics_{tag}.json"
        if p.exists():
            e2_metrics[str(int(frac * 100))] = json.loads(p.read_text())

    comparison = print_e2_vs_e3(e2_metrics, test_results, eval_fracs)
    with (output_dir / "e2_vs_e3_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    # ---- Error analysis per fraction ----
    e2_preds_dir = base_dir / "outputs" / "early_detection_e2"
    for frac in eval_fracs:
        tag = f"{int(frac * 100):03d}pct"
        e2_preds_path = e2_preds_dir / f"predictions_{tag}.csv"
        err = compute_error_analysis(
            e2_preds_path,
            test_results[frac]["preds"],
            test_results[frac]["y_true"],
            test_results[frac]["record_ids"],
            label_to_id,
            frac,
        )
        with (output_dir / f"error_analysis_{tag}.json").open("w", encoding="utf-8") as fh:
            json.dump(err, fh, indent=2)
        print(f"  [{int(frac*100)}%] E2→E3: "
              f"E2✓E3✗={err['n_e2_right_e3_wrong']}  "
              f"E2✗E3✓={err['n_e2_wrong_e3_right']}  "
              f"both✗={err['n_both_wrong']}")

    # ---- Summary file ----
    _write_summary(output_dir, best_epoch, best_sel_metric, test_results,
                   e2_metrics, eval_fracs, n_beh)
    print(f"\nAll artifacts saved to: {output_dir.resolve()}")


def _write_summary(
    output_dir: Path,
    best_epoch: int,
    best_sel_metric: float,
    test_results: Dict[float, Dict[str, Any]],
    e2_metrics: Dict[str, Dict],
    fractions: List[float],
    n_beh: int,
) -> None:
    lines = [
        "E3 Experiment Summary",
        "=" * 60,
        f"Best checkpoint epoch: {best_epoch}",
        f"Selection metric (avg macro F1 at 25/50/75%): {best_sel_metric:.4f}",
        f"Behavioral features: {n_beh} pure non-lexical features",
        f"Concat dim: 768 + {n_beh} = {768 + n_beh}",
        "",
        "Test Results (E3 vs E2):",
        f"{'Prefix':>8}  {'E2 Macro F1':>12}  {'E3 Macro F1':>12}  {'Delta':>8}",
        "-" * 48,
    ]
    for frac in fractions:
        frac_str = str(int(frac * 100))
        e3_mf1 = test_results[frac]["metrics"]["macro_f1"]
        e2_mf1 = e2_metrics.get(frac_str, {}).get("macro_f1", float("nan"))
        delta_str = f"{(e3_mf1 - e2_mf1)*100:+.2f}pp" if not np.isnan(e2_mf1) else "N/A"
        lines.append(f"{int(frac*100):>7}%  {e2_mf1:>12.4f}  {e3_mf1:>12.4f}  {delta_str:>8}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
