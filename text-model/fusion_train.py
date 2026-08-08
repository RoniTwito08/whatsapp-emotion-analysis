"""fusion_train.py

Model D — AlephBERT + Pure Behavioral Features Fusion.

Research question:
    Does non-lexical conversational structure (26 pure behavioral features)
    add predictive information beyond the Hebrew text representation already
    learned by AlephBERT?

Architecture:
    concat(pooler_output[768] + standardized_behavioral[26])
    → Linear(794, 256) → ReLU → Dropout(0.1) → Linear(256, 2)

    The text representation is bert.pooler_output — tanh(linear([CLS])),
    which is identical to what the original AlephBERT classifier used.
    We replace Linear(768→2) with the fusion MLP head.

Two strategies:
    D1 (frozen)   : Freeze AlephBERT backbone, train only the fusion head.
                    Answers: does behavioral structure complement a fixed
                    AlephBERT representation?
    D2 (finetuned): Continue fine-tuning AlephBERT jointly with the head.
                    Answers: can the language model and structural features
                    jointly improve over AlephBERT alone?

Important:
    - Only 26 PURE BEHAVIORAL (non-lexical) features are used.
    - StandardScaler is fitted on TRAIN split only.
    - Best checkpoint is selected by validation macro F1.
    - Test set is evaluated exactly once.

Usage:
    python fusion_train.py --config config.json
    python fusion_train.py --config config.json --strategy frozen
    python fusion_train.py --config config.json --strategy finetuned
    python fusion_train.py --config config.json --strategy both
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
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

from behavioral_features_design import (
    FORBIDDEN_FIELDS,
    LEXICAL_FEATURE_NAMES,
    extract_behavioral_features,
    pure_behavioral_feature_names,
)
from data_loader import get_device, load_config
from dataset import HebrewConversationDataset
from splitter import load_split_ids

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("fusion_train")

# ---------------------------------------------------------------------------
# Known baselines for comparison table
# ---------------------------------------------------------------------------
KNOWN_RESULTS = {
    "pure_behavioral_rf": {
        "accuracy": 0.6536, "macro_f1": 0.6507, "f1": 0.6187,
    },
    "behavioral_lexical_rf": {
        "accuracy": 0.8671, "macro_f1": 0.8669, "f1": 0.8617,
    },
    "alephbert": {
        "accuracy": 0.9695, "macro_f1": 0.9695, "f1": 0.9690,
    },
}


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """AlephBERT backbone + pure behavioral MLP fusion head.

    Text representation: bert.pooler_output (768-dim tanh(linear([CLS]))),
    identical to what the original AlephBERT sequence classifier used.
    """

    def __init__(
        self,
        bert_backbone: nn.Module,
        n_behavioral: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = bert_backbone
        bert_dim = bert_backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(bert_dim + n_behavioral, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self._bert_dim = bert_dim
        self._n_behavioral = n_behavioral

    @property
    def concat_dim(self) -> int:
        return self._bert_dim + self._n_behavioral

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        behavioral_features: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = bert_out.pooler_output               # (B, 768)
        x = torch.cat([pooled, behavioral_features], dim=-1)  # (B, 794)
        return self.head(x)                            # (B, 2)

    def freeze_bert(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Fusion dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """Wraps HebrewConversationDataset, adding a pre-built behavioral tensor.

    Behavioral features must already be standardized (scaler.transform called
    externally). The tensor must be aligned to text_dataset.record_ids order.
    """

    def __init__(
        self,
        text_dataset: HebrewConversationDataset,
        behavioral_tensor: torch.Tensor,
    ) -> None:
        if len(text_dataset) != len(behavioral_tensor):
            raise ValueError(
                f"Length mismatch: text_dataset has {len(text_dataset)} items "
                f"but behavioral_tensor has {len(behavioral_tensor)} rows."
            )
        self.text = text_dataset
        self.beh = behavioral_tensor          # (N, n_behavioral), float32

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.text[i]                   # dict from HebrewConversationDataset
        item["behavioral_features"] = self.beh[i]
        return item


# ---------------------------------------------------------------------------
# Building behavioral tensors
# ---------------------------------------------------------------------------

def build_behavioral_matrix(
    records: List[Dict[str, Any]],
    record_ids: List[str],
    feat_names: List[str],
    id_field: str = "conversation_id",
) -> np.ndarray:
    """Return float64 array (N, F) for the given record_ids in that order."""
    id_to_record: Dict[str, Dict[str, Any]] = {
        str(r.get(id_field, "")): r for r in records
    }
    rows: List[List[float]] = []
    for rid in record_ids:
        rec = id_to_record.get(rid, {})
        feats = extract_behavioral_features(rec) if rec else {}
        rows.append([float(feats.get(f, 0.0)) for f in feat_names])
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_epoch(
    model: FusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        beh            = batch["behavioral_features"].to(device)
        labels         = batch["labels"].to(device)
        tt_ids = batch.get("token_type_ids")
        if tt_ids is not None:
            tt_ids = tt_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, beh, token_type_ids=tt_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (batch_idx + 1) % max(1, n_batches // 4) == 0:
            logger.info("  batch %d/%d  loss=%.4f", batch_idx + 1, n_batches,
                        total_loss / (batch_idx + 1))
    return total_loss / n_batches


def evaluate(
    model: FusionModel,
    dataloader: DataLoader,
    device: torch.device,
    label_to_id: Dict[str, int],
    positive_class: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_rids:   List[str] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            beh            = batch["behavioral_features"].to(device)
            labels         = batch["labels"].to(device)
            tt_ids = batch.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(device)

            logits = model(input_ids, attention_mask, beh, token_type_ids=tt_ids)
            total_loss += criterion(logits, labels).item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_rids.extend(list(batch["record_id"]))

    logits  = torch.cat(all_logits)
    labels  = torch.cat(all_labels)
    probs   = torch.softmax(logits, dim=-1).numpy()
    preds   = logits.argmax(dim=-1).numpy()
    y_true  = labels.numpy()
    pos_id  = label_to_id[positive_class]

    metrics: Dict[str, float] = {
        "loss":        total_loss / len(dataloader),
        "accuracy":    float(accuracy_score(y_true, preds)),
        "precision":   float(precision_score(y_true, preds, pos_label=pos_id, average="binary", zero_division=0)),
        "recall":      float(recall_score(y_true, preds, pos_label=pos_id, average="binary", zero_division=0)),
        "f1":          float(f1_score(y_true, preds, pos_label=pos_id, average="binary", zero_division=0)),
        "macro_f1":    float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }
    return metrics, probs, preds, all_rids


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(
    output_dir: Path,
    strategy_name: str,
    model: FusionModel,
    scaler: StandardScaler,
    feat_names: List[str],
    training_history: List[Dict[str, Any]],
    test_metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    record_ids: List[str],
    label_to_id: Dict[str, int],
    model_config: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]

    # Model state dict
    best_dir = output_dir / "best_model"
    best_dir.mkdir(exist_ok=True)
    torch.save(model.head.state_dict(), best_dir / "head_state_dict.pt")
    if not model_config.get("freeze_bert", True):
        torch.save(model.bert.state_dict(), best_dir / "bert_state_dict.pt")

    # Scaler
    joblib.dump(scaler, output_dir / "behavioral_scaler.joblib")

    # Feature names
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
        json.dump(feat_names, fh, indent=2)

    # Model config
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as fh:
        json.dump(model_config, fh, ensure_ascii=False, indent=2)

    # Training history
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(training_history, fh, indent=2)

    # Test metrics
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, indent=2)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    cm_rows = [[""] + [f"pred_{c}" for c in class_labels]]
    for i, c in enumerate(class_labels):
        cm_rows.append([f"actual_{c}"] + [str(v) for v in cm[i]])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(cm_rows)

    # Test predictions
    fieldnames = (["conversation_id", "actual_label", "predicted_label"]
                  + [f"probability_{c}" for c in class_labels])
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, rid in enumerate(record_ids):
            row: Dict[str, Any] = {
                "conversation_id": rid,
                "actual_label": id_to_label[int(y_true[i])],
                "predicted_label": id_to_label[int(y_pred[i])],
            }
            for j, c in enumerate(class_labels):
                row[f"probability_{c}"] = f"{probs[i, j]:.6f}"
            writer.writerow(row)

    logger.info("Saved all artifacts to %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# Single strategy training + evaluation
# ---------------------------------------------------------------------------

def run_strategy(
    strategy: str,
    config: Dict[str, Any],
    base_dir: Path,
    records: List[Dict[str, Any]],
    id_to_label_map: Dict[str, str],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    scaler: StandardScaler,
    feat_names: List[str],
    tokenizer: Any,
    device: torch.device,
    label_to_id: Dict[str, int],
    experiment_base_dir: Path,
) -> Optional[Dict[str, float]]:
    """Run one strategy (frozen or finetuned). Returns test_metrics dict."""
    fusion_cfg = config["fusion"]
    strat_cfg  = fusion_cfg[strategy]
    is_frozen  = (strategy == "frozen")

    positive_class = fusion_cfg.get("positive_class", "losing_interest")
    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    output_dir = experiment_base_dir / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs     = int(strat_cfg["epochs"])
    lr         = float(strat_cfg["learning_rate"])
    head_lr    = float(strat_cfg.get("head_learning_rate", lr))
    batch_size = int(strat_cfg["batch_size"])
    wd         = float(strat_cfg["weight_decay"])
    warmup_r   = float(strat_cfg["warmup_ratio"])
    grad_clip  = float(strat_cfg["gradient_clip"])
    hidden_dim = int(fusion_cfg.get("head_hidden_dim", 256))
    dropout    = float(fusion_cfg.get("head_dropout", 0.1))

    print(f"\n{'='*64}")
    print(f"Strategy: {strategy.upper()}")
    print(f"  AlephBERT: {'FROZEN' if is_frozen else 'FINE-TUNED'}")
    print(f"  Epochs: {epochs}  LR(backbone): {lr}  LR(head): {head_lr}")
    print(f"  Batch: {batch_size}  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'='*64}")

    # ---- Load AlephBERT checkpoint ----
    checkpoint_dir = base_dir / fusion_cfg["bert_checkpoint"]
    logger.info("Loading AlephBERT from %s ...", checkpoint_dir)
    base_clf = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), num_labels=2, ignore_mismatched_sizes=True,
        attn_implementation="eager",   # MPS does not support dropout in SDPA
    )
    backbone = base_clf.bert   # BertModel (768-dim pooler_output)

    # ---- Build fusion model ----
    n_beh = len(feat_names)
    model = FusionModel(backbone, n_behavioral=n_beh, hidden_dim=hidden_dim, dropout=dropout)
    if is_frozen:
        model.freeze_bert()
        frozen_params = sum(p.numel() for p in model.bert.parameters())
        trainable_params = sum(p.numel() for p in model.head.parameters())
        logger.info("BERT frozen (%d params). Training only fusion head (%d params).",
                    frozen_params, trainable_params)
    else:
        n_total = sum(p.numel() for p in model.parameters())
        logger.info("All params trainable (%d total).", n_total)
    model.to(device)

    # ---- Build datasets (tokenized text + behavioral) ----
    def _make_fusion_dataset(split_ids: List[str]) -> FusionDataset:
        text_ds = HebrewConversationDataset(
            config, tokenizer, base_dir=base_dir, subset_ids=set(split_ids)
        )
        beh_mat = build_behavioral_matrix(
            records, text_ds.record_ids, feat_names,
            id_field=config["data"].get("id_field", "conversation_id"),
        )
        beh_scaled = scaler.transform(beh_mat).astype(np.float32)
        beh_tensor = torch.tensor(beh_scaled, dtype=torch.float32)
        return FusionDataset(text_ds, beh_tensor)

    logger.info("Building datasets...")
    train_ds = _make_fusion_dataset(train_ids)
    val_ds   = _make_fusion_dataset(val_ids)
    test_ds  = _make_fusion_dataset(test_ids)
    logger.info("Dataset sizes: train=%d  val=%d  test=%d",
                len(train_ds), len(val_ds), len(test_ds))

    dataloader_cfg = config.get("dataloader", {})
    n_workers = int(dataloader_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=n_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=n_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False, num_workers=n_workers)

    # ---- Optimizer + scheduler ----
    if is_frozen:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=wd)
    else:
        # Differential learning rates
        optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ], weight_decay=wd)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_r)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")

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
            torch.save(model.head.state_dict(), best_model_dir / "head_state_dict.pt")
            if not is_frozen:
                torch.save(model.bert.state_dict(), best_model_dir / "bert_state_dict.pt")
            logger.info("Saved best model (epoch %d, macro_f1=%.4f)", epoch, best_val_macro_f1)

    print(f"\nBest epoch: {best_epoch}  Val macro_f1: {best_val_macro_f1:.4f}")

    # ---- Reload best checkpoint ----
    model.head.load_state_dict(torch.load(best_model_dir / "head_state_dict.pt",
                                          map_location=device, weights_only=True))
    if not is_frozen:
        model.bert.load_state_dict(torch.load(best_model_dir / "bert_state_dict.pt",
                                              map_location=device, weights_only=True))

    # ---- Final test evaluation (once) ----
    logger.info("Evaluating on held-out test set...")
    test_metrics, test_probs, test_preds, test_rids = evaluate(
        model, test_loader, device, label_to_id, positive_class
    )
    test_metrics["best_epoch"]       = best_epoch
    test_metrics["strategy"]         = strategy
    test_metrics["freeze_bert"]      = is_frozen
    test_metrics["best_val_macro_f1"] = best_val_macro_f1

    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]

    print(f"\n{'='*64}")
    print(f"TEST SET RESULTS ({strategy.upper()})")
    print(f"{'='*64}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
    print()
    # Recover y_true in the same order evaluate() iterated the dataloader
    id_map = {rid: i for i, rid in enumerate(test_ds.text.record_ids)}
    y_true_ordered = test_ds.text.labels.numpy()[[id_map[rid] for rid in test_rids]]

    model_config_record = {
        "strategy": strategy,
        "freeze_bert": is_frozen,
        "bert_checkpoint": str(checkpoint_dir),
        "n_behavioral_features": n_beh,
        "behavioral_feature_names": feat_names,
        "concat_dim": model.concat_dim,
        "head_hidden_dim": hidden_dim,
        "head_dropout": dropout,
        "epochs": epochs, "best_epoch": best_epoch,
        "learning_rate": lr, "head_learning_rate": head_lr,
        "batch_size": batch_size, "weight_decay": wd,
    }

    save_artifacts(
        output_dir, strategy, model, scaler, feat_names,
        training_history, test_metrics, y_true_ordered, test_preds,
        test_probs, test_rids, label_to_id, model_config_record,
    )

    print(classification_report(y_true_ordered, test_preds, target_names=class_labels, zero_division=0))

    return test_metrics


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: Dict[str, Optional[Dict[str, float]]]) -> None:
    pos = "losing_interest"
    print("\n" + "=" * 82)
    print("FULL MODEL COMPARISON — TEST SET (459 conversations, seed=42)")
    print("=" * 82)
    cols = ["Pure BehRF", "Beh+Lex RF", "AlephBERT", "Fusion D1", "Fusion D2"]
    keys = ["pure_behavioral_rf", "behavioral_lexical_rf", "alephbert", "d1", "d2"]
    print(f"{'Metric':<28}" + "".join(f"{c:>14}" for c in cols))
    print("-" * 82)

    all_results: Dict[str, Optional[Dict]] = {
        "pure_behavioral_rf": KNOWN_RESULTS["pure_behavioral_rf"],
        "behavioral_lexical_rf": KNOWN_RESULTS["behavioral_lexical_rf"],
        "alephbert": KNOWN_RESULTS["alephbert"],
        "d1": results.get("frozen"),
        "d2": results.get("finetuned"),
    }

    for metric, k_map in [
        ("accuracy",              ("accuracy",  "accuracy",  "accuracy",  "accuracy",  "accuracy")),
        ("macro F1",              ("macro_f1",  "macro_f1",  "macro_f1",  "macro_f1",  "macro_f1")),
        (f"{pos} F1",             ("f1",        "f1",        "f1",        "f1",        "f1")),
        (f"{pos} precision",      ("precision", "precision", "precision", "precision", "precision")),  # not stored in known, skip
        (f"{pos} recall",         ("recall",    "recall",    "recall",    "recall",    "recall")),
    ]:
        row_vals = []
        for key_name, metric_key in zip(keys, k_map):
            r = all_results.get(key_name)
            val = r.get(metric_key, float("nan")) if r else float("nan")
            row_vals.append(val)
        print(f"  {metric:<26}" + "".join(
            f"  {v:>10.4f}  " if not np.isnan(v) else f"  {'N/A':>10}  " for v in row_vals
        ))

    print("-" * 82)

    # Delta vs AlephBERT
    ab_macro = KNOWN_RESULTS["alephbert"]["macro_f1"]
    print(f"\n  Delta macro F1 vs AlephBERT:")
    for label, key in [("Fusion D1", "d1"), ("Fusion D2", "d2")]:
        r = all_results.get(key)
        if r:
            delta = r["macro_f1"] - ab_macro
            sign = "+" if delta >= 0 else ""
            print(f"    {label}: {sign}{delta*100:.2f}pp  "
                  f"({'improvement' if delta > 0 else 'degradation' if delta < 0 else 'no change'})")
        else:
            print(f"    {label}: not run")

    print()
    print("  Representations:")
    print("    Pure BehRF   : 26 structural features, no text")
    print("    Beh+Lex RF   : 43 features (structural + regex signals)")
    print("    AlephBERT    : customer text only, max_length=512")
    print("    Fusion D1    : AlephBERT(frozen) + 26 pure behavioral")
    print("    Fusion D2    : AlephBERT(fine-tuned) + 26 pure behavioral")
    print()
    print("  CAUTION: All data is synthetic. Results may not generalise.")
    print("=" * 82)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model D: AlephBERT + Pure Behavioral Fusion.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strategy", choices=["frozen", "finetuned", "both"], default="both")
    return parser.parse_args(argv)


def run(config_path: Path, strategy: str = "both") -> None:
    config   = load_config(config_path)
    base_dir = config_path.resolve().parent
    device   = get_device()
    seed     = int(config.get("random_seed", 42))
    set_seed(seed)

    fusion_cfg = config.get("fusion", {})
    positive_class = fusion_cfg.get("positive_class", "losing_interest")

    label_mapping = config["data"]["label_mapping"]
    label_to_id   = {lbl: i for i, lbl in enumerate(sorted(label_mapping.keys()))}
    feat_names    = pure_behavioral_feature_names()

    experiment_base_dir = base_dir / "outputs" / "fusion_alephbert_behavioral_v1"

    print("=" * 64)
    print("Fusion Model D — AlephBERT + Pure Behavioral Features")
    print("=" * 64)
    print(f"  Device          : {device}")
    print(f"  Seed            : {seed}")
    print(f"  Behavioral feats: {len(feat_names)} pure structural (no lexical)")
    print(f"  Concat dim      : 768 + {len(feat_names)} = {768 + len(feat_names)}")
    print(f"  Labels          : {label_to_id}")
    print(f"  Positive class  : {positive_class}")
    print(f"  Strategy        : {strategy}")
    print(f"  Output base     : {experiment_base_dir}")
    print("=" * 64)

    # ---- Load data ----
    from behavioral_baseline import load_data
    records, id_to_label_map = load_data(config_path, config)

    split_ids_path = base_dir / config.get("split", {}).get("split_ids_output", "outputs/split_ids.json")
    train_ids, val_ids, test_ids = load_split_ids(split_ids_path)
    assert not (set(train_ids) & set(val_ids)),  "train/val overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap"
    assert not (set(val_ids)   & set(test_ids)), "val/test overlap"
    print(f"\nSplit: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ---- Fit StandardScaler on TRAIN ONLY ----
    logger.info("Fitting StandardScaler on training set...")
    id_field = config["data"].get("id_field", "conversation_id")
    id_to_record: Dict[str, Dict[str, Any]] = {
        str(r.get(id_field, "")): r for r in records
    }
    X_train_raw = build_behavioral_matrix(records, train_ids, feat_names, id_field)
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    logger.info("Scaler fitted on %d train conversations.", len(train_ids))

    # Save scaler in the experiment base directory (shared across strategies)
    experiment_base_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, experiment_base_dir / "behavioral_scaler.joblib")
    with (experiment_base_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
        json.dump(feat_names, fh, indent=2)
    with (experiment_base_dir / "split_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump({"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}, fh, indent=2)

    # ---- Load tokenizer ----
    checkpoint_dir = base_dir / fusion_cfg["bert_checkpoint"]
    logger.info("Loading tokenizer from %s ...", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    # ---- Run strategies ----
    strategies_to_run: List[str] = (
        ["frozen", "finetuned"] if strategy == "both" else [strategy]
    )

    results: Dict[str, Optional[Dict[str, float]]] = {"frozen": None, "finetuned": None}
    for strat in strategies_to_run:
        result = run_strategy(
            strategy=strat,
            config=config,
            base_dir=base_dir,
            records=records,
            id_to_label_map=id_to_label_map,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            scaler=scaler,
            feat_names=feat_names,
            tokenizer=tokenizer,
            device=device,
            label_to_id=label_to_id,
            experiment_base_dir=experiment_base_dir,
        )
        results[strat] = result

    # ---- Comparison table ----
    print_comparison_table(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.config, strategy=args.strategy)
    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
