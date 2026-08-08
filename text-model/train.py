"""train.py

AlephBERT fine-tuning pipeline for Hebrew conversation classification.

Trains onlplab/alephbert-base for binary sequence classification
(interested vs. losing_interest), with:
  - deterministic stratified train / val / test split at conversation level
  - AdamW + linear warmup schedule
  - best-model checkpointing by validation macro F1
  - final evaluation on the held-out test set (once)
  - all artifacts saved to outputs/<experiment_name>/

Usage:
    python train.py --config config.json
    python train.py --config config.json --epochs 3
    python train.py --config config.json --experiment-name alephbert_v2
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
from transformers import get_linear_schedule_with_warmup

from data_loader import (
    ConfigError,
    create_dataloader,
    create_dataset,
    get_device,
    load_config,
    load_corpus_records,
    load_model,
    load_tokenizer,
    set_random_seed,
)
from splitter import (
    SplitError,
    load_split_ids,
    print_split_statistics,
    save_split_ids,
    stratified_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune AlephBERT on Hebrew conversation classification."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_full_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        extra = {}
        if "token_type_ids" in batch:
            extra["token_type_ids"] = batch["token_type_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **extra,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            logger.info(
                "  batch %d/%d  loss=%.4f",
                batch_idx + 1, n_batches, total_loss / (batch_idx + 1),
            )

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    label_to_id: Dict[str, int],
    positive_class: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    """Evaluate on a dataloader. Returns (metrics, probabilities, predictions, record_ids)."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_record_ids: List[str] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            extra = {}
            if "token_type_ids" in batch:
                extra["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **extra,
            )
            total_loss += outputs.loss.item()
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())
            all_record_ids.extend(list(batch["record_id"]))

    logits = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    y_true = labels_tensor.numpy()

    positive_id = label_to_id[positive_class]

    metrics: Dict[str, float] = {
        "loss": total_loss / len(dataloader),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, pos_label=positive_id, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, preds, pos_label=positive_id, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, preds, pos_label=positive_id, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }
    return metrics, probs, preds, all_record_ids


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_training_artifacts(
    output_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    config: Dict[str, Any],
    split_stats: Dict[str, Any],
    training_history: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    with (output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    with (output_dir / "split_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(split_stats, fh, ensure_ascii=False, indent=2)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(training_history, fh, ensure_ascii=False, indent=2)
    logger.info("Saved config, split stats, and training history to %s", output_dir)


def save_test_results(
    output_dir: Path,
    test_metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    record_ids: List[str],
    label_to_id: Dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, ensure_ascii=False, indent=2)

    report = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    cm_rows = [[""] + [f"pred_{c}" for c in class_labels]]
    for i, c in enumerate(class_labels):
        cm_rows.append([f"actual_{c}"] + [str(v) for v in cm[i]])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(cm_rows)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = (
            ["conversation_id", "actual_label", "predicted_label"]
            + [f"probability_{c}" for c in class_labels]
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, record_id in enumerate(record_ids):
            row: Dict[str, Any] = {
                "conversation_id": record_id,
                "actual_label": id_to_label[int(y_true[i])],
                "predicted_label": id_to_label[int(y_pred[i])],
            }
            for j, c in enumerate(class_labels):
                row[f"probability_{c}"] = f"{probs[i, j]:.6f}"
            writer.writerow(row)

    logger.info("Saved test results to %s", output_dir)


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------

def run(config_path: Path, epochs_override: Optional[int] = None, experiment_name_override: Optional[str] = None) -> None:
    config = load_config(config_path)
    base_dir = config_path.resolve().parent

    experiment_name = experiment_name_override or config.get("experiment_name", "alephbert_baseline_v1")
    output_dir = base_dir / "outputs" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("random_seed", 42))
    set_full_seed(seed)

    training_cfg = config.get("training", {})
    split_cfg = config.get("split", {})
    epochs = epochs_override if epochs_override is not None else int(training_cfg.get("epochs", 5))
    lr = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    grad_clip = float(training_cfg.get("gradient_clip", 1.0))
    positive_class = training_cfg.get("positive_class", "losing_interest")
    save_best_metric = training_cfg.get("save_best_metric", "macro_f1")

    device = get_device()

    print("=" * 64)
    print("AlephBERT Training Pipeline")
    print("=" * 64)
    print(f"  Experiment   : {experiment_name}")
    print(f"  Model        : {config['model_name']}")
    print(f"  Dataset      : {config['data']['input_path']}")
    print(f"  Device       : {device}")
    print(f"  Seed         : {seed}")
    print(f"  Epochs       : {epochs}")
    print(f"  LR           : {lr}")
    print(f"  Positive class (for binary F1): {positive_class}")
    print(f"  Best-model metric: {save_best_metric}")
    print(f"  max_length   : {config.get('tokenizer', {}).get('max_length', 512)}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 64)

    # ---- Load corpus records ----
    logger.info("Loading corpus records...")
    records = load_corpus_records(config, base_dir=base_dir)
    logger.info("Loaded %d raw records.", len(records))

    # ---- Stratified split ----
    split_ids_output = split_cfg.get("split_ids_output", "outputs/split_ids.json")
    split_ids_path = base_dir / split_ids_output

    train_ids, val_ids, test_ids = stratified_split(
        records=records,
        label_mapping=config["data"]["label_mapping"],
        id_field=config["data"].get("id_field", "conversation_id"),
        label_field=config["data"].get("label_field", "final_outcome"),
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(split_cfg.get("random_seed", seed)),
    )

    print("\nSplit statistics:")
    print_split_statistics(
        records, train_ids, val_ids, test_ids,
        config["data"]["label_mapping"],
        id_field=config["data"].get("id_field", "conversation_id"),
        label_field=config["data"].get("label_field", "final_outcome"),
    )
    print()

    save_split_ids(
        train_ids, val_ids, test_ids,
        output_path=split_ids_path,
        config_snapshot={
            "train_ratio": split_cfg.get("train_ratio", 0.70),
            "val_ratio": split_cfg.get("val_ratio", 0.15),
            "test_ratio": split_cfg.get("test_ratio", 0.15),
            "seed": seed,
            "model": config["model_name"],
        },
    )

    split_stats = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "total": len(train_ids) + len(val_ids) + len(test_ids),
    }

    # ---- Tokenizer ----
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config)

    # ---- Datasets ----
    logger.info("Building datasets...")
    train_dataset = create_dataset(config, tokenizer, base_dir=base_dir, subset_ids=set(train_ids))
    val_dataset = create_dataset(config, tokenizer, base_dir=base_dir, subset_ids=set(val_ids))
    test_dataset = create_dataset(config, tokenizer, base_dir=base_dir, subset_ids=set(test_ids))

    logger.info("Train: %d | Val: %d | Test: %d", len(train_dataset), len(val_dataset), len(test_dataset))

    label_to_id = train_dataset.label_to_id
    id_to_label = train_dataset.id_to_label

    if positive_class not in label_to_id:
        raise ConfigError(
            f"positive_class='{positive_class}' not found in label_to_id={label_to_id}. "
            f"Check config['training']['positive_class']."
        )

    # ---- DataLoaders ----
    train_loader = create_dataloader(train_dataset, config, split="train")
    val_loader = create_dataloader(val_dataset, config, split="val")
    test_loader = create_dataloader(test_dataset, config, split="test")

    # ---- Model ----
    logger.info("Loading model...")
    model = load_model(config, label_to_id, id_to_label)
    model.to(device)

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Total training steps: {total_steps}  |  Warmup steps: {warmup_steps}")

    # ---- Training loop ----
    best_metric_value = -1.0
    best_epoch = -1
    training_history: List[Dict[str, Any]] = []
    best_model_dir = output_dir / "best_model"

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, label_to_id, positive_class)

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        training_history.append(epoch_record)

        metric_value = val_metrics[save_best_metric]
        is_best = metric_value > best_metric_value

        print(
            f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | f1={val_metrics['f1']:.4f}  "
            f"{'<-- best' if is_best else ''}"
        )

        if is_best:
            best_metric_value = metric_value
            best_epoch = epoch
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            logger.info("Saved best model (epoch %d, %s=%.4f)", epoch, save_best_metric, metric_value)

    print(f"\nBest epoch: {best_epoch}  |  Best val {save_best_metric}: {best_metric_value:.4f}")

    # ---- Save training artifacts ----
    save_training_artifacts(output_dir, model, tokenizer, config, split_stats, training_history)

    # ---- Load best checkpoint for test evaluation ----
    logger.info("Loading best checkpoint from %s...", best_model_dir)
    from transformers import AutoModelForSequenceClassification
    best_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))
    best_model.to(device)

    # ---- Final test evaluation (once) ----
    logger.info("Evaluating on held-out test set...")
    test_metrics, test_probs, test_preds, test_record_ids = evaluate(
        best_model, test_loader, device, label_to_id, positive_class
    )

    print("\n" + "=" * 64)
    print("TEST SET RESULTS (best checkpoint, epoch %d)" % best_epoch)
    print("=" * 64)
    for key, value in test_metrics.items():
        print(f"  {key:20s}: {value:.4f}")
    print()

    label_array = test_dataset.labels.numpy()
    id_map = {rid: i for i, rid in enumerate(test_dataset.record_ids)}
    y_true_ordered = np.array([label_array[id_map[rid]] for rid in test_record_ids])

    class_names = [id_to_label[i] for i in range(len(label_to_id))]
    report = classification_report(y_true_ordered, test_preds, target_names=class_names, zero_division=0)
    print(report)

    # Augment test_metrics with additional aggregates
    test_metrics["best_epoch"] = best_epoch
    test_metrics["experiment_name"] = experiment_name

    save_test_results(
        output_dir=output_dir,
        test_metrics=test_metrics,
        y_true=y_true_ordered,
        y_pred=test_preds,
        probs=test_probs,
        record_ids=test_record_ids,
        label_to_id=label_to_id,
    )

    print(f"\nAll artifacts saved to: {output_dir.resolve()}")
    print("Training complete.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.config, epochs_override=args.epochs, experiment_name_override=args.experiment_name)
    except (ConfigError, SplitError) as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
