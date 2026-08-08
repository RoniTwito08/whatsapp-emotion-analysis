"""early_detection_e1.py

Experiment E1 — Early Detection via Prefix Evaluation (no retraining).

Research question:
    How early in a Hebrew WhatsApp sales conversation can the best trained
    model reliably predict whether the customer will remain interested or
    lose interest?

Approach:
    Take the frozen best text-only checkpoint (alephbert_continued_ablation_v1)
    and evaluate it on four prefixes of every held-out TEST conversation:
        25%  50%  75%  100%

No model is retrained. This is pure inference.

Prefix construction rule:
    For a conversation with N total messages (ALL roles, chronological):
        prefix_length = max(1, ceil(N * fraction))
        prefix_msgs   = messages[:prefix_length]          # first prefix_length messages
        customer_text = join(m.text for m in prefix_msgs  # filter to included_roles
                             if m.role in included_roles)
        text_input    = customer_text joined with message_separator

    Conversation progress is defined over ALL messages (customer + business),
    not just customer messages, because that reflects real chronological progress.

    If the prefix contains zero customer messages (rare; does not occur in this
    dataset at any fraction), the fallback is an empty string "". The tokenizer
    encodes this as [CLS][SEP] + padding, which is a valid — if uninformative —
    model input. The conversation is never removed from evaluation so that all
    four prefix levels always evaluate the same 459 test conversations.

Consistency requirement:
    At 100%, this procedure produces exactly the same text as
    dataset.HebrewConversationDataset for the same conversation. Verified
    by checking that 100% prefix results match the saved ablation test metrics.

Usage:
    python early_detection_e1.py --config config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import get_device, load_config
from splitter import load_split_ids

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("early_detection")

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
EXPERIMENT_NAME = "early_detection_e1"
INFERENCE_BATCH_SIZE = 32

# Reference values from the ablation (expected 100% prefix result)
ABLATION_TEST_MACRO_F1 = 0.9804
ABLATION_TEST_ACCURACY = 0.9804


# ---------------------------------------------------------------------------
# Prefix construction
# ---------------------------------------------------------------------------

def prefix_length(total_messages: int, fraction: float) -> int:
    """Return the number of messages to include for a given fraction.

    Rule: max(1, ceil(total_messages * fraction)).
    Applies to ALL messages (customer + business) to represent real
    chronological progress of the conversation.
    """
    return max(1, math.ceil(total_messages * fraction))


def build_prefix_text(
    conversation: Dict[str, Any],
    fraction: float,
    data_config: Dict[str, Any],
) -> Tuple[str, int, int]:
    """Extract the text input for a given prefix fraction.

    Args:
        conversation: Raw conversation dict from the corpus.
        fraction: Fraction of total messages to include (0 < fraction <= 1).
        data_config: config['data'] section.

    Returns:
        Tuple of:
            text             — joined customer message text (may be "" if no customer msgs)
            n_total_in_prefix — total messages in this prefix
            n_customer_in_prefix — customer messages in this prefix
    """
    messages_field  = data_config.get("messages_field", "messages")
    role_field      = data_config.get("role_field", "role")
    text_field      = data_config.get("text_field", "text")
    included_roles  = data_config.get("included_roles", ["customer"])
    separator       = data_config.get("message_separator", " [SEP] ")

    all_msgs = conversation.get(messages_field, [])
    plen = prefix_length(len(all_msgs), fraction)
    prefix_msgs = all_msgs[:plen]

    customer_texts: List[str] = []
    for msg in prefix_msgs:
        if msg.get(role_field) in included_roles:
            t = msg.get(text_field, "")
            if t:
                customer_texts.append(str(t))

    text = separator.join(customer_texts)
    return text, plen, len(customer_texts)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def infer_prefix(
    model: Any,
    tokenizer: Any,
    conversations: List[Dict[str, Any]],
    fraction: float,
    data_config: Dict[str, Any],
    label_to_id: Dict[str, int],
    positive_class: str,
    device: torch.device,
    max_length: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Run inference on all conversations at a given prefix fraction.

    Returns:
        metrics        — dict of accuracy, precision, recall, f1, macro_f1, weighted_f1
        probs          — (N, 2) probability array
        preds          — (N,) predicted label IDs
        y_true         — (N,) true label IDs
        conv_ids       — list of conversation IDs in evaluation order
        prefix_stats   — per-fraction statistics
    """
    model.eval()

    label_mapping = data_config.get("label_mapping", {})
    raw_to_binary: Dict[str, str] = {}
    for mapped, raws in label_mapping.items():
        for raw in raws:
            raw_to_binary[raw] = mapped

    id_field    = data_config.get("id_field", "conversation_id")
    label_field = data_config.get("label_field", "final_outcome")

    texts: List[str] = []
    true_labels: List[int] = []
    conv_ids: List[str] = []
    n_total_list: List[int] = []
    n_customer_list: List[int] = []
    n_zero_customer = 0

    for conv in conversations:
        raw_label = str(conv.get(label_field, ""))
        if raw_label not in raw_to_binary:
            continue
        binary_label = raw_to_binary[raw_label]
        label_id = label_to_id[binary_label]

        text, n_total, n_customer = build_prefix_text(conv, fraction, data_config)
        if n_customer == 0:
            n_zero_customer += 1

        texts.append(text)            # may be "" — handled by tokenizer as [CLS][SEP]
        true_labels.append(label_id)
        conv_ids.append(str(conv.get(id_field, "")))
        n_total_list.append(n_total)
        n_customer_list.append(n_customer)

    # Batch tokenization + inference
    all_logits: List[torch.Tensor] = []
    for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
        batch_texts = texts[i : i + INFERENCE_BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        all_logits.append(out.logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()
    y_true = np.array(true_labels, dtype=np.int64)
    pos_id = label_to_id[positive_class]

    # Measure true truncation using attention mask (last token not padding = not truncated)
    # We approximate: if seq uses exactly max_length non-pad tokens → truncated
    # Actually, we can count by re-running tokenizer without truncation and comparing length
    # For speed, use character estimate (already reliable enough for reporting)
    sep = data_config.get("message_separator", " [SEP] ")
    n_estimated_truncated = sum(
        1 for t in texts if (len(t) / 1.7 + 2) > max_length
    )

    metrics: Dict[str, float] = {
        "accuracy":    float(accuracy_score(y_true, preds)),
        "precision":   float(precision_score(y_true, preds, pos_label=pos_id, average="binary", zero_division=0)),
        "recall":      float(recall_score(y_true, preds, pos_label=pos_id, average="binary", zero_division=0)),
        "f1":          float(f1_score(y_true, preds, pos_label=pos_id, average="binary", zero_division=0)),
        "macro_f1":    float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }

    prefix_stats: Dict[str, Any] = {
        "fraction": fraction,
        "n_conversations": len(conv_ids),
        "avg_total_msgs_in_prefix":    float(np.mean(n_total_list)),
        "avg_customer_msgs_in_prefix": float(np.mean(n_customer_list)),
        "pct_zero_customer_messages":  n_zero_customer / len(conv_ids) * 100,
        "pct_estimated_truncated":     n_estimated_truncated / len(conv_ids) * 100,
    }

    return metrics, probs, preds, y_true, conv_ids, prefix_stats


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_prefix_outputs(
    output_dir: Path,
    fraction: float,
    metrics: Dict[str, float],
    probs: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
    conv_ids: List[str],
    label_to_id: Dict[str, int],
    conversations_lookup: Dict[str, Dict[str, Any]],
    data_config: Dict[str, Any],
) -> None:
    tag = f"{int(fraction * 100):03d}pct"
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_labels = [id_to_label[i] for i in range(len(label_to_id))]

    # Metrics JSON
    with (output_dir / f"metrics_{tag}.json").open("w", encoding="utf-8") as fh:
        json.dump({**metrics, "fraction": fraction}, fh, indent=2)

    # Classification report
    report = classification_report(y_true, preds, target_names=class_labels, zero_division=0)
    (output_dir / f"classification_report_{tag}.txt").write_text(report, encoding="utf-8")

    # Confusion matrix
    cm = confusion_matrix(y_true, preds, labels=list(range(len(class_labels))))
    cm_rows = [[""] + [f"pred_{c}" for c in class_labels]]
    for i, c in enumerate(class_labels):
        cm_rows.append([f"actual_{c}"] + [str(v) for v in cm[i]])
    with (output_dir / f"confusion_matrix_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(cm_rows)

    # Per-conversation predictions
    id_field    = data_config.get("id_field", "conversation_id")
    label_field = data_config.get("label_field", "final_outcome")
    raw_to_binary: Dict[str, str] = {}
    for mapped, raws in data_config.get("label_mapping", {}).items():
        for raw in raws:
            raw_to_binary[raw] = mapped

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
        for i, cid in enumerate(conv_ids):
            conv = conversations_lookup.get(cid, {})
            msgs = conv.get(data_config.get("messages_field", "messages"), [])
            plen = prefix_length(len(msgs), fraction)
            prefix_msgs = msgs[:plen]
            n_cust = sum(1 for m in prefix_msgs if m.get(data_config.get("role_field", "role"))
                         in data_config.get("included_roles", ["customer"]))

            # Recover probability ordering by label_to_id
            int_prob  = float(probs[i, label_to_id["interested"]])
            li_prob   = float(probs[i, label_to_id["losing_interest"]])

            writer.writerow({
                "conversation_id":              cid,
                "fraction":                     fraction,
                "num_total_msgs_in_prefix":     plen,
                "num_customer_msgs_in_prefix":  n_cust,
                "actual_label":                 id_to_label[int(y_true[i])],
                "predicted_label":              id_to_label[int(preds[i])],
                "probability_interested":       f"{int_prob:.6f}",
                "probability_losing_interest":  f"{li_prob:.6f}",
                "correct":                      int(y_true[i] == preds[i]),
            })


# ---------------------------------------------------------------------------
# Comparison + interpretation
# ---------------------------------------------------------------------------

def print_comparison_table(
    all_metrics: Dict[float, Dict[str, float]],
    all_stats:   Dict[float, Dict[str, Any]],
) -> None:
    pos = "losing_interest"
    print("\n" + "=" * 92)
    print("EARLY DETECTION E1 — PREFIX EVALUATION RESULTS")
    print("=" * 92)
    print(f"  Model: alephbert_continued_ablation_v1 (frozen, inference only)")
    print(f"  Test set: 459 conversations")
    print()

    # Stats table
    print(f"  {'Prefix':>8}  {'Avg total msgs':>15}  {'Avg cust msgs':>14}  "
          f"{'Zero cust %':>12}  {'Truncated %':>12}")
    print("  " + "-" * 66)
    for frac in FRACTIONS:
        s = all_stats[frac]
        print(f"  {int(frac*100):>7}%  {s['avg_total_msgs_in_prefix']:>15.1f}  "
              f"{s['avg_customer_msgs_in_prefix']:>14.1f}  "
              f"{s['pct_zero_customer_messages']:>11.1f}%  "
              f"{s['pct_estimated_truncated']:>11.1f}%")

    # Metrics table
    print()
    print(f"  {'Prefix':>8}  {'Accuracy':>10}  {'Macro F1':>10}  "
          f"{'LI Prec':>9}  {'LI Recall':>10}  {'LI F1':>8}")
    print("  " + "-" * 62)
    for frac in FRACTIONS:
        m = all_metrics[frac]
        print(f"  {int(frac*100):>7}%  {m['accuracy']:>10.4f}  {m['macro_f1']:>10.4f}  "
              f"{m['precision']:>9.4f}  {m['recall']:>10.4f}  {m['f1']:>8.4f}")

    # Deltas
    print()
    print(f"  Macro F1 gains across prefix levels:")
    fracs = FRACTIONS
    for i in range(1, len(fracs)):
        prev = all_metrics[fracs[i-1]]["macro_f1"]
        curr = all_metrics[fracs[i]]["macro_f1"]
        delta = (curr - prev) * 100
        print(f"    {int(fracs[i-1]*100)}% → {int(fracs[i]*100)}%: {delta:+.2f}pp")

    # Thresholds
    print()
    print("  Earliest prefix reaching performance thresholds:")
    for threshold_name, metric_key, threshold in [
        ("Macro F1 ≥ 0.80",           "macro_f1", 0.80),
        ("Macro F1 ≥ 0.90",           "macro_f1", 0.90),
        ("Macro F1 ≥ 0.95",           "macro_f1", 0.95),
        ("LI recall ≥ 0.90",          "recall",   0.90),
    ]:
        earliest = next(
            (f"{int(frac*100)}%" for frac in FRACTIONS
             if all_metrics[frac].get(metric_key, 0) >= threshold),
            "not reached"
        )
        print(f"    {threshold_name:<30}: {earliest}")

    # 100% consistency check
    m100 = all_metrics[1.0]["macro_f1"]
    match = abs(m100 - ABLATION_TEST_MACRO_F1) < 0.002
    print()
    print(f"  100% prefix macro F1: {m100:.4f}  "
          f"(reference: {ABLATION_TEST_MACRO_F1})  "
          f"{'✓ MATCHES' if match else '✗ MISMATCH — investigate!'}")
    print("=" * 92)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1: prefix evaluation, no retraining.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def run(config_path: Path) -> None:
    config   = load_config(config_path)
    base_dir = config_path.resolve().parent
    device   = get_device()

    data_config = config["data"]
    max_length  = int(config["tokenizer"]["max_length"])
    positive_class = config.get("fusion", {}).get("positive_class", "losing_interest")

    label_mapping = data_config["label_mapping"]
    label_to_id   = {lbl: i for i, lbl in enumerate(sorted(label_mapping.keys()))}

    output_dir = base_dir / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guard: do not overwrite existing experiments
    for protected in ["alephbert_baseline_v1", "alephbert_continued_ablation_v1",
                      "fusion_alephbert_behavioral_v1", "behavioral_baseline_v1",
                      "pure_behavioral_baseline_v1"]:
        if (base_dir / "outputs" / protected) == output_dir:
            raise RuntimeError(f"Output dir would overwrite protected path: {protected}")

    print("=" * 64)
    print("Early Detection Experiment E1")
    print("=" * 64)

    # ---- Load best checkpoint ----
    ckpt_dir = base_dir / "outputs" / "alephbert_continued_ablation_v1" / "best_model"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    logger.info("Loading tokenizer from %s ...", ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    logger.info("Loading model from %s ...", ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir),
        num_labels=2,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    print(f"  Checkpoint : {ckpt_dir}")
    print(f"  Device     : {device}")
    print(f"  max_length : {max_length}")
    print(f"  Fractions  : {[int(f*100) for f in FRACTIONS]}%")

    # ---- Load test IDs ----
    split_ids_path = base_dir / config.get("split", {}).get("split_ids_output",
                                                              "outputs/split_ids.json")
    _, _, test_ids = load_split_ids(split_ids_path)
    test_id_set = set(test_ids)
    print(f"  Test IDs   : {len(test_ids)}")

    # ---- Load corpus, filter to test set ----
    input_path = Path(data_config["input_path"])
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    logger.info("Loading corpus from %s ...", input_path)
    with input_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    corpus = raw if isinstance(raw, list) else next(
        (raw[k] for k in ("data", "records", "conversations", "items") if k in raw), [raw]
    )

    id_field = data_config.get("id_field", "conversation_id")
    conversations_lookup: Dict[str, Dict[str, Any]] = {
        str(c.get(id_field, "")): c for c in corpus
    }
    test_convs = [conversations_lookup[cid] for cid in test_ids
                  if cid in conversations_lookup]
    print(f"  Corpus test convs found: {len(test_convs)} / {len(test_ids)}")
    if len(test_convs) != len(test_ids):
        logger.warning("Could not find all test IDs in corpus!")

    # ---- Evaluate each prefix ----
    all_metrics: Dict[float, Dict[str, float]] = {}
    all_stats:   Dict[float, Dict[str, Any]]   = {}

    for fraction in FRACTIONS:
        pct_label = f"{int(fraction * 100)}%"
        logger.info("Evaluating prefix %s ...", pct_label)
        metrics, probs, preds, y_true, conv_ids, prefix_stats = infer_prefix(
            model, tokenizer, test_convs, fraction,
            data_config, label_to_id, positive_class, device, max_length,
        )
        all_metrics[fraction] = metrics
        all_stats[fraction]   = prefix_stats

        id_to_lbl = {v: k for k, v in label_to_id.items()}
        class_labels = [id_to_lbl[i] for i in range(len(label_to_id))]
        print(f"\n  [{pct_label}] acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
              f"f1={metrics['f1']:.4f}  prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}")
        print(classification_report(y_true, preds, target_names=class_labels, zero_division=0))

        save_prefix_outputs(
            output_dir, fraction, metrics, probs, preds, y_true, conv_ids,
            label_to_id, conversations_lookup, data_config,
        )
        logger.info("Saved outputs for %s to %s", pct_label, output_dir)

    # ---- Save experiment config + stats ----
    exp_cfg = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(ckpt_dir),
        "fractions": FRACTIONS,
        "prefix_rule": "max(1, ceil(total_messages * fraction))",
        "prefix_over": "ALL messages (chronological order by message_index)",
        "then_filter": data_config.get("included_roles", ["customer"]),
        "fallback_zero_customer": "empty string — tokenizer encodes as [CLS][SEP]+padding",
        "max_length": max_length,
        "test_conversations": len(test_ids),
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "no_retraining": True,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as fh:
        json.dump(exp_cfg, fh, indent=2)

    with (output_dir / "prefix_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in all_stats.items()}, fh, indent=2)

    # Comparison summary
    comparison = {
        str(int(frac * 100)): {**all_metrics[frac], **all_stats[frac]}
        for frac in FRACTIONS
    }
    with (output_dir / "comparison_table.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    print_comparison_table(all_metrics, all_stats)

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
