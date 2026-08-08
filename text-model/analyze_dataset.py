"""analyze_dataset.py

Dataset analysis for the Hebrew conversation corpus.

Reports: total conversations, class distribution, outcome distribution,
conversation length distribution, truncation impact at common max_length
values, domains, empty messages, and missing labels.

Usage:
    python analyze_dataset.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from data_loader import load_config, load_corpus_records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the Hebrew conversation dataset.")
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json")
    return parser.parse_args(argv)


def invert_label_mapping(label_mapping: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    inverted: Dict[str, str] = {}
    for mapped, raws in label_mapping.items():
        for raw in raws:
            inverted[raw] = mapped
    return inverted


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def analyze(records: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    data_cfg = config.get("data", {})
    label_mapping = data_cfg.get("label_mapping", {})
    raw_to_mapped = invert_label_mapping(label_mapping)
    label_field = data_cfg.get("label_field", "final_outcome")
    id_field = data_cfg.get("id_field", "conversation_id")
    messages_field = data_cfg.get("messages_field", "messages")
    role_field = data_cfg.get("role_field", "role")
    text_field = data_cfg.get("text_field", "text")
    included_roles = data_cfg.get("included_roles", ["customer"])
    separator = data_cfg.get("message_separator", " [SEP] ")

    print("=" * 64)
    print("DATASET ANALYSIS")
    print("=" * 64)

    # Basic counts
    total = len(records)
    print(f"\nTotal records  : {total}")

    # Unique IDs
    ids = [str(r.get(id_field, "")) for r in records]
    unique_ids = len(set(ids))
    duplicate_ids = total - unique_ids
    print(f"Unique IDs     : {unique_ids}")
    if duplicate_ids:
        print(f"Duplicate IDs  : {duplicate_ids}  *** WARNING ***")
    else:
        print(f"Duplicate IDs  : {duplicate_ids}  (none — OK)")

    # Raw outcome distribution
    print("\n--- Raw final_outcome distribution ---")
    raw_counts = Counter(str(r.get(label_field, "MISSING")) for r in records)
    for outcome, count in sorted(raw_counts.items()):
        mapped = raw_to_mapped.get(outcome, "UNMAPPED")
        print(f"  {outcome:30s}  {count:5d}  -> {mapped}")

    # Binary label distribution
    print("\n--- Binary label distribution ---")
    binary_counts: Counter = Counter()
    missing_label = 0
    unmapped_label = 0
    for r in records:
        raw = str(r.get(label_field, ""))
        if not raw or raw == "None":
            missing_label += 1
        elif raw not in raw_to_mapped:
            unmapped_label += 1
        else:
            binary_counts[raw_to_mapped[raw]] += 1
    for label, count in sorted(binary_counts.items()):
        pct = count / total * 100
        print(f"  {label:20s}: {count:5d}  ({pct:.1f}%)")
    if missing_label:
        print(f"  Missing label : {missing_label}")
    if unmapped_label:
        print(f"  Unmapped      : {unmapped_label}")

    # Conversation length analysis
    customer_char_counts: List[int] = []
    customer_msg_counts: List[int] = []
    total_msg_counts: List[int] = []
    empty_messages = 0
    missing_messages = 0

    for r in records:
        msgs = r.get(messages_field, [])
        if not isinstance(msgs, list) or not msgs:
            missing_messages += 1
            continue
        total_msg_counts.append(len(msgs))
        customer_msgs = [m for m in msgs if isinstance(m, dict) and m.get(role_field) in included_roles]
        customer_msg_counts.append(len(customer_msgs))
        texts = [m.get(text_field, "") for m in customer_msgs if m.get(text_field)]
        for t in texts:
            if not str(t).strip():
                empty_messages += 1
        joined = separator.join(str(m.get(text_field, "")) for m in customer_msgs if m.get(text_field))
        customer_char_counts.append(len(joined))

    print(f"\n--- Conversation lengths ---")
    print(f"  Missing message list  : {missing_messages}")
    print(f"  Empty message texts   : {empty_messages}")
    n = len(customer_char_counts)
    if n:
        def pct(v: List[int], p: float) -> float:
            return _percentile([float(x) for x in v], p)

        print(f"\n  Total messages per conversation:")
        print(f"    min={min(total_msg_counts)}  median={int(pct(total_msg_counts, 50))}  "
              f"90th={int(pct(total_msg_counts, 90))}  max={max(total_msg_counts)}")

        print(f"\n  Customer messages per conversation:")
        print(f"    min={min(customer_msg_counts)}  median={int(pct(customer_msg_counts, 50))}  "
              f"90th={int(pct(customer_msg_counts, 90))}  max={max(customer_msg_counts)}")

        print(f"\n  Customer text character count:")
        print(f"    min={min(customer_char_counts)}  median={int(pct(customer_char_counts, 50))}  "
              f"90th={int(pct(customer_char_counts, 90))}  "
              f"99th={int(pct(customer_char_counts, 99))}  max={max(customer_char_counts)}")

        # Truncation estimate (rough: ~1.7 chars per Hebrew subword token + special tokens)
        # True value requires the tokenizer; this is a pre-tokenizer estimate.
        estimated_tokens = [(c / 1.7) + 2 for c in customer_char_counts]
        print(f"\n  Estimated token count (customer text, chars/1.7 + 2 special):")
        print(f"    median={int(pct(estimated_tokens, 50))}  "
              f"90th={int(pct(estimated_tokens, 90))}  "
              f"95th={int(pct(estimated_tokens, 95))}  "
              f"99th={int(pct(estimated_tokens, 99))}")

        for max_len in [128, 256, 512]:
            pct_truncated = sum(1 for t in estimated_tokens if t > max_len) / n * 100
            print(f"\n  Estimated % truncated at max_length={max_len:3d}: {pct_truncated:5.1f}%")
            if pct_truncated > 10:
                note = "  *** RECOMMENDATION: consider max_length=512 ***" if max_len == 256 else ""
                print(f"    -> {pct_truncated:.1f}% of conversations lose content.{note}")

    # Domain distribution
    print("\n--- Domain distribution (top 10) ---")
    domain_counts = Counter(str(r.get("domain", "MISSING")) for r in records)
    for domain, count in domain_counts.most_common(10):
        print(f"  {domain:30s}: {count}")
    if len(domain_counts) > 10:
        print(f"  ... and {len(domain_counts) - 10} more domains")

    # Synthetic flag
    synthetic_counts = Counter(str(r.get("synthetic", "MISSING")) for r in records)
    print(f"\n--- Synthetic flag ---")
    for val, count in sorted(synthetic_counts.items()):
        print(f"  {val}: {count}")

    print("\n" + "=" * 64)
    print("LEAKAGE WARNINGS")
    print("=" * 64)
    print("""
  1. Per-message 'interest_label' field: Each message contains an
     interest_label (e.g. 'converted', 'rejected', 'losing_interest').
     The current text extraction reads ONLY msg['text'] — this field is
     NEVER fed to the model. Safe as long as text_field = 'text'.

  2. Terminal customer messages: The last customer message in converted
     conversations contains commitment language; in rejected conversations
     it contains rejection language. This is a natural correlation (not
     a bug), but in synthetic data it may be more stereotyped than real
     data, which could inflate test performance. Document this clearly.

  3. Fields that MUST NOT enter the model input:
       - final_outcome  (the label itself)
       - interest_trajectory  (encodes the outcome path)
       - final_interest_score (correlated with outcome)
       - interest_score per message (correlated with outcome)
     Verify that included_roles='customer' and text_field='text' are
     the only data entering the tokenizer.
""")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        base_dir = args.config.resolve().parent
        records = load_corpus_records(config, base_dir=base_dir)
        analyze(records, config)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
