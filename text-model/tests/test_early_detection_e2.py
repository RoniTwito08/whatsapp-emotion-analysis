"""Tests for early_detection_e2.py — no model downloads required."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from early_detection_e1 import build_prefix_text, prefix_length
from early_detection_e2 import PrefixAugmentedDataset, EXPERIMENT_NAME, selection_metric

FRACTIONS = [0.25, 0.50, 0.75, 1.00]

DATA_CONFIG = {
    "messages_field": "messages",
    "role_field": "role",
    "text_field": "text",
    "included_roles": ["customer"],
    "message_separator": " [SEP] ",
    "id_field": "conversation_id",
    "label_field": "final_outcome",
    "label_mapping": {
        "interested":      ["converted", "pending"],
        "losing_interest": ["ghosted", "rejected"],
    },
}

LABEL_TO_ID = {"interested": 0, "losing_interest": 1}


def _fake_conv(cid: str, outcome: str, n_msgs: int = 8) -> Dict[str, Any]:
    msgs = []
    for i in range(n_msgs):
        msgs.append({
            "message_index": i,
            "role": "customer" if i % 2 == 0 else "business",
            "text": f"msg {i} of {cid}",
            "response_delay_minutes": 10,
        })
    return {"conversation_id": cid, "final_outcome": outcome, "messages": msgs}


def _make_mock_tokenizer(n: int, max_length: int = 16) -> MagicMock:
    tok = MagicMock()
    tok.return_value = {
        "input_ids":      torch.zeros(n, max_length, dtype=torch.long),
        "attention_mask": torch.ones(n, max_length, dtype=torch.long),
    }
    return tok


# ---------------------------------------------------------------------------
# Prefix augmentation count
# ---------------------------------------------------------------------------

def test_augmented_count_is_n_convs_times_n_fracs():
    """N conversations × F fractions → N×F examples."""
    convs = [_fake_conv(f"c{i}", "converted" if i % 2 == 0 else "ghosted", 8) for i in range(5)]
    n, f = len(convs), len(FRACTIONS)
    tokenizer = _make_mock_tokenizer(n * f)
    ds = PrefixAugmentedDataset(convs, FRACTIONS, DATA_CONFIG, tokenizer, 16, LABEL_TO_ID)
    assert len(ds) == n * f


def test_augmented_count_single_fraction():
    convs = [_fake_conv(f"c{i}", "converted", 8) for i in range(3)]
    tokenizer = _make_mock_tokenizer(3)
    # Only 1 fraction
    ds = PrefixAugmentedDataset(convs, [1.0], DATA_CONFIG, tokenizer, 16, LABEL_TO_ID)
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# All prefixes of a conversation inherit the same final label
# ---------------------------------------------------------------------------

def test_all_prefixes_inherit_same_label():
    convs = [_fake_conv("c_converted", "converted", 12), _fake_conv("c_ghosted", "ghosted", 12)]
    tokenizer = _make_mock_tokenizer(len(convs) * len(FRACTIONS))
    ds = PrefixAugmentedDataset(convs, FRACTIONS, DATA_CONFIG, tokenizer, 16, LABEL_TO_ID)

    labels_by_conv: Dict[str, List[int]] = {}
    for idx in range(len(ds)):
        cid = ds.conv_ids[idx]
        lbl = int(ds.labels[idx])
        labels_by_conv.setdefault(cid, []).append(lbl)

    for cid, lbls in labels_by_conv.items():
        assert len(set(lbls)) == 1, f"Conv {cid} has mixed labels {lbls}"


def test_all_prefixes_in_same_split():
    """No prefix of a training conversation should appear in val or test."""
    split_path = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    payload = json.loads(split_path.read_text())
    train_set = set(payload["train_ids"])
    val_set   = set(payload["val_ids"])
    test_set  = set(payload["test_ids"])

    assert not (train_set & val_set),  "train/val overlap"
    assert not (train_set & test_set), "train/test overlap"
    assert not (val_set   & test_set), "val/test overlap"
    # Since all prefixes of a conv get the same split, no prefix can cross splits
    # (guaranteed by design — this test validates the underlying split integrity)


# ---------------------------------------------------------------------------
# Chronological prefix ordering
# ---------------------------------------------------------------------------

def test_prefix_text_chronological():
    conv = {
        "messages": [
            {"message_index": 0, "role": "customer", "text": "first", "response_delay_minutes": 0},
            {"message_index": 1, "role": "business", "text": "biz",   "response_delay_minutes": 0},
            {"message_index": 2, "role": "customer", "text": "second","response_delay_minutes": 0},
            {"message_index": 3, "role": "business", "text": "biz2",  "response_delay_minutes": 0},
        ]
    }
    text, _, _ = build_prefix_text(conv, 1.0, DATA_CONFIG)
    assert text.index("first") < text.index("second")


def test_25pct_prefix_smaller_than_50pct():
    conv = _fake_conv("c0", "converted", 12)
    plen_25 = prefix_length(12, 0.25)
    plen_50 = prefix_length(12, 0.50)
    assert plen_25 <= plen_50


# ---------------------------------------------------------------------------
# No leaky fields in text construction
# ---------------------------------------------------------------------------

def test_no_leaky_fields_in_prefix_text():
    conv = {
        "messages": [
            {
                "message_index": 0,
                "role": "customer",
                "text": "clean text only",
                "interest_label": "converted",    # leaky
                "interest_score": 0.99,            # leaky
                "response_delay_minutes": 5,
            }
        ],
        "final_outcome": "converted",              # label — must not appear in text
        "interest_trajectory": "high_to_conversion",
    }
    text, _, _ = build_prefix_text(conv, 1.0, DATA_CONFIG)
    assert text == "clean text only"
    assert "converted" not in text
    assert "interest" not in text


# ---------------------------------------------------------------------------
# Same 459 test IDs at every fraction
# ---------------------------------------------------------------------------

def test_test_ids_count_is_459():
    split_path = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    payload = json.loads(split_path.read_text())
    assert len(payload["test_ids"]) == 459


def test_dataset_produces_same_ids_across_fractions():
    """The set of conv_ids in the dataset is the same regardless of which fraction subset."""
    convs = [_fake_conv(f"c{i}", "converted" if i % 2 == 0 else "ghosted", 8) for i in range(4)]
    expected_ids = {c["conversation_id"] for c in convs}
    for frac in FRACTIONS:
        tok = _make_mock_tokenizer(len(convs))
        ds = PrefixAugmentedDataset(convs, [frac], DATA_CONFIG, tok, 16, LABEL_TO_ID)
        assert {ds.conv_ids[i] for i in range(len(ds))} == expected_ids


# ---------------------------------------------------------------------------
# Metrics reported per fraction
# ---------------------------------------------------------------------------

def test_selection_metric_averages_specified_fractions():
    """selection_metric should average only the specified fractions."""
    import numpy as np
    fake_results = {
        0.25: {"metrics": {"macro_f1": 0.60}},
        0.50: {"metrics": {"macro_f1": 0.70}},
        0.75: {"metrics": {"macro_f1": 0.80}},
        1.00: {"metrics": {"macro_f1": 0.98}},
    }
    sel = selection_metric(fake_results, [0.25, 0.50, 0.75])
    assert abs(sel - np.mean([0.60, 0.70, 0.80])) < 1e-9

    # If only 1.0 is selected, result should be 0.98
    sel_full = selection_metric(fake_results, [1.00])
    assert abs(sel_full - 0.98) < 1e-9


# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------

def test_deterministic_prefix_text():
    conv = _fake_conv("c0", "converted", 10)
    t1, n1, c1 = build_prefix_text(conv, 0.5, DATA_CONFIG)
    t2, n2, c2 = build_prefix_text(conv, 0.5, DATA_CONFIG)
    assert t1 == t2 and n1 == n2 and c1 == c2


def test_deterministic_dataset_order():
    """Same input convs + same fractions → same order in dataset."""
    convs = [_fake_conv(f"c{i}", "converted" if i % 2 == 0 else "ghosted", 8) for i in range(3)]
    n = len(convs) * len(FRACTIONS)
    tok1 = _make_mock_tokenizer(n)
    tok2 = _make_mock_tokenizer(n)
    ds1 = PrefixAugmentedDataset(convs, FRACTIONS, DATA_CONFIG, tok1, 16, LABEL_TO_ID)
    ds2 = PrefixAugmentedDataset(convs, FRACTIONS, DATA_CONFIG, tok2, 16, LABEL_TO_ID)
    assert ds1.conv_ids  == ds2.conv_ids
    assert ds1.fractions == ds2.fractions
    assert (ds1.labels == ds2.labels).all()


# ---------------------------------------------------------------------------
# Protected output directories
# ---------------------------------------------------------------------------

def test_experiment_name_does_not_collide_with_protected():
    protected = [
        "alephbert_baseline_v1",
        "alephbert_continued_ablation_v1",
        "early_detection_e1",
        "fusion_alephbert_behavioral_v1",
        "behavioral_baseline_v1",
        "pure_behavioral_baseline_v1",
    ]
    assert EXPERIMENT_NAME not in protected
