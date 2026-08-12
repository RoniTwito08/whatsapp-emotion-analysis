"""Tests for early_detection_e3.py — no model downloads required.

The most critical test is no-leakage: behavioral features at fraction f
must ONLY use messages from the prefix, never from later messages.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from behavioral_features_design import pure_behavioral_feature_names, LEXICAL_FEATURE_NAMES
from early_detection_e1 import prefix_length
from early_detection_e3 import (
    EXPERIMENT_NAME,
    PrefixFusionDataset,
    extract_prefix_behavioral,
)

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
N_BEH = len(pure_behavioral_feature_names())

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
        "losing_interest": ["ghosted",   "rejected"],
    },
}
LABEL_TO_ID = {"interested": 0, "losing_interest": 1}
FEAT_NAMES  = pure_behavioral_feature_names()


def _fake_msg(idx: int, role: str = "customer") -> Dict[str, Any]:
    return {
        "message_index": idx,
        "role": role,
        "text": f"message {idx}",
        "response_delay_minutes": float(10 + idx),
        "behavioral_features": {
            "message_length_chars": 20 + idx,
            "question_count": 0,
            "emoji_count": 0,
            "contains_price": False,
            "contains_negotiation": False,
            "contains_objection": False,
            "contains_commitment": False,
            "contains_delay_signal": False,
        },
    }


def _fake_conv(cid: str, outcome: str, n: int = 12) -> Dict[str, Any]:
    msgs = []
    for i in range(n):
        role = "customer" if i % 2 == 0 else "business"
        msgs.append(_fake_msg(i, role))
    return {"conversation_id": cid, "final_outcome": outcome, "messages": msgs}


def _mock_tokenizer(n: int, max_length: int = 16) -> MagicMock:
    tok = MagicMock()
    tok.return_value = {
        "input_ids":      torch.zeros(n, max_length, dtype=torch.long),
        "attention_mask": torch.ones(n,  max_length, dtype=torch.long),
    }
    return tok


# ---------------------------------------------------------------------------
# Leakage tests — the most critical
# ---------------------------------------------------------------------------

class TestNoLeakage:
    """Verify that behavioral features never use future messages."""

    def test_total_messages_reflects_prefix_only(self):
        """total_messages at 25% of a 12-msg conv should be 3, not 12."""
        conv = _fake_conv("c0", "converted", 12)
        feats = extract_prefix_behavioral(conv, 0.25, FEAT_NAMES, DATA_CONFIG)
        idx = FEAT_NAMES.index("total_messages")
        assert feats[idx] == 3.0, f"Expected 3 total messages for 25% of 12, got {feats[idx]}"

    def test_customer_messages_reflects_prefix_only(self):
        """Customer count at 50% prefix must be from first 6 messages only."""
        conv = _fake_conv("c0", "converted", 12)  # alternating: 6 customer in first 6
        feats_50 = extract_prefix_behavioral(conv, 0.50, FEAT_NAMES, DATA_CONFIG)
        feats_100 = extract_prefix_behavioral(conv, 1.00, FEAT_NAMES, DATA_CONFIG)
        idx = FEAT_NAMES.index("customer_messages")
        # 50%: first 6 msgs, alternating → 3 customer
        assert feats_50[idx] < feats_100[idx], "50% should have fewer customer msgs than 100%"
        assert feats_50[idx] == 3.0

    def test_modifying_future_messages_does_not_change_prefix_features(self):
        """Changing messages beyond the prefix must not affect prefix features."""
        conv = _fake_conv("c0", "converted", 12)
        feats_before = extract_prefix_behavioral(conv, 0.25, FEAT_NAMES, DATA_CONFIG)

        # Poison messages 3-11 (beyond 25% prefix)
        poisoned = copy.deepcopy(conv)
        for msg in poisoned["messages"][3:]:
            msg["response_delay_minutes"] = 9999.0
            msg["behavioral_features"]["message_length_chars"] = 9999

        feats_after = extract_prefix_behavioral(poisoned, 0.25, FEAT_NAMES, DATA_CONFIG)
        assert feats_before == feats_after, \
            "Poisoning future messages should not affect 25% prefix features"

    def test_25pct_features_different_from_100pct(self):
        """25% and 100% features must differ (otherwise slicing is not happening)."""
        conv = _fake_conv("c0", "converted", 12)
        feats_25  = extract_prefix_behavioral(conv, 0.25, FEAT_NAMES, DATA_CONFIG)
        feats_100 = extract_prefix_behavioral(conv, 1.00, FEAT_NAMES, DATA_CONFIG)
        assert feats_25 != feats_100, "25% and 100% features should differ"

    def test_no_lexical_features_in_behavioral_vector(self):
        """Pure behavioral set must not contain any lexical regex features."""
        overlap = set(FEAT_NAMES) & LEXICAL_FEATURE_NAMES
        assert not overlap, f"Lexical features found in pure set: {overlap}"

    def test_features_monotone_in_fraction_for_message_counts(self):
        """total_messages should be non-decreasing as fraction increases."""
        conv = _fake_conv("c0", "converted", 12)
        prev = 0.0
        idx = FEAT_NAMES.index("total_messages")
        for frac in FRACTIONS:
            feats = extract_prefix_behavioral(conv, frac, FEAT_NAMES, DATA_CONFIG)
            assert feats[idx] >= prev, \
                f"total_messages at {frac} ({feats[idx]}) < at previous fraction ({prev})"
            prev = feats[idx]

    def test_feature_vector_length(self):
        """extract_prefix_behavioral must always return exactly N_BEH floats."""
        conv = _fake_conv("c0", "converted", 8)
        for frac in FRACTIONS:
            feats = extract_prefix_behavioral(conv, frac, FEAT_NAMES, DATA_CONFIG)
            assert len(feats) == N_BEH, \
                f"Expected {N_BEH} features at {frac}, got {len(feats)}"


# ---------------------------------------------------------------------------
# PrefixFusionDataset structure
# ---------------------------------------------------------------------------

def test_prefix_fusion_dataset_length():
    """N convs × F fracs → N×F dataset length."""
    convs = [_fake_conv(f"c{i}", "converted" if i % 2 == 0 else "ghosted") for i in range(4)]
    n, f = len(convs), len(FRACTIONS)
    tok = _mock_tokenizer(n * f)
    ds = PrefixFusionDataset(convs, FRACTIONS, DATA_CONFIG, tok, 16, LABEL_TO_ID, FEAT_NAMES)
    assert len(ds) == n * f


def test_prefix_fusion_dataset_has_behavioral_features():
    """Each item must include 'behavioral_features' tensor of shape (N_BEH,)."""
    convs = [_fake_conv(f"c{i}", "converted" if i % 2 == 0 else "ghosted") for i in range(4)]
    tok = _mock_tokenizer(len(convs) * len(FRACTIONS))
    ds = PrefixFusionDataset(convs, FRACTIONS, DATA_CONFIG, tok, 16, LABEL_TO_ID, FEAT_NAMES)
    item = ds[0]
    assert "behavioral_features" in item
    assert item["behavioral_features"].shape == (N_BEH,)


def test_prefix_fusion_dataset_scaler_applied():
    """With a fitted scaler, behavioral features should be standardised."""
    from sklearn.preprocessing import StandardScaler
    convs = [_fake_conv(f"c{i}", "converted" if i % 2 == 0 else "ghosted") for i in range(4)]
    n_items = len(convs) * len(FRACTIONS)

    # Build raw features for fitting
    raw = [extract_prefix_behavioral(c, f, FEAT_NAMES, DATA_CONFIG)
           for c in convs for f in FRACTIONS]
    sc = StandardScaler().fit(np.array(raw))

    tok = _mock_tokenizer(n_items)
    ds_scaled   = PrefixFusionDataset(convs, FRACTIONS, DATA_CONFIG, tok, 16,
                                      LABEL_TO_ID, FEAT_NAMES, scaler=sc)
    tok2 = _mock_tokenizer(n_items)
    ds_unscaled = PrefixFusionDataset(convs, FRACTIONS, DATA_CONFIG, tok2, 16,
                                      LABEL_TO_ID, FEAT_NAMES, scaler=None)

    # Scaled and unscaled should differ
    scaled_vec   = ds_scaled[0]["behavioral_features"].numpy()
    unscaled_vec = ds_unscaled[0]["behavioral_features"].numpy()
    assert not np.allclose(scaled_vec, unscaled_vec), \
        "Scaler must change the behavioral feature values"


def test_all_prefixes_inherit_same_label():
    """All fractions of a conversation produce the same label_id."""
    conv = _fake_conv("c0", "converted", 12)
    for frac in FRACTIONS:
        tok = _mock_tokenizer(1)
        ds = PrefixFusionDataset([conv], [frac], DATA_CONFIG, tok, 16,
                                 LABEL_TO_ID, FEAT_NAMES)
        assert int(ds.labels[0]) == LABEL_TO_ID["interested"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_extract_prefix_behavioral_deterministic():
    conv = _fake_conv("c0", "converted", 10)
    f1 = extract_prefix_behavioral(conv, 0.5, FEAT_NAMES, DATA_CONFIG)
    f2 = extract_prefix_behavioral(conv, 0.5, FEAT_NAMES, DATA_CONFIG)
    assert f1 == f2


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------

def test_test_ids_count_459():
    p = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    assert len(json.loads(p.read_text())["test_ids"]) == 459


def test_experiment_name_not_protected():
    protected = [
        "alephbert_baseline_v1", "alephbert_continued_ablation_v1",
        "early_detection_e1", "early_detection_e2",
        "fusion_alephbert_behavioral_v1", "behavioral_baseline_v1",
        "pure_behavioral_baseline_v1",
    ]
    assert EXPERIMENT_NAME not in protected


# ---------------------------------------------------------------------------
# Model input dimensions
# ---------------------------------------------------------------------------

def test_concat_dim_is_794():
    """768 BERT dim + 26 behavioral = 794."""
    assert 768 + N_BEH == 794


def test_n_behavioral_features_is_26():
    assert N_BEH == 26
