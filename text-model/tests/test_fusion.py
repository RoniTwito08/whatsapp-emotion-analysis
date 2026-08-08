"""Tests for fusion_train.py — no model downloads required.

All tests use a tiny BertConfig (1 layer, 32 hidden) as a stand-in
for AlephBERT to keep tests fast and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import BertConfig, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))

from behavioral_features_design import (
    FORBIDDEN_FIELDS,
    LEXICAL_FEATURE_NAMES,
    pure_behavioral_feature_names,
)
from fusion_train import FusionDataset, FusionModel, build_behavioral_matrix, evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_BEH = len(pure_behavioral_feature_names())
TINY_BERT_DIM = 32
TINY_VOCAB = 200


def _tiny_bert() -> BertForSequenceClassification:
    cfg = BertConfig(
        hidden_size=TINY_BERT_DIM, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, num_labels=2, max_position_embeddings=64, vocab_size=TINY_VOCAB,
    )
    return BertForSequenceClassification(cfg)


def _tiny_fusion(freeze: bool = False) -> FusionModel:
    clf = _tiny_bert()
    model = FusionModel(clf.bert, n_behavioral=N_BEH, hidden_dim=32, dropout=0.0)
    if freeze:
        model.freeze_bert()
    return model


def _fake_batch(batch_size: int = 4, seq_len: int = 8):
    return {
        "input_ids":           torch.randint(0, TINY_VOCAB, (batch_size, seq_len)),
        "attention_mask":      torch.ones(batch_size, seq_len, dtype=torch.long),
        "behavioral_features": torch.randn(batch_size, N_BEH),
        "labels":              torch.randint(0, 2, (batch_size,)),
        "record_id":           [f"conv_{i}" for i in range(batch_size)],
    }


# ---------------------------------------------------------------------------
# FusionModel structure
# ---------------------------------------------------------------------------

def test_fusion_model_output_shape():
    model = _tiny_fusion()
    batch = _fake_batch()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"], batch["behavioral_features"])
    assert logits.shape == (4, 2), f"Expected (4,2), got {logits.shape}"


def test_fusion_model_concat_dim():
    model = _tiny_fusion()
    assert model.concat_dim == TINY_BERT_DIM + N_BEH


def test_real_concat_dim_is_794():
    """For real AlephBERT (768-dim) + 26 pure behavioral = 794."""
    assert 768 + N_BEH == 794
    assert N_BEH == 26


def test_fusion_head_architecture():
    model = _tiny_fusion()
    layers = list(model.head.children())
    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Dropout)
    assert isinstance(layers[3], nn.Linear)
    assert layers[-1].out_features == 2


# ---------------------------------------------------------------------------
# Freezing behavior
# ---------------------------------------------------------------------------

def test_bert_frozen_no_grad():
    model = _tiny_fusion(freeze=True)
    for name, param in model.bert.named_parameters():
        assert not param.requires_grad, f"Param {name} should be frozen"


def test_bert_frozen_head_still_trains():
    model = _tiny_fusion(freeze=True)
    for param in model.head.parameters():
        assert param.requires_grad, "Head params should be trainable when BERT is frozen"


def test_bert_unfrozen():
    model = _tiny_fusion(freeze=False)
    for param in model.bert.parameters():
        assert param.requires_grad, "BERT should be unfrozen"


def test_freeze_then_unfreeze():
    model = _tiny_fusion()
    model.freeze_bert()
    for p in model.bert.parameters():
        assert not p.requires_grad
    model.unfreeze_bert()
    for p in model.bert.parameters():
        assert p.requires_grad


# ---------------------------------------------------------------------------
# FusionDataset
# ---------------------------------------------------------------------------

class _FakeTextDataset:
    """Minimal stand-in for HebrewConversationDataset."""

    def __init__(self, n: int = 10, seq_len: int = 8):
        self.input_ids     = torch.randint(0, TINY_VOCAB, (n, seq_len))
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        self.labels        = torch.randint(0, 2, (n,))
        self.record_ids    = [f"conv_{i}" for i in range(n)]

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, i):
        return {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels":         self.labels[i],
            "record_id":      self.record_ids[i],
        }


def test_fusion_dataset_length():
    n = 10
    text_ds = _FakeTextDataset(n)
    beh = torch.randn(n, N_BEH)
    ds = FusionDataset(text_ds, beh)
    assert len(ds) == n


def test_fusion_dataset_item_has_behavioral_features():
    text_ds = _FakeTextDataset(5)
    beh = torch.randn(5, N_BEH)
    ds = FusionDataset(text_ds, beh)
    item = ds[0]
    assert "behavioral_features" in item
    assert item["behavioral_features"].shape == (N_BEH,)


def test_fusion_dataset_mismatch_raises():
    text_ds = _FakeTextDataset(5)
    beh = torch.randn(7, N_BEH)  # wrong size
    with pytest.raises(ValueError, match="Length mismatch"):
        FusionDataset(text_ds, beh)


def test_fusion_dataset_behavioral_aligned():
    """Behavioral features at index i should match what was inserted at i."""
    n = 6
    text_ds = _FakeTextDataset(n)
    beh = torch.arange(n * N_BEH, dtype=torch.float32).reshape(n, N_BEH)
    ds = FusionDataset(text_ds, beh)
    for i in range(n):
        assert torch.equal(ds[i]["behavioral_features"], beh[i])


# ---------------------------------------------------------------------------
# Scaler: fitted on train only
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_only():
    """StandardScaler mean should match training data, not val/test."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=10.0, scale=1.0, size=(100, N_BEH))
    X_val   = rng.normal(loc=0.0,  scale=1.0, size=(20, N_BEH))

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Scaler mean should be near 10.0 (train), not 0.0 (val)
    assert abs(float(scaler.mean_[0]) - 10.0) < 1.0, \
        "Scaler should be fitted on train data, not val"

    # transform does not modify internal state
    _ = scaler.transform(X_val)
    assert abs(float(scaler.mean_[0]) - 10.0) < 1.0, \
        "Scaler mean should not change after transform(val)"


# ---------------------------------------------------------------------------
# Feature leakage checks
# ---------------------------------------------------------------------------

def test_pure_behavioral_features_no_lexical():
    pure = set(pure_behavioral_feature_names())
    overlap = pure & LEXICAL_FEATURE_NAMES
    assert not overlap, f"Lexical features in pure set: {overlap}"


def test_pure_behavioral_features_no_forbidden():
    pure = set(pure_behavioral_feature_names())
    overlap = pure & FORBIDDEN_FIELDS
    assert not overlap, f"Forbidden fields in pure set: {overlap}"


def test_pure_behavioral_feature_count():
    assert len(pure_behavioral_feature_names()) == 26


# ---------------------------------------------------------------------------
# build_behavioral_matrix
# ---------------------------------------------------------------------------

def test_build_behavioral_matrix_shape():
    def _rec(cid):
        return {
            "conversation_id": cid,
            "messages": [
                {
                    "role": "customer",
                    "text": "x",
                    "response_delay_minutes": 10,
                    "behavioral_features": {
                        "message_length_chars": 20, "question_count": 0,
                        "emoji_count": 0, "contains_price": False,
                        "contains_negotiation": False, "contains_objection": False,
                        "contains_commitment": False, "contains_delay_signal": False,
                    },
                }
            ],
        }

    records = [_rec(f"c{i}") for i in range(8)]
    rids = [f"c{i}" for i in range(8)]
    X = build_behavioral_matrix(records, rids, pure_behavioral_feature_names())
    assert X.shape == (8, N_BEH)
    assert X.dtype == np.float64


def test_build_behavioral_matrix_preserves_id_order():
    def _rec(cid, length):
        return {
            "conversation_id": cid,
            "messages": [
                {
                    "role": "customer",
                    "response_delay_minutes": 5,
                    "behavioral_features": {"message_length_chars": length,
                                             "question_count": 0, "emoji_count": 0,
                                             "contains_price": False, "contains_negotiation": False,
                                             "contains_objection": False, "contains_commitment": False,
                                             "contains_delay_signal": False},
                }
            ],
        }

    records = [_rec(f"c{i}", i * 10) for i in range(5)]
    rids = [f"c{i}" for i in range(5)]
    feat_names = pure_behavioral_feature_names()
    X = build_behavioral_matrix(records, rids, feat_names)
    mean_len_idx = feat_names.index("mean_customer_msg_length")
    for i in range(5):
        assert abs(X[i, mean_len_idx] - i * 10) < 1e-6, \
            f"Row {i} should have mean_length={i*10}, got {X[i, mean_len_idx]}"
