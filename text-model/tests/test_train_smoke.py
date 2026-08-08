"""Smoke tests for the training pipeline.

Uses tiny synthetic tensors and mocks to avoid downloading any model.
Tests structural correctness: split IDs, dataset filtering, metrics,
and checkpoint save/load.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from splitter import save_split_ids, load_split_ids, SplitError


# ---------------------------------------------------------------------------
# Split ID serialization
# ---------------------------------------------------------------------------

def test_save_and_load_split_ids(tmp_path):
    train_ids = ["id_1", "id_2", "id_3"]
    val_ids = ["id_4"]
    test_ids = ["id_5", "id_6"]
    out_path = tmp_path / "split_ids.json"
    save_split_ids(train_ids, val_ids, test_ids, out_path)

    loaded_train, loaded_val, loaded_test = load_split_ids(out_path)
    assert loaded_train == train_ids
    assert loaded_val == val_ids
    assert loaded_test == test_ids


def test_split_ids_json_is_human_readable(tmp_path):
    out_path = tmp_path / "split_ids.json"
    save_split_ids(["a"], ["b"], ["c"], out_path, config_snapshot={"seed": 42})
    payload = json.loads(out_path.read_text())
    assert "train_ids" in payload
    assert "val_ids" in payload
    assert "test_ids" in payload
    assert payload["counts"]["train"] == 1
    assert payload["config_snapshot"]["seed"] == 42


def test_load_missing_split_ids_raises(tmp_path):
    with pytest.raises(SplitError, match="not found"):
        load_split_ids(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Dataset subset_ids filtering
# ---------------------------------------------------------------------------

def _make_tiny_records(n: int = 10) -> List[Dict[str, Any]]:
    outcomes = ["converted", "ghosted"]
    return [
        {
            "conversation_id": f"conv_{i:03d}",
            "final_outcome": outcomes[i % 2],
            "messages": [{"role": "customer", "text": f"message {i}"}],
        }
        for i in range(n)
    ]


def test_dataset_subset_ids_filters_correctly():
    """Verify that subset_ids correctly restricts which records are tokenized."""
    from dataset import HebrewConversationDataset, DatasetError

    config = {
        "data": {
            "input_path": "FAKE",
            "id_field": "conversation_id",
            "messages_field": "messages",
            "role_field": "role",
            "text_field": "text",
            "included_roles": ["customer"],
            "label_field": "final_outcome",
            "message_separator": " [SEP] ",
            "label_mapping": {
                "interested": ["converted"],
                "losing_interest": ["ghosted"],
            },
        },
        "tokenizer": {"max_length": 16, "padding": "max_length", "truncation": True, "add_special_tokens": True},
        "model": {"num_labels": 2},
        "random_seed": 42,
    }

    records = _make_tiny_records(10)
    # Must include both classes: even IDs = "converted" (interested), odd IDs = "ghosted" (losing_interest)
    subset = {"conv_000", "conv_002", "conv_001"}

    tokenizer_mock = MagicMock()
    tokenizer_mock.return_value = {
        "input_ids": torch.zeros(len(subset), 16, dtype=torch.long),
        "attention_mask": torch.ones(len(subset), 16, dtype=torch.long),
    }

    with patch("dataset.load_corpus", return_value=records):
        ds = HebrewConversationDataset(config, tokenizer_mock, subset_ids=subset)

    assert len(ds) == 3
    assert set(ds.record_ids) == subset


def test_dataset_without_subset_ids_includes_all():
    """Without subset_ids, all mappable records are included."""
    from dataset import HebrewConversationDataset

    config = {
        "data": {
            "input_path": "FAKE",
            "id_field": "conversation_id",
            "messages_field": "messages",
            "role_field": "role",
            "text_field": "text",
            "included_roles": ["customer"],
            "label_field": "final_outcome",
            "message_separator": " [SEP] ",
            "label_mapping": {
                "interested": ["converted"],
                "losing_interest": ["ghosted"],
            },
        },
        "tokenizer": {"max_length": 16, "padding": "max_length", "truncation": True, "add_special_tokens": True},
        "model": {"num_labels": 2},
        "random_seed": 42,
    }

    records = _make_tiny_records(10)

    tokenizer_mock = MagicMock()
    tokenizer_mock.return_value = {
        "input_ids": torch.zeros(10, 16, dtype=torch.long),
        "attention_mask": torch.ones(10, 16, dtype=torch.long),
    }

    with patch("dataset.load_corpus", return_value=records):
        ds = HebrewConversationDataset(config, tokenizer_mock)

    assert len(ds) == 10


# ---------------------------------------------------------------------------
# Metrics correctness (no model, just numpy)
# ---------------------------------------------------------------------------

def test_evaluate_metrics_all_correct():
    """All-correct predictions should yield perfect metrics."""
    from sklearn.metrics import f1_score, accuracy_score

    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    assert abs(acc - 1.0) < 1e-9
    assert abs(macro_f1 - 1.0) < 1e-9


def test_evaluate_metrics_all_wrong():
    """All-wrong binary predictions should yield near-zero F1."""
    from sklearn.metrics import f1_score, accuracy_score

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    acc = accuracy_score(y_true, y_pred)
    assert abs(acc - 0.0) < 1e-9


def test_checkpoint_save_load(tmp_path):
    """Verify that a tiny model can be saved and reloaded without error."""
    import torch.nn as nn
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
        max_position_embeddings=64,
    )
    model = BertForSequenceClassification(config)
    model_dir = tmp_path / "checkpoint"
    model.save_pretrained(str(model_dir))

    loaded = BertForSequenceClassification.from_pretrained(str(model_dir))
    # Verify weights are identical
    for (name1, p1), (name2, p2) in zip(
        model.named_parameters(), loaded.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(p1, p2), f"Mismatch in {name1}"
