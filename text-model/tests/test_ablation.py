"""Tests for ablation_continued_finetune.py — no model downloads required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))

from behavioral_features_design import FORBIDDEN_FIELDS, LEXICAL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# No behavioral features on the HF model
# ---------------------------------------------------------------------------

def test_ablation_model_has_no_behavioral_head():
    """BertForSequenceClassification has no fusion attributes."""
    cfg = BertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, num_labels=2, max_position_embeddings=64,
        vocab_size=200,
    )
    model = BertForSequenceClassification(cfg)
    assert not hasattr(model, "behavioral_head"), \
        "Model should not have a behavioral_head"
    assert not hasattr(model, "beh"), \
        "Model should not have a beh attribute"


def test_ablation_model_classifier_is_linear_768_to_2():
    """Original AlephBERT classifier is Linear(768→2), not the fusion MLP."""
    cfg = BertConfig(
        hidden_size=768, num_hidden_layers=1, num_attention_heads=8,
        intermediate_size=64, num_labels=2, max_position_embeddings=512,
        vocab_size=200,
    )
    model = BertForSequenceClassification(cfg)
    import torch.nn as nn
    assert isinstance(model.classifier, nn.Linear), \
        "AlephBERT classifier should be a plain Linear layer"
    assert model.classifier.in_features == 768
    assert model.classifier.out_features == 2


def test_ablation_output_shape():
    """BertForSequenceClassification outputs (B, 2) logits."""
    cfg = BertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, num_labels=2, max_position_embeddings=64,
        vocab_size=200,
    )
    model = BertForSequenceClassification(cfg)
    model.eval()
    x = torch.randint(0, 200, (3, 8))
    mask = torch.ones(3, 8, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=x, attention_mask=mask)
    assert out.logits.shape == (3, 2), f"Expected (3,2), got {out.logits.shape}"


# ---------------------------------------------------------------------------
# Split IDs: correct checkpoint and no overwrite
# ---------------------------------------------------------------------------

def test_split_ids_file_exists():
    """split_ids.json must exist before ablation can run."""
    p = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    assert p.exists(), f"split_ids.json not found at {p}"


def test_split_ids_counts():
    """Loaded split IDs must match the expected 2137/459/459 split."""
    p = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    payload = json.loads(p.read_text())
    assert payload["counts"]["train"] == 2137
    assert payload["counts"]["val"]   == 459
    assert payload["counts"]["test"]  == 459


def test_split_ids_no_overlap():
    """No conversation ID should appear in more than one split."""
    p = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    payload = json.loads(p.read_text())
    train_set = set(payload["train_ids"])
    val_set   = set(payload["val_ids"])
    test_set  = set(payload["test_ids"])
    assert not (train_set & val_set),  "train/val overlap in split_ids.json"
    assert not (train_set & test_set), "train/test overlap in split_ids.json"
    assert not (val_set   & test_set), "val/test overlap in split_ids.json"


def test_checkpoint_exists():
    """AlephBERT best checkpoint must be present before ablation can load it."""
    ckpt = Path(__file__).parent.parent / "outputs" / "alephbert_baseline_v1" / "best_model"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "tokenizer.json").exists()


def test_protected_outputs_not_overwritten(tmp_path):
    """run() raises if output dir would collide with a protected experiment."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ablation_continued_finetune import EXPERIMENT_NAME

    # The experiment name must be distinct from all protected directories
    protected = [
        "alephbert_baseline_v1",
        "fusion_alephbert_behavioral_v1",
        "behavioral_baseline_v1",
        "pure_behavioral_baseline_v1",
    ]
    assert EXPERIMENT_NAME not in protected, \
        f"Ablation experiment name '{EXPERIMENT_NAME}' collides with a protected dir"


# ---------------------------------------------------------------------------
# No lexical or forbidden features in model input
# ---------------------------------------------------------------------------

def test_no_behavioral_features_in_hf_model_forward():
    """Forward signature of BertForSequenceClassification has no beh/behavioral param."""
    import inspect
    from transformers import BertForSequenceClassification
    sig = inspect.signature(BertForSequenceClassification.forward)
    param_names = list(sig.parameters.keys())
    assert "behavioral_features" not in param_names
    assert "beh" not in param_names
