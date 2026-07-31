"""Tests for checkpoint save/restore."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint_manager import (
    Checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_generated,
    load_generated,
    save_combined,
)


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(
        last_accepted_id="he_sales_000305",
        requested_count=20,
        accepted_count=5,
        rejected_count=2,
        retry_count=3,
        random_seed=42,
        used_plan_signatures=["domain:product:traj"],
        outcome_distribution={"converted": 3, "ghosted": 2},
        domain_distribution={"furniture": 5},
    )
    path = tmp_path / "checkpoint.json"
    save_checkpoint(cp, path)
    loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.accepted_count == 5
    assert loaded.last_accepted_id == "he_sales_000305"
    assert loaded.random_seed == 42
    assert "domain:product:traj" in loaded.used_plan_signatures
    assert loaded.outcome_distribution["converted"] == 3


def test_load_checkpoint_missing(tmp_path):
    result = load_checkpoint(tmp_path / "nonexistent.json")
    assert result is None


def test_save_checkpoint_atomic(tmp_path):
    cp = Checkpoint(accepted_count=10)
    path = tmp_path / "cp.json"
    save_checkpoint(cp, path)
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    assert data["accepted_count"] == 10


def test_generated_roundtrip(tmp_path):
    convs = [{"conversation_id": "he_sales_000301", "domain": "test"}]
    path = tmp_path / "generated.json"
    save_generated(convs, path)
    loaded = load_generated(path)
    assert len(loaded) == 1
    assert loaded[0]["conversation_id"] == "he_sales_000301"


def test_load_generated_missing(tmp_path):
    result = load_generated(tmp_path / "nonexistent.json")
    assert result == []


def test_save_combined(tmp_path):
    source = [{"conversation_id": "he_sales_000001"}]
    generated = [{"conversation_id": "he_sales_000301"}]
    path = tmp_path / "combined.json"
    save_combined(source, generated, path)
    with path.open() as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["conversation_id"] == "he_sales_000001"
    assert data[1]["conversation_id"] == "he_sales_000301"
