"""Tests for splitter.py — no model downloads required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from splitter import SplitError, _validate_no_overlap, stratified_split


LABEL_MAPPING = {
    "interested": ["converted", "pending"],
    "losing_interest": ["ghosted", "rejected"],
}


def _make_records(n_per_outcome: int = 20) -> list:
    records = []
    for i in range(n_per_outcome):
        records.append({"conversation_id": f"converted_{i}", "final_outcome": "converted"})
    for i in range(n_per_outcome):
        records.append({"conversation_id": f"pending_{i}", "final_outcome": "pending"})
    for i in range(n_per_outcome):
        records.append({"conversation_id": f"ghosted_{i}", "final_outcome": "ghosted"})
    for i in range(n_per_outcome):
        records.append({"conversation_id": f"rejected_{i}", "final_outcome": "rejected"})
    return records


def test_basic_split_sizes():
    records = _make_records(20)
    train, val, test = stratified_split(records, LABEL_MAPPING, seed=42)
    total = len(records)
    assert len(train) + len(val) + len(test) == total
    assert abs(len(train) / total - 0.70) < 0.05
    assert abs(len(val) / total - 0.15) < 0.05
    assert abs(len(test) / total - 0.15) < 0.05


def test_no_id_overlap():
    records = _make_records(20)
    train, val, test = stratified_split(records, LABEL_MAPPING, seed=42)
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    assert len(train_set & val_set) == 0, "train/val overlap"
    assert len(train_set & test_set) == 0, "train/test overlap"
    assert len(val_set & test_set) == 0, "val/test overlap"


def test_all_ids_assigned():
    records = _make_records(20)
    train, val, test = stratified_split(records, LABEL_MAPPING, seed=42)
    all_ids = set(train) | set(val) | set(test)
    expected = {r["conversation_id"] for r in records}
    assert all_ids == expected


def test_stratification_preserves_balance():
    records = _make_records(50)
    train, val, test = stratified_split(records, LABEL_MAPPING, seed=42)
    # Build id->label lookup
    id_to_label = {r["conversation_id"]: r["final_outcome"] for r in records}
    for split_ids, split_name in [(train, "train"), (val, "val"), (test, "test")]:
        outcomes = [id_to_label[i] for i in split_ids]
        interested = sum(1 for o in outcomes if o in ["converted", "pending"])
        losing = sum(1 for o in outcomes if o in ["ghosted", "rejected"])
        ratio = interested / (interested + losing)
        assert 0.40 <= ratio <= 0.60, f"{split_name} imbalanced: interested={ratio:.2f}"


def test_deterministic_with_same_seed():
    records = _make_records(30)
    train1, val1, test1 = stratified_split(records, LABEL_MAPPING, seed=42)
    train2, val2, test2 = stratified_split(records, LABEL_MAPPING, seed=42)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_different_seed_gives_different_split():
    records = _make_records(30)
    train1, _, _ = stratified_split(records, LABEL_MAPPING, seed=42)
    train2, _, _ = stratified_split(records, LABEL_MAPPING, seed=99)
    assert train1 != train2


def test_unmapped_labels_are_skipped():
    records = _make_records(20)
    records.append({"conversation_id": "unknown_0", "final_outcome": "unknown_outcome"})
    train, val, test = stratified_split(records, LABEL_MAPPING, seed=42)
    all_ids = set(train) | set(val) | set(test)
    assert "unknown_0" not in all_ids


def test_ratios_must_sum_to_one():
    records = _make_records(10)
    with pytest.raises(SplitError, match="sum to 1.0"):
        stratified_split(records, LABEL_MAPPING, train_ratio=0.6, val_ratio=0.2, test_ratio=0.3)


def test_too_few_records_raises():
    records = [{"conversation_id": "a", "final_outcome": "converted"}]
    with pytest.raises(SplitError):
        stratified_split(records, LABEL_MAPPING, seed=42)


def test_validate_no_overlap_raises_on_overlap():
    with pytest.raises(SplitError, match="overlap"):
        _validate_no_overlap(["a", "b", "c"], ["c", "d"], ["e"])
