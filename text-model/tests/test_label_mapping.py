"""Tests for label mapping in dataset.py — no model downloads required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import DatasetError, build_label_to_id, invert_label_mapping


LABEL_MAPPING = {
    "interested": ["converted", "appointment_set", "pending", "reengaged_pending"],
    "losing_interest": ["explicit_rejection", "competitor_loss", "delivery_loss", "trust_loss", "ghosted"],
}


def test_build_label_to_id_deterministic():
    mapping = build_label_to_id(LABEL_MAPPING)
    assert mapping == {"interested": 0, "losing_interest": 1}


def test_build_label_to_id_alphabetical_order():
    mapping = build_label_to_id({"zebra": ["z"], "alpha": ["a"]})
    assert mapping == {"alpha": 0, "zebra": 1}


def test_invert_label_mapping_basic():
    inverted = invert_label_mapping(LABEL_MAPPING)
    assert inverted["converted"] == "interested"
    assert inverted["ghosted"] == "losing_interest"
    assert inverted["explicit_rejection"] == "losing_interest"
    assert inverted["pending"] == "interested"


def test_invert_label_mapping_covers_all_raw():
    inverted = invert_label_mapping(LABEL_MAPPING)
    all_raw = [r for raws in LABEL_MAPPING.values() for r in raws]
    for raw in all_raw:
        assert raw in inverted


def test_invert_label_mapping_duplicate_raises():
    bad_mapping = {
        "class_a": ["label_x"],
        "class_b": ["label_x"],  # duplicate raw label
    }
    with pytest.raises(DatasetError, match="more than one"):
        invert_label_mapping(bad_mapping)


def test_label_to_id_stable_across_runs():
    m1 = build_label_to_id(LABEL_MAPPING)
    m2 = build_label_to_id(LABEL_MAPPING)
    assert m1 == m2


def test_binary_label_ids():
    mapping = build_label_to_id(LABEL_MAPPING)
    assert set(mapping.values()) == {0, 1}
    assert len(mapping) == 2
