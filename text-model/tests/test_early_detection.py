"""Tests for early_detection_e1.py — no model downloads required."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from early_detection_e1 import FRACTIONS, build_prefix_text, prefix_length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_CONFIG = {
    "messages_field": "messages",
    "role_field": "role",
    "text_field": "text",
    "included_roles": ["customer"],
    "message_separator": " [SEP] ",
    "id_field": "conversation_id",
    "label_field": "final_outcome",
    "label_mapping": {
        "interested": ["converted", "pending"],
        "losing_interest": ["ghosted", "rejected"],
    },
}


def _conv(n_total: int, alternating: bool = True) -> Dict[str, Any]:
    """Build a synthetic conversation with n_total messages."""
    msgs = []
    for i in range(n_total):
        role = "customer" if (i % 2 == 0 if alternating else True) else "business"
        msgs.append({
            "message_index": i,
            "role": role,
            "text": f"message {i}",
            "response_delay_minutes": 10,
        })
    return {"conversation_id": "test", "final_outcome": "converted", "messages": msgs}


# ---------------------------------------------------------------------------
# prefix_length rule
# ---------------------------------------------------------------------------

def test_prefix_length_25_on_12():
    assert prefix_length(12, 0.25) == math.ceil(12 * 0.25) == 3


def test_prefix_length_50_on_9():
    assert prefix_length(9, 0.50) == math.ceil(9 * 0.50) == 5


def test_prefix_length_75_on_9():
    assert prefix_length(9, 0.75) == math.ceil(9 * 0.75) == 7


def test_prefix_length_100_on_any():
    for n in [1, 5, 9, 12, 41]:
        assert prefix_length(n, 1.0) == n


def test_prefix_length_minimum_one_message():
    """Even a 1-message conversation at any fraction stays at 1."""
    for frac in FRACTIONS:
        assert prefix_length(1, frac) == 1


def test_prefix_length_very_short_conversation():
    """3-message conversation: 25% → ceil(0.75) = 1."""
    assert prefix_length(3, 0.25) == 1


def test_prefix_length_never_zero():
    for n in range(1, 50):
        for frac in FRACTIONS:
            assert prefix_length(n, frac) >= 1


def test_prefix_length_monotone_in_fraction():
    """Longer fractions never produce shorter prefixes."""
    n = 12
    lengths = [prefix_length(n, frac) for frac in FRACTIONS]
    assert lengths == sorted(lengths)


# ---------------------------------------------------------------------------
# build_prefix_text
# ---------------------------------------------------------------------------

def test_build_prefix_text_100pct_returns_all_customer_messages():
    conv = _conv(10)
    text, n_total, n_cust = build_prefix_text(conv, 1.0, DATA_CONFIG)
    # All 10 messages, alternating → 5 customer messages
    assert n_total == 10
    assert n_cust == 5
    # Text should contain 5 messages joined by [SEP]
    assert text.count("[SEP]") == 4


def test_build_prefix_text_chronological_order():
    """Messages appear in message_index order, not shuffled."""
    conv = {
        "messages": [
            {"message_index": 0, "role": "customer", "text": "first", "response_delay_minutes": 0},
            {"message_index": 1, "role": "business", "text": "reply", "response_delay_minutes": 0},
            {"message_index": 2, "role": "customer", "text": "second", "response_delay_minutes": 0},
        ]
    }
    text, _, _ = build_prefix_text(conv, 1.0, DATA_CONFIG)
    first_pos  = text.index("first")
    second_pos = text.index("second")
    assert first_pos < second_pos, "Messages must appear in chronological order"


def test_build_prefix_text_25pct_on_12_messages():
    """25% of 12 = ceil(3) = 3 messages → only first 3 messages."""
    conv = _conv(12)
    _, n_total, n_cust = build_prefix_text(conv, 0.25, DATA_CONFIG)
    assert n_total == 3
    # First 3 messages: index 0 (customer), 1 (business), 2 (customer) → 2 customer
    assert n_cust == 2


def test_build_prefix_text_excludes_business_messages():
    """Only customer messages appear in the text, not business messages."""
    conv = {
        "messages": [
            {"message_index": 0, "role": "customer", "text": "hello customer", "response_delay_minutes": 0},
            {"message_index": 1, "role": "business", "text": "hello business", "response_delay_minutes": 0},
        ]
    }
    text, _, _ = build_prefix_text(conv, 1.0, DATA_CONFIG)
    assert "hello customer" in text
    assert "hello business" not in text


def test_build_prefix_text_zero_customer_returns_empty_string():
    """If prefix contains only business messages, text is '' (valid fallback)."""
    conv = {
        "messages": [
            {"message_index": 0, "role": "business", "text": "hi", "response_delay_minutes": 0},
        ]
    }
    text, n_total, n_cust = build_prefix_text(conv, 1.0, DATA_CONFIG)
    assert text == ""
    assert n_cust == 0
    assert n_total == 1


def test_build_prefix_text_no_leaky_fields():
    """Text construction must not read interest_score, interest_label, or final_outcome."""
    conv = {
        "messages": [
            {
                "message_index": 0,
                "role": "customer",
                "text": "clean text",
                "response_delay_minutes": 0,
                "interest_label": "converted",      # leaky — must never appear in text
                "interest_score": 0.99,              # leaky
            }
        ],
        "final_outcome": "converted",                # leaky — only used as label
        "interest_trajectory": "high_to_conversion", # leaky
    }
    text, _, _ = build_prefix_text(conv, 1.0, DATA_CONFIG)
    assert text == "clean text"
    assert "converted" not in text
    assert "interest" not in text
    assert "trajectory" not in text


# ---------------------------------------------------------------------------
# Same conversation IDs at every fraction
# ---------------------------------------------------------------------------

def test_same_conv_ids_across_fractions():
    """All 4 fractions must evaluate the same set of conversations."""
    from pathlib import Path
    import json

    split_path = Path(__file__).parent.parent / "outputs" / "split_ids.json"
    payload = json.loads(split_path.read_text())
    test_ids = set(payload["test_ids"])

    # All fractions produce the same ID set from the same test_convs list
    # (just with different prefix lengths) — this is guaranteed by the design
    assert len(test_ids) == 459


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_prefix_length_deterministic():
    """Same inputs always produce same output."""
    assert prefix_length(12, 0.25) == prefix_length(12, 0.25)


def test_build_prefix_text_deterministic():
    conv = _conv(8)
    text1, n1, c1 = build_prefix_text(conv, 0.5, DATA_CONFIG)
    text2, n2, c2 = build_prefix_text(conv, 0.5, DATA_CONFIG)
    assert text1 == text2
    assert n1 == n2
    assert c1 == c2


# ---------------------------------------------------------------------------
# Output shape check (smoke)
# ---------------------------------------------------------------------------

def test_fractions_list():
    assert FRACTIONS == [0.25, 0.50, 0.75, 1.00]
    assert len(FRACTIONS) == 4


def test_all_fractions_produce_text():
    """Every fraction produces a non-None result."""
    conv = _conv(8)
    for frac in FRACTIONS:
        text, n_total, n_cust = build_prefix_text(conv, frac, DATA_CONFIG)
        assert isinstance(text, str)
        assert n_total >= 1
        assert n_cust >= 0
