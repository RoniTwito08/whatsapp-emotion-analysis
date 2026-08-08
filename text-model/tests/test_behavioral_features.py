"""Tests for behavioral feature extraction — no model downloads required."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from behavioral_features_design import (
    FORBIDDEN_FIELDS,
    extract_behavioral_features,
    feature_count,
    feature_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conversation(
    outcomes: List[Dict[str, Any]] = None,
    n_customer: int = 3,
    n_business: int = 2,
) -> Dict[str, Any]:
    """Build a minimal synthetic conversation for testing.

    Creates n_customer customer messages followed by n_business business messages,
    so counts are exact and predictable.
    """
    messages = []
    for i in range(n_customer):
        messages.append({
            "role": "customer",
            "text": f"customer message {i}",
            "response_delay_minutes": 30.0,
            "behavioral_features": {
                "message_length_chars": 20 + i,
                "question_count": 1 if i == 0 else 0,
                "emoji_count": 0,
                "contains_price": False,
                "contains_negotiation": False,
                "contains_objection": False,
                "contains_commitment": False,
                "contains_delay_signal": False,
            },
        })
    for i in range(n_business):
        messages.append({
            "role": "business",
            "text": f"business message {i}",
            "response_delay_minutes": 15.0,
            "behavioral_features": {},
        })
    return {"messages": messages, "conversation_id": "test_conv_001"}


def _conv_with_flags(**flags) -> Dict[str, Any]:
    """Create a 2-message conversation with specified flags on the customer message."""
    return {
        "messages": [
            {
                "role": "customer",
                "text": "customer text",
                "response_delay_minutes": 10.0,
                "behavioral_features": {
                    "message_length_chars": 50,
                    "question_count": flags.get("question_count", 0),
                    "emoji_count": flags.get("emoji_count", 0),
                    "contains_price": flags.get("contains_price", False),
                    "contains_negotiation": flags.get("contains_negotiation", False),
                    "contains_objection": flags.get("contains_objection", False),
                    "contains_commitment": flags.get("contains_commitment", False),
                    "contains_delay_signal": flags.get("contains_delay_signal", False),
                },
            },
            {
                "role": "business",
                "text": "business text",
                "response_delay_minutes": 5.0,
                "behavioral_features": {},
            },
        ]
    }


# ---------------------------------------------------------------------------
# Feature name and count checks
# ---------------------------------------------------------------------------

def test_feature_names_returns_nonempty_list():
    names = feature_names()
    assert isinstance(names, list)
    assert len(names) > 0


def test_feature_count_matches_feature_names():
    assert feature_count() == len(feature_names())


def test_feature_names_are_unique():
    names = feature_names()
    assert len(names) == len(set(names)), "Duplicate feature names"


def test_no_forbidden_fields_in_feature_names():
    names = set(feature_names())
    overlap = names & FORBIDDEN_FIELDS
    assert not overlap, f"Forbidden fields found in feature names: {overlap}"


def test_feature_count_at_least_30():
    assert feature_count() >= 30, "Expected at least 30 features"


# ---------------------------------------------------------------------------
# Feature extraction correctness
# ---------------------------------------------------------------------------

def test_extract_returns_correct_keys():
    conv = _make_conversation()
    feats = extract_behavioral_features(conv)
    expected = set(feature_names())
    assert set(feats.keys()) == expected


def test_all_feature_values_are_float():
    conv = _make_conversation()
    feats = extract_behavioral_features(conv)
    for name, val in feats.items():
        assert isinstance(val, float), f"Feature '{name}' is not float: {type(val)}"


def test_message_counts():
    conv = _make_conversation(n_customer=4, n_business=3)
    feats = extract_behavioral_features(conv)
    assert feats["customer_messages"] == 4.0
    assert feats["business_messages"] == 3.0
    assert feats["total_messages"] == 7.0


def test_ratio_customer_to_business():
    conv = _make_conversation(n_customer=4, n_business=2)
    feats = extract_behavioral_features(conv)
    assert abs(feats["ratio_customer_to_business"] - 2.0) < 1e-9


def test_commitment_flag_captured():
    conv = _conv_with_flags(contains_commitment=True)
    feats = extract_behavioral_features(conv)
    assert feats["customer_commitment_mentions"] == 1.0
    assert feats["last_customer_has_commitment"] == 1.0
    # commitment_in_second_half depends on the split point; just check the count
    assert feats["customer_commitment_mentions"] >= 1.0


def test_objection_flag_captured():
    conv = _conv_with_flags(contains_objection=True)
    feats = extract_behavioral_features(conv)
    assert feats["customer_objection_mentions"] == 1.0
    assert feats["last_customer_has_objection"] == 1.0


def test_no_flags_all_zero():
    conv = _conv_with_flags()
    feats = extract_behavioral_features(conv)
    for key in ["customer_commitment_mentions", "customer_objection_mentions",
                "customer_negotiation_mentions", "customer_delay_signals"]:
        assert feats[key] == 0.0, f"{key} should be 0"


def test_question_rate():
    conv = _conv_with_flags(question_count=2)
    feats = extract_behavioral_features(conv)
    assert feats["total_customer_questions"] == 2.0
    assert abs(feats["question_rate"] - 2.0) < 1e-9  # 2 questions / 1 customer msg


def test_empty_messages_returns_empty_dict():
    feats = extract_behavioral_features({"messages": []})
    assert feats == {}


def test_missing_messages_field_returns_empty_dict():
    feats = extract_behavioral_features({"conversation_id": "abc"})
    assert feats == {}


def test_session_count_single_session():
    """Messages with ≤60 min gaps → 1 session."""
    conv = {
        "messages": [
            {"role": "customer", "response_delay_minutes": 30, "behavioral_features": {}},
            {"role": "business", "response_delay_minutes": 15, "behavioral_features": {}},
        ]
    }
    feats = extract_behavioral_features(conv)
    assert feats["session_count"] == 1.0


def test_session_count_two_sessions():
    """A gap >60 min creates a second session."""
    conv = {
        "messages": [
            {"role": "customer", "response_delay_minutes": 5, "behavioral_features": {}},
            {"role": "business", "response_delay_minutes": 90, "behavioral_features": {}},  # >60 → new session
            {"role": "customer", "response_delay_minutes": 5, "behavioral_features": {}},
        ]
    }
    feats = extract_behavioral_features(conv)
    assert feats["session_count"] == 2.0


def test_last_role_is_customer_true():
    conv = {
        "messages": [
            {"role": "business", "response_delay_minutes": 0, "behavioral_features": {}},
            {"role": "customer", "response_delay_minutes": 10,
             "behavioral_features": {"message_length_chars": 20, "question_count": 0,
                                      "emoji_count": 0, "contains_price": False,
                                      "contains_negotiation": False, "contains_objection": False,
                                      "contains_commitment": False, "contains_delay_signal": False}},
        ]
    }
    feats = extract_behavioral_features(conv)
    assert feats["last_role_is_customer"] == 1.0


def test_last_role_is_customer_false():
    conv = {
        "messages": [
            {"role": "customer", "response_delay_minutes": 10,
             "behavioral_features": {"message_length_chars": 20, "question_count": 0,
                                      "emoji_count": 0, "contains_price": False,
                                      "contains_negotiation": False, "contains_objection": False,
                                      "contains_commitment": False, "contains_delay_signal": False}},
            {"role": "business", "response_delay_minutes": 5, "behavioral_features": {}},
        ]
    }
    feats = extract_behavioral_features(conv)
    assert feats["last_role_is_customer"] == 0.0


# ---------------------------------------------------------------------------
# Feature matrix shape and split ID usage
# ---------------------------------------------------------------------------

def test_build_feature_matrix_shape():
    """build_feature_matrix returns X with shape (n_conversations, n_features)."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from behavioral_baseline import build_feature_matrix

    label_to_id = {"interested": 0, "losing_interest": 1}
    label_mapping = {
        "interested": ["converted"],
        "losing_interest": ["ghosted"],
    }

    def _make_rec(cid: str, outcome: str) -> Dict[str, Any]:
        return {
            "conversation_id": cid,
            "final_outcome": outcome,
            "messages": [
                {
                    "role": "customer",
                    "text": "hello",
                    "response_delay_minutes": 10,
                    "behavioral_features": {
                        "message_length_chars": 30, "question_count": 1,
                        "emoji_count": 0, "contains_price": False,
                        "contains_negotiation": False, "contains_objection": False,
                        "contains_commitment": False, "contains_delay_signal": False,
                    },
                },
                {
                    "role": "business",
                    "text": "hi",
                    "response_delay_minutes": 5,
                    "behavioral_features": {},
                },
            ],
        }

    records = [_make_rec(f"conv_{i}", "converted" if i % 2 == 0 else "ghosted") for i in range(10)]
    id_to_label = {f"conv_{i}": "interested" if i % 2 == 0 else "losing_interest" for i in range(10)}
    split_ids = [f"conv_{i}" for i in range(10)]

    config = {
        "data": {
            "id_field": "conversation_id",
            "label_field": "final_outcome",
            "label_mapping": label_mapping,
        }
    }

    X, y, kept_ids = build_feature_matrix(records, id_to_label, split_ids, config, label_to_id)

    assert X.shape == (10, feature_count())
    assert y.shape == (10,)
    assert len(kept_ids) == 10


def test_build_feature_matrix_uses_only_split_ids():
    """Only conversations in split_ids should appear in output."""
    from behavioral_baseline import build_feature_matrix

    label_to_id = {"interested": 0, "losing_interest": 1}
    config = {"data": {"id_field": "conversation_id", "label_field": "final_outcome",
                        "label_mapping": {"interested": ["converted"], "losing_interest": ["ghosted"]}}}

    records = [
        {"conversation_id": f"conv_{i}", "final_outcome": "converted" if i % 2 == 0 else "ghosted",
         "messages": [{"role": "customer", "response_delay_minutes": 5,
                       "behavioral_features": {"message_length_chars": 10, "question_count": 0,
                                               "emoji_count": 0, "contains_price": False,
                                               "contains_negotiation": False, "contains_objection": False,
                                               "contains_commitment": False, "contains_delay_signal": False}}]}
        for i in range(20)
    ]
    id_to_label = {f"conv_{i}": "interested" if i % 2 == 0 else "losing_interest" for i in range(20)}
    subset = [f"conv_{i}" for i in range(10)]  # only first 10

    X, y, kept_ids = build_feature_matrix(records, id_to_label, subset, config, label_to_id)

    assert len(kept_ids) == 10
    for cid in kept_ids:
        assert int(cid.split("_")[1]) < 10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    from behavioral_baseline import compute_metrics
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_pred, positive_class_id=1)
    assert abs(metrics["accuracy"] - 1.0) < 1e-9
    assert abs(metrics["macro_f1"] - 1.0) < 1e-9
    assert abs(metrics["f1"] - 1.0) < 1e-9


def test_compute_metrics_all_wrong():
    from behavioral_baseline import compute_metrics
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred, positive_class_id=1)
    assert abs(metrics["accuracy"] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Pure behavioral feature set
# ---------------------------------------------------------------------------

def test_pure_behavioral_has_no_lexical_features():
    from behavioral_features_design import pure_behavioral_feature_names, LEXICAL_FEATURE_NAMES
    pure = set(pure_behavioral_feature_names())
    overlap = pure & LEXICAL_FEATURE_NAMES
    assert not overlap, f"Lexical features found in pure set: {overlap}"


def test_pure_behavioral_has_no_forbidden_fields():
    from behavioral_features_design import pure_behavioral_feature_names, FORBIDDEN_FIELDS
    pure = set(pure_behavioral_feature_names())
    overlap = pure & FORBIDDEN_FIELDS
    assert not overlap, f"Forbidden fields found in pure set: {overlap}"


def test_pure_behavioral_is_subset_of_all_features():
    from behavioral_features_design import pure_behavioral_feature_names
    all_f = set(feature_names())
    pure = set(pure_behavioral_feature_names())
    assert pure.issubset(all_f), "Pure features not a subset of all features"


def test_pure_behavioral_includes_structural_change_features():
    from behavioral_features_design import pure_behavioral_feature_names
    pure = pure_behavioral_feature_names()
    required = [
        "first_half_mean_customer_length",
        "second_half_mean_customer_length",
        "customer_length_change",
        "first_half_mean_response_delay",
        "second_half_mean_response_delay",
        "response_delay_change",
        "first_half_customer_message_count",
        "second_half_customer_message_count",
    ]
    for feat in required:
        assert feat in pure, f"Missing structural change feature: {feat}"


def test_build_feature_matrix_with_pure_features():
    """build_feature_matrix respects explicit feat_names — produces correct column count."""
    from behavioral_baseline import build_feature_matrix
    from behavioral_features_design import pure_behavioral_feature_names

    pure = pure_behavioral_feature_names()
    label_to_id = {"interested": 0, "losing_interest": 1}
    config = {"data": {"id_field": "conversation_id", "label_field": "final_outcome",
                        "label_mapping": {"interested": ["converted"],
                                          "losing_interest": ["ghosted"]}}}
    records = [
        {"conversation_id": f"c{i}", "final_outcome": "converted" if i % 2 == 0 else "ghosted",
         "messages": [
             {"role": "customer", "response_delay_minutes": 10,
              "behavioral_features": {"message_length_chars": 30, "question_count": 0,
                                       "emoji_count": 0, "contains_price": False,
                                       "contains_negotiation": False, "contains_objection": False,
                                       "contains_commitment": False, "contains_delay_signal": False}},
             {"role": "business", "response_delay_minutes": 5, "behavioral_features": {}},
         ]}
        for i in range(10)
    ]
    id_to_label = {f"c{i}": "interested" if i % 2 == 0 else "losing_interest" for i in range(10)}
    split_ids = [f"c{i}" for i in range(10)]

    X, y, kept = build_feature_matrix(records, id_to_label, split_ids, config, label_to_id, pure)
    assert X.shape == (10, len(pure)), f"Expected ({10}, {len(pure)}), got {X.shape}"
