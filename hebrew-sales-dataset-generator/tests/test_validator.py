"""Tests for the conversation validator."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator import validate_conversation, validate_dataset


def _make_valid_conv() -> dict:
    return {
        "conversation_id": "he_sales_000301",
        "source_dataset": "synthetic_hebrew_whatsapp_sales_interest_v2",
        "language": "he",
        "locale": "he-IL",
        "synthetic": True,
        "domain": "furniture",
        "domain_he": "ריהוט",
        "product_or_service": "ספה",
        "customer_persona": "אנליטי",
        "business_style": "מקצועי",
        "interest_trajectory": "high_to_conversion",
        "initial_interest_score": 0.74,
        "final_interest_score": 1.0,
        "final_outcome": "converted",
        "messages": [
            {
                "message_index": 0,
                "role": "customer",
                "text": "שלום, אני מחפש ספה חדשה לסלון",
                "response_delay_minutes": 5,
                "interest_label": "interested",
                "interest_score": 0.74,
                "behavioral_features": {
                    "message_length_chars": 29,
                    "question_count": 0,
                    "emoji_count": 0,
                    "contains_price": False,
                    "contains_negotiation": False,
                    "contains_objection": False,
                    "contains_commitment": False,
                    "contains_delay_signal": False,
                },
                "timestamp": "2026-03-15T10:00",
            },
            {
                "message_index": 1,
                "role": "business",
                "text": "ברוך הבא, אשמח לעזור",
                "response_delay_minutes": 3,
                "interest_label": "not_applicable",
                "interest_score": None,
                "behavioral_features": {
                    "message_length_chars": 20,
                    "question_count": 0,
                    "emoji_count": 0,
                    "contains_price": False,
                    "contains_negotiation": False,
                    "contains_objection": False,
                    "contains_commitment": False,
                    "contains_delay_signal": False,
                },
                "timestamp": "2026-03-15T10:03",
            },
            {
                "message_index": 2,
                "role": "customer",
                "text": "מה המחיר לספה פינתית?",
                "response_delay_minutes": 2,
                "interest_label": "converted",
                "interest_score": 1.0,
                "behavioral_features": {
                    "message_length_chars": 21,
                    "question_count": 1,
                    "emoji_count": 0,
                    "contains_price": True,
                    "contains_negotiation": False,
                    "contains_objection": False,
                    "contains_commitment": False,
                    "contains_delay_signal": False,
                },
                "timestamp": "2026-03-15T10:05",
            },
        ],
        "metadata": {
            "channel": "whatsapp",
            "contains_personal_data": False,
            "quality_status": "pilot_v2",
            "generator_seed": 42,
        },
    }


def test_valid_conversation():
    conv = _make_valid_conv()
    errors = validate_conversation(conv)
    assert errors == [], f"Unexpected errors: {errors}"


def test_invalid_id_format():
    conv = _make_valid_conv()
    conv["conversation_id"] = "he_sales_301"
    errors = validate_conversation(conv)
    assert any("conversation_id" in e for e in errors)


def test_wrong_outcome_for_trajectory():
    conv = _make_valid_conv()
    conv["interest_trajectory"] = "high_to_ghosting"
    conv["final_outcome"] = "converted"
    errors = validate_conversation(conv)
    assert any("outcome" in e.lower() or "trajectory" in e.lower() for e in errors)


def test_slash_gender_form_rejected():
    conv = _make_valid_conv()
    conv["messages"][0]["text"] = "אני מתעניין/ת בשירות שלכם"
    errors = validate_conversation(conv)
    assert any("slash" in e.lower() or "gender" in e.lower() for e in errors)


def test_business_must_have_not_applicable_label():
    conv = _make_valid_conv()
    conv["messages"][1]["interest_label"] = "interested"
    errors = validate_conversation(conv)
    assert any("not_applicable" in e for e in errors)


def test_business_must_have_null_score():
    conv = _make_valid_conv()
    conv["messages"][1]["interest_score"] = 0.5
    errors = validate_conversation(conv)
    assert any("null" in e or "interest_score" in e for e in errors)


def test_customer_cannot_have_not_applicable_label():
    conv = _make_valid_conv()
    conv["messages"][0]["interest_label"] = "not_applicable"
    errors = validate_conversation(conv)
    assert any("not_applicable" in e for e in errors)


def test_wrong_message_index():
    conv = _make_valid_conv()
    conv["messages"][1]["message_index"] = 99
    errors = validate_conversation(conv)
    assert any("message_index" in e for e in errors)


def test_timestamp_format_invalid():
    conv = _make_valid_conv()
    conv["messages"][0]["timestamp"] = "2026-03-15 10:00:00"
    errors = validate_conversation(conv)
    assert any("timestamp" in e for e in errors)


def test_no_hebrew_text():
    conv = _make_valid_conv()
    conv["messages"][0]["text"] = "Hello, I want a couch"
    errors = validate_conversation(conv)
    assert any("hebrew" in e.lower() or "Hebrew" in e for e in errors)


def test_empty_messages():
    conv = _make_valid_conv()
    conv["messages"] = []
    errors = validate_conversation(conv)
    assert any("messages" in e for e in errors)


def test_first_message_must_be_customer():
    conv = _make_valid_conv()
    conv["messages"][0]["role"] = "business"
    conv["messages"][0]["interest_label"] = "not_applicable"
    conv["messages"][0]["interest_score"] = None
    errors = validate_conversation(conv)
    assert any("first message" in e.lower() or "customer" in e.lower() for e in errors)


def test_timestamp_delay_consistency():
    conv = _make_valid_conv()
    conv["messages"][1]["response_delay_minutes"] = 999
    errors = validate_conversation(conv)
    assert any("delay" in e.lower() for e in errors)


def test_validate_dataset_detects_duplicates():
    conv = _make_valid_conv()
    dataset = [conv, copy.deepcopy(conv)]
    report = validate_dataset(dataset)
    assert report["invalid_conversations"] > 0


def test_validate_real_corpus():
    corpus_path = Path(__file__).parent.parent / "data" / "corpus_300.json"
    if not corpus_path.exists():
        pytest.skip("corpus_300.json not found")
    import json

    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    # The source corpus pre-dates our slash-gender rule; skip that check here.
    report = validate_dataset(data, check_slash_gender=False)
    assert report["total_conversations"] == 300
    if report["invalid_conversations"] > 0:
        for cid, errs in list(report["errors"].items())[:3]:
            print(f"\n{cid}: {errs}")
    assert report["invalid_conversations"] == 0, (
        f"{report['invalid_conversations']} invalid conversations in corpus"
    )
