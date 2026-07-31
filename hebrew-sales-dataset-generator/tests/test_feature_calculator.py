"""Tests for the behavioral feature calculator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_calculator import (
    calculate_behavioral_features,
    count_emojis,
    count_questions,
    detect_contains_commitment,
    detect_contains_delay_signal,
    detect_contains_negotiation,
    detect_contains_objection,
    detect_contains_price,
)


# ──────────────────────── question count ────────────────────────
def test_question_count_none():
    assert count_questions("שלום, מה נשמע") == 0


def test_question_count_one():
    assert count_questions("כמה זה עולה?") == 1


def test_question_count_multiple():
    assert count_questions("מה זה? למה? מתי?") == 3


# ──────────────────────── emoji count ────────────────────────
def test_emoji_count_none():
    assert count_emojis("שלום, מה נשמע") == 0


def test_emoji_count_one():
    assert count_emojis("היי 🙂 מה שלומך") == 1


def test_emoji_count_multiple():
    assert count_emojis("👍 תודה 🙏 מצוין") == 2


# ──────────────────────── contains_price ────────────────────────
def test_price_shekels_sign():
    assert detect_contains_price("עלות ₪1,200") is True


def test_price_shekel_symbol():
    assert detect_contains_price('350 ש"ח') is True


def test_price_shekel_gershayim():
    assert detect_contains_price("350 ש״ח") is True


def test_price_keyword_aleph():
    assert detect_contains_price("אלף שקל בלבד") is True


def test_price_keyword_machir():
    assert detect_contains_price("מה המחיר שלכם?") is True


def test_price_keyword_payment():
    assert detect_contains_price("אפשר לשלם בתשלומים?") is True


def test_price_false_positive_time():
    """Time like 9:00 should not trigger price."""
    assert detect_contains_price("נפגש ב-9:00 בבוקר") is False


def test_price_false_positive_phone():
    """Phone number should not trigger price."""
    assert detect_contains_price("תתקשר אלי ב-050-1234567") is False


def test_price_false_positive_date():
    """Date numbers should not trigger price."""
    assert detect_contains_price("ביום 15 לחודש") is False


# ──────────────────────── contains_negotiation ────────────────────────
def test_negotiation_discount_request():
    assert detect_contains_negotiation("אפשר לקבל הנחה?") is True


def test_negotiation_split_payment():
    assert detect_contains_negotiation("אפשר לפצל את התשלום?") is True


def test_negotiation_competitor_quote():
    assert detect_contains_negotiation("קיבלתי הצעה יותר זולה ממקום אחר") is True


def test_negotiation_false_positive_simple_price_question():
    """Asking what the price is not negotiation."""
    assert detect_contains_negotiation("כמה עולה השירות?") is False


def test_negotiation_false_positive_info():
    """General business info is not negotiation."""
    assert detect_contains_negotiation("אנחנו נותנים שירות מעולה") is False


# ──────────────────────── contains_objection ────────────────────────
def test_objection_expensive():
    assert detect_contains_objection("זה יקר לי מדי") is True


def test_objection_uncertain():
    assert detect_contains_objection("אני לא בטוח שזה בשבילי") is True


def test_objection_hesitation():
    assert detect_contains_objection("אני מהסס") is True


def test_objection_no_budget():
    assert detect_contains_objection("אין לי את התקציב לזה") is True


def test_objection_false_positive_positive_question():
    """Simple question is not an objection."""
    assert detect_contains_objection("מה כלול בחבילה?") is False


def test_objection_false_positive_confirmation():
    """Positive confirmation is not an objection."""
    assert detect_contains_objection("מצוין, ממש התרשמתי") is False


# ──────────────────────── contains_commitment ────────────────────────
def test_commitment_close_deal():
    assert detect_contains_commitment("בוא נסגור, אני מוכן") is True


def test_commitment_set_appointment():
    assert detect_contains_commitment("קבענו פגישה ליום ראשון") is True


def test_commitment_send_documents():
    assert detect_contains_commitment("שולח לך את הפרטים עכשיו") is True


def test_commitment_agree():
    assert detect_contains_commitment("אני מסכים לתנאים") is True


def test_commitment_false_positive_vague():
    """Vague 'נראה' is not a commitment."""
    assert detect_contains_commitment("נראה מה יהיה") is False


def test_commitment_false_positive_business_offer():
    """Business offering an appointment is not a customer commitment."""
    assert detect_contains_commitment("אני יכול להציע לך פגישה") is False


# ──────────────────────── contains_delay_signal ────────────────────────
def test_delay_tomorrow():
    assert detect_contains_delay_signal("אחזור אליך מחר") is True


def test_delay_check():
    assert detect_contains_delay_signal("צריך לבדוק עם השותף שלי") is True


def test_delay_after_holiday():
    assert detect_contains_delay_signal("נדבר אחרי החג") is True


def test_delay_next_week():
    assert detect_contains_delay_signal("אדבר איתך בשבוע הבא") is True


def test_delay_give_days():
    assert detect_contains_delay_signal("תן לי כמה ימים לחשוב") is True


def test_delay_false_positive_wait_a_sec():
    """'רגע' meaning 'wait a moment' is not a delay signal."""
    assert detect_contains_delay_signal("רגע אחד בבקשה") is False


def test_delay_false_positive_now():
    """Immediate action is not a delay signal."""
    assert detect_contains_delay_signal("שולח לך עכשיו") is False


# ──────────────────────── full feature calculation ────────────────────────
def test_full_features():
    text = "היי 🙂 כמה עולה השירות? יש הנחה?"
    features = calculate_behavioral_features(text)
    assert features["message_length_chars"] == len(text)
    assert features["question_count"] == 2
    assert features["emoji_count"] == 1
    assert features["contains_price"] is True


def test_full_features_verification_against_corpus():
    """Spot-check feature calculation against known corpus messages."""
    corpus_path = Path(__file__).parent.parent / "data" / "corpus_300.json"
    if not corpus_path.exists():
        pytest.skip("corpus_300.json not found")

    import json

    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    checked = 0
    mismatches = []
    for conv in data[:10]:
        for msg in conv["messages"]:
            text = msg["text"]
            expected = msg["behavioral_features"]
            actual = calculate_behavioral_features(text)

            if actual["message_length_chars"] != expected["message_length_chars"]:
                mismatches.append(f"length: {text[:40]!r}")
            if actual["question_count"] != expected["question_count"]:
                mismatches.append(f"q_count: {text[:40]!r}")
            if actual["emoji_count"] != expected["emoji_count"]:
                mismatches.append(f"emoji: {text[:40]!r}")
            checked += 1

    assert checked > 0
    assert len(mismatches) == 0, f"Feature mismatches: {mismatches}"
