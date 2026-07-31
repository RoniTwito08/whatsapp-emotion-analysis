"""Tests for the similarity checker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.similarity_checker import SimilarityChecker


def _make_conv(cid: str, texts: list[str]) -> dict:
    msgs = []
    for i, text in enumerate(texts):
        role = "customer" if i % 2 == 0 else "business"
        label = "interested" if role == "customer" else "not_applicable"
        msgs.append(
            {
                "message_index": i,
                "role": role,
                "text": text,
                "interest_label": label,
                "interest_score": 0.7 if role == "customer" else None,
            }
        )
    return {
        "conversation_id": cid,
        "domain": "furniture",
        "interest_trajectory": "high_to_conversion",
        "final_outcome": "converted",
        "messages": msgs,
    }


def test_exact_duplicate_detected():
    checker = SimilarityChecker()
    conv = _make_conv("he_sales_000001", ["אני מחפש ספה פינתית", "כמה זה עולה?"])
    checker.load_existing([conv])

    match = checker.check_exact_duplicate("אני מחפש ספה פינתית")
    assert match == "he_sales_000001"


def test_no_exact_duplicate_for_new_text():
    checker = SimilarityChecker()
    conv = _make_conv("he_sales_000001", ["שלום, יש לכם ספות?"])
    checker.load_existing([conv])

    match = checker.check_exact_duplicate("אני מחפש כורסא נוחה")
    assert match is None


def test_fuzzy_duplicate_similar_phrase():
    checker = SimilarityChecker(message_threshold=85)
    # Use near-identical phrases to ensure fuzzy matching fires
    conv = _make_conv(
        "he_sales_000001",
        ["יש לכם המלצות מלקוחות מהאזור שלי?"],
    )
    checker.load_existing([conv])

    score, matched = checker.check_fuzzy_duplicate("יש לכם המלצות מלקוחות מהאזור?")
    assert score > 80


def test_fuzzy_duplicate_completely_different():
    checker = SimilarityChecker(message_threshold=85)
    conv = _make_conv("he_sales_000001", ["שלום, יש לכם ספות?"])
    checker.load_existing([conv])

    score, matched = checker.check_fuzzy_duplicate("מה הטלפון שלכם?")
    assert score < 70


def test_short_messages_ignored():
    checker = SimilarityChecker()
    checker.load_existing([_make_conv("he_sales_000001", ["כן", "תודה"])])

    match = checker.check_exact_duplicate("כן")
    assert match is None


def test_check_new_conversation_no_issues():
    checker = SimilarityChecker()
    existing = _make_conv(
        "he_sales_000001",
        ["אני מחפש מזגן חדש", "יש מגוון רחב, מה ההספק שאתה צריך?"],
    )
    checker.load_existing([existing])

    new_conv = _make_conv(
        "he_sales_000002",
        ["שלום, מתעניין בריצוף לגינה", "יש לנו כמה אפשרויות מצוינות"],
    )
    issues = checker.check_new_conversation(new_conv)
    exact_issues = [i for i in issues if i[0] == "exact_duplicate"]
    assert len(exact_issues) == 0


def test_check_new_conversation_detects_exact():
    checker = SimilarityChecker()
    existing = _make_conv("he_sales_000001", ["שלום, אני מחפש ספה", "כמה עולה?"])
    checker.load_existing([existing])

    new_conv = _make_conv("he_sales_000002", ["שלום, אני מחפש ספה", "מה כלול?"])
    issues = checker.check_new_conversation(new_conv)
    assert any(i[0] == "exact_duplicate" for i in issues)


def test_accept_conversation_updates_index():
    checker = SimilarityChecker()
    conv = _make_conv("he_sales_000001", ["ספה חדשה לסלון, יש לכם?"])
    checker.accept_conversation(conv)

    match = checker.check_exact_duplicate("ספה חדשה לסלון, יש לכם?")
    assert match == "he_sales_000001"


def test_structure_signature():
    checker = SimilarityChecker()
    conv = _make_conv("he_sales_000001", ["a", "b", "c"])
    sig = checker.make_structure_signature(conv)
    assert "furniture" in sig
    assert "converted" in sig


def test_intra_conversation_duplicate_detected():
    checker = SimilarityChecker()
    new_conv = _make_conv(
        "he_sales_000001",
        ["המשפט הזה מופיע פעמיים", "תגובה עסקית", "המשפט הזה מופיע פעמיים"],
    )
    issues = checker.check_new_conversation(new_conv)
    assert any(i[0] == "exact_duplicate" for i in issues)
