"""Tests for dataset_loader module."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_loader import (
    build_message_text_index,
    extract_id_number,
    get_max_id,
    make_next_id,
    normalize_text,
)


def test_extract_id_number_valid():
    assert extract_id_number("he_sales_000001") == 1
    assert extract_id_number("he_sales_000300") == 300
    assert extract_id_number("he_sales_005000") == 5000


def test_extract_id_number_invalid():
    with pytest.raises(ValueError):
        extract_id_number("he_sales_001")
    with pytest.raises(ValueError):
        extract_id_number("sales_000001")
    with pytest.raises(ValueError):
        extract_id_number("")


def test_get_max_id_basic():
    convs = [
        {"conversation_id": "he_sales_000001"},
        {"conversation_id": "he_sales_000050"},
        {"conversation_id": "he_sales_000025"},
    ]
    assert get_max_id(convs) == 50


def test_get_max_id_empty():
    assert get_max_id([]) == 0


def test_make_next_id():
    assert make_next_id(0) == "he_sales_000001"
    assert make_next_id(300) == "he_sales_000301"
    assert make_next_id(4999) == "he_sales_005000"


def test_make_next_id_resume():
    """After accepting 5 conversations starting at 300, next ID is 306."""
    assert make_next_id(300 + 5) == "he_sales_000306"


def test_normalize_text_whitespace():
    assert normalize_text("  שלום   עולם  ") == "שלום עולם"


def test_normalize_text_punctuation_stripped():
    result = normalize_text("שלום.")
    assert not result.endswith(".")
    result2 = normalize_text("שלום,")
    assert not result2.endswith(",")


def test_normalize_text_quotes():
    text = normalize_text("5,000 ש״ח")
    assert "״" not in text or '"' in text or "ש" in text


def test_build_message_text_index():
    convs = [
        {
            "conversation_id": "he_sales_000001",
            "messages": [
                {"text": "שלום, אני מתעניין"},
                {"text": "תודה רבה"},
            ],
        }
    ]
    idx = build_message_text_index(convs)
    norm_hello = normalize_text("שלום, אני מתעניין")
    assert norm_hello in idx
    assert idx[norm_hello] == "he_sales_000001"


def test_load_corpus(tmp_path):
    from src.dataset_loader import load_corpus

    data = [{"conversation_id": "he_sales_000001", "messages": []}]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_corpus(p)
    assert len(loaded) == 1
    assert loaded[0]["conversation_id"] == "he_sales_000001"


def test_load_corpus_real():
    """Ensure the real corpus_300.json loads correctly."""
    corpus_path = Path(__file__).parent.parent / "data" / "corpus_300.json"
    if not corpus_path.exists():
        pytest.skip("corpus_300.json not found")
    from src.dataset_loader import load_corpus

    data = load_corpus(corpus_path)
    assert len(data) == 300
    assert data[0]["conversation_id"] == "he_sales_000001"
    assert data[-1]["conversation_id"] == "he_sales_000300"
    assert get_max_id(data) == 300
    assert make_next_id(get_max_id(data)) == "he_sales_000301"
