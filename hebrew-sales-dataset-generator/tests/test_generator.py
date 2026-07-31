"""End-to-end generation tests using mocked LLM."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.generator import run_generation, _parse_llm_response, _validate_llm_output
from src.llm_client import MockLLMClient


def _mock_response(messages: list[dict], outcome: str = "pending") -> str:
    return json.dumps(
        {
            "messages": messages,
            "actual_outcome": outcome,
            "trajectory_notes": "Test trajectory",
        },
        ensure_ascii=False,
    )


def _mock_7_message_response(trajectory: str = "medium_to_pending") -> str:
    return _mock_response(
        [
            {"role": "customer", "text": "שלום, אני מחפש שירות איכותי"},
            {"role": "business", "text": "ברוך הבא, אשמח לעזור"},
            {"role": "customer", "text": "כמה זה עולה לחודש?"},
            {"role": "business", "text": "המחיר הוא 500 ש״ח לחודש"},
            {"role": "customer", "text": "צריך לחשוב על זה"},
            {"role": "business", "text": "בסדר גמור, אני פה"},
            {"role": "customer", "text": "אחזור אליך בשבוע הבא"},
        ],
        outcome="pending",
    )


# ──────────────────────── JSON parsing ────────────────────────
def test_parse_llm_response_plain_json():
    raw = '{"messages": [], "actual_outcome": "pending", "trajectory_notes": "ok"}'
    result = _parse_llm_response(raw)
    assert result["actual_outcome"] == "pending"


def test_parse_llm_response_markdown_fence():
    raw = '```json\n{"messages": [], "actual_outcome": "x", "trajectory_notes": "y"}\n```'
    result = _parse_llm_response(raw)
    assert result["actual_outcome"] == "x"


def test_parse_llm_response_plain_fence():
    raw = '```\n{"messages": [], "actual_outcome": "z", "trajectory_notes": "w"}\n```'
    result = _parse_llm_response(raw)
    assert result["actual_outcome"] == "z"


def test_parse_llm_response_invalid_json():
    import json as _json

    with pytest.raises(_json.JSONDecodeError):
        _parse_llm_response("not json at all")


# ──────────────────────── LLM output validation ────────────────────────
def test_validate_llm_output_correct():
    data = {
        "messages": [
            {"role": "customer", "text": "שלום"},
            {"role": "business", "text": "היי"},
        ],
        "actual_outcome": "pending",
        "trajectory_notes": "ok",
    }
    result = _validate_llm_output(data, expected_count=2)
    assert len(result.messages) == 2
    assert result.messages[0].role == "customer"


def test_validate_llm_output_wrong_count():
    data = {
        "messages": [{"role": "customer", "text": "שלום"}],
        "actual_outcome": "pending",
        "trajectory_notes": "ok",
    }
    with pytest.raises(ValueError, match="Expected 7 messages"):
        _validate_llm_output(data, expected_count=7)


def test_validate_llm_output_invalid_role():
    data = {
        "messages": [{"role": "agent", "text": "שלום"}],
        "actual_outcome": "pending",
        "trajectory_notes": "ok",
    }
    with pytest.raises(ValueError):
        _validate_llm_output(data, expected_count=1)


def test_validate_llm_output_empty_text():
    data = {
        "messages": [{"role": "customer", "text": ""}],
        "actual_outcome": "pending",
        "trajectory_notes": "ok",
    }
    with pytest.raises(ValueError):
        _validate_llm_output(data, expected_count=1)


# ──────────────────────── mock LLM client ────────────────────────
def test_mock_llm_client_returns_valid_json():
    client = MockLLMClient()
    result = client.generate("system", "user")
    parsed = json.loads(result.content)
    assert "messages" in parsed
    assert "actual_outcome" in parsed


def test_mock_llm_client_custom_response():
    custom = [_mock_7_message_response()]
    client = MockLLMClient(responses=custom)
    result = client.generate("s", "u")
    data = json.loads(result.content)
    assert len(data["messages"]) == 7


# ──────────────────────── dry-run generation ────────────────────────
def test_dry_run_generates_no_api_calls():
    config = Config()
    stats = run_generation(
        count=3,
        dry_run=True,
        seed=42,
        config=config,
    )
    assert stats.api_requests == 0
    assert stats.requested_new_conversations == 3


# ──────────────────────── mocked end-to-end generation ────────────────────────
def _make_mock_responses_for_count(n: int) -> list[str]:
    """
    We use MockLLMClient with no pre-set responses (responses=[]) so it
    auto-generates the correct number of messages from the prompt.
    This helper is kept for tests that need explicit fixed responses.
    """
    responses = []
    for i in range(n + 5):
        msgs = [
            {"role": "customer", "text": f"שלום, אני מחפש שירות מספר {i + 1}"},
            {"role": "business", "text": f"ברוך הבא, אשמח לעזור {i + 1}"},
            {"role": "customer", "text": f"כמה זה עולה? שאלה {i + 1}"},
            {"role": "business", "text": f"המחיר הוא {500 + i * 10} ש״ח"},
            {"role": "customer", "text": f"צריך לחשוב על זה {i + 1}"},
            {"role": "business", "text": f"בסדר גמור, אני זמין {i + 1}"},
            {"role": "customer", "text": f"אחזור אליך בשבוע הבא, {i + 1}"},
        ]
        responses.append(_mock_response(msgs, "pending"))
    return responses


def test_mocked_generation_3_conversations(tmp_path, monkeypatch):
    """Mock full pipeline for 3 conversations."""
    import src.generator as gen_module

    monkeypatch.setattr(gen_module, "SOURCE_PATH", Path(__file__).parent.parent / "data" / "corpus_300.json")
    monkeypatch.setattr(gen_module, "GENERATED_PATH", tmp_path / "generated.json")
    monkeypatch.setattr(gen_module, "COMBINED_PATH", tmp_path / "combined.json")
    monkeypatch.setattr(gen_module, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(gen_module, "REJECTED_PATH", tmp_path / "rejected.jsonl")

    corpus_path = Path(__file__).parent.parent / "data" / "corpus_300.json"
    if not corpus_path.exists():
        pytest.skip("corpus_300.json not found")

    # Use MockLLMClient with no pre-set responses so it auto-matches requested count
    mock_client = MockLLMClient()

    config = Config()
    config.conversation_similarity_threshold = 101.0  # skip TF-IDF in tests
    stats = run_generation(
        count=3,
        dry_run=False,
        seed=99,
        config=config,
        llm_client=mock_client,
    )

    assert stats.accepted_new_conversations == 3
    assert stats.total_conversations == 303

    generated_path = tmp_path / "generated.json"
    assert generated_path.exists()
    with generated_path.open(encoding="utf-8") as f:
        generated = json.load(f)
    assert len(generated) == 3
    for i, conv in enumerate(generated):
        assert conv["conversation_id"] == f"he_sales_0003{i + 1:02d}"

    combined_path = tmp_path / "combined.json"
    with combined_path.open(encoding="utf-8") as f:
        combined = json.load(f)
    assert len(combined) == 303


def test_mocked_generation_unique_ids(tmp_path, monkeypatch):
    """All generated conversation IDs must be unique."""
    import src.generator as gen_module

    corpus_path = Path(__file__).parent.parent / "data" / "corpus_300.json"
    if not corpus_path.exists():
        pytest.skip("corpus_300.json not found")

    monkeypatch.setattr(gen_module, "SOURCE_PATH", corpus_path)
    monkeypatch.setattr(gen_module, "GENERATED_PATH", tmp_path / "generated.json")
    monkeypatch.setattr(gen_module, "COMBINED_PATH", tmp_path / "combined.json")
    monkeypatch.setattr(gen_module, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(gen_module, "REJECTED_PATH", tmp_path / "rejected.jsonl")

    mock_client = MockLLMClient()
    config = Config()
    config.conversation_similarity_threshold = 101.0
    stats = run_generation(count=5, dry_run=False, seed=7, config=config, llm_client=mock_client)

    generated_path = tmp_path / "generated.json"
    with generated_path.open(encoding="utf-8") as f:
        generated = json.load(f)

    ids = [c["conversation_id"] for c in generated]
    assert len(ids) == len(set(ids)), "Duplicate IDs detected"


def test_invalid_llm_json_retried(tmp_path, monkeypatch):
    """When LLM returns invalid JSON, generation should retry and eventually succeed."""
    import src.generator as gen_module

    corpus_path = Path(__file__).parent.parent / "data" / "corpus_300.json"
    if not corpus_path.exists():
        pytest.skip("corpus_300.json not found")

    monkeypatch.setattr(gen_module, "SOURCE_PATH", corpus_path)
    monkeypatch.setattr(gen_module, "GENERATED_PATH", tmp_path / "generated.json")
    monkeypatch.setattr(gen_module, "COMBINED_PATH", tmp_path / "combined.json")
    monkeypatch.setattr(gen_module, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(gen_module, "REJECTED_PATH", tmp_path / "rejected.jsonl")

    # First two responses are invalid JSON; third is valid (auto-sized to match plan)
    bad = "this is not json"

    class _PartialMockClient(MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self._fail_count = 0

        def generate(self, system_prompt: str, user_prompt: str) -> "LLMResult":
            if self._fail_count < 2:
                self._fail_count += 1
                from src.llm_client import LLMResult, UsageStats
                return LLMResult(content=bad, usage=UsageStats())
            return super().generate(system_prompt, user_prompt)

    mock_client = _PartialMockClient()
    config = Config()
    config.conversation_similarity_threshold = 101.0

    stats = run_generation(count=1, dry_run=False, seed=1, config=config, llm_client=mock_client)
    assert stats.accepted_new_conversations == 1
    assert stats.retry_count >= 2
