"""Validate conversation objects against the corpus schema."""

from __future__ import annotations

import logging
import re
from typing import Any

from .models import (
    ALLOWED_INTEREST_LABELS,
    ALLOWED_OUTCOMES,
    ALLOWED_ROLES,
    ALLOWED_TRAJECTORIES,
    TRAJECTORY_TO_OUTCOME,
)

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^he_sales_\d{6}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")

# Hebrew gender slash forms always have a full word (2+ chars) before the slash
# and a 1-2 char gender suffix after it (ה, ת, י, ים, נה, ן).
# This avoids false positives for legitimate dual-term slashes (לפני/אחרי,
# צילום/סיכום, בוקר/אחר) which have full words on both sides.
_SLASH_GENDER_FORMS = re.compile(r"[א-ת]{2,}/[א-ת]{1,2}(?=[^א-ת]|$)", re.UNICODE)

_HEBREW_RE = re.compile(r"[א-ת]")

REQUIRED_CONVERSATION_FIELDS = [
    "conversation_id",
    "source_dataset",
    "language",
    "locale",
    "synthetic",
    "domain",
    "domain_he",
    "product_or_service",
    "customer_persona",
    "business_style",
    "interest_trajectory",
    "initial_interest_score",
    "final_interest_score",
    "final_outcome",
    "messages",
    "metadata",
]

REQUIRED_MESSAGE_FIELDS = [
    "message_index",
    "role",
    "text",
    "response_delay_minutes",
    "interest_label",
    "interest_score",
    "behavioral_features",
    "timestamp",
]

REQUIRED_BEHAVIORAL_FIELDS = [
    "message_length_chars",
    "question_count",
    "emoji_count",
    "contains_price",
    "contains_negotiation",
    "contains_objection",
    "contains_commitment",
    "contains_delay_signal",
]

REQUIRED_METADATA_FIELDS = ["channel", "contains_personal_data", "quality_status", "generator_seed"]


class ValidationError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_conversation(conv: dict[str, Any], check_slash_gender: bool = True) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []

    for field in REQUIRED_CONVERSATION_FIELDS:
        if field not in conv:
            errors.append(f"Missing field: {field!r}")

    if errors:
        return errors

    cid = conv["conversation_id"]
    if not _ID_RE.match(str(cid)):
        errors.append(f"Invalid conversation_id format: {cid!r}")

    if conv.get("language") != "he":
        errors.append(f"language must be 'he', got {conv.get('language')!r}")

    if conv.get("locale") != "he-IL":
        errors.append(f"locale must be 'he-IL', got {conv.get('locale')!r}")

    if conv.get("synthetic") is not True:
        errors.append("synthetic must be true")

    traj = conv.get("interest_trajectory", "")
    if traj not in ALLOWED_TRAJECTORIES:
        errors.append(f"Invalid interest_trajectory: {traj!r}")

    outcome = conv.get("final_outcome", "")
    if outcome not in ALLOWED_OUTCOMES:
        errors.append(f"Invalid final_outcome: {outcome!r}")

    expected_outcome = TRAJECTORY_TO_OUTCOME.get(traj)
    if expected_outcome and outcome != expected_outcome:
        errors.append(
            f"Trajectory {traj!r} should produce outcome {expected_outcome!r}, got {outcome!r}"
        )

    messages = conv.get("messages", [])
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors

    msg_errors = _validate_messages(messages, check_slash_gender=check_slash_gender)
    errors.extend(msg_errors)

    metadata = conv.get("metadata", {})
    meta_errors = _validate_metadata(metadata)
    errors.extend(meta_errors)

    return errors


def _validate_messages(messages: list[Any], check_slash_gender: bool = True) -> list[str]:
    errors: list[str] = []

    if not messages or messages[0].get("role") != "customer":
        errors.append("First message must be from 'customer'")

    customer_roles = [m for m in messages if isinstance(m, dict) and m.get("role") == "customer"]
    business_roles = [m for m in messages if isinstance(m, dict) and m.get("role") == "business"]
    if not customer_roles:
        errors.append("Conversation must have at least one customer message")
    if not business_roles:
        errors.append("Conversation must have at least one business message")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Message {i} is not a dict")
            continue
        for field in REQUIRED_MESSAGE_FIELDS:
            if field not in msg:
                errors.append(f"Message {i} missing field: {field!r}")

        if msg.get("message_index") != i:
            errors.append(f"Message {i} has wrong message_index: {msg.get('message_index')}")

        role = msg.get("role", "")
        if role not in ALLOWED_ROLES:
            errors.append(f"Message {i} invalid role: {role!r}")

        text = msg.get("text", "")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"Message {i} has empty text")
        else:
            if not _HEBREW_RE.search(text):
                errors.append(f"Message {i} appears to have no Hebrew characters")
            if check_slash_gender and _SLASH_GENDER_FORMS.search(text):
                errors.append(f"Message {i} contains forbidden slash-gender form: {text[:80]!r}")

        label = msg.get("interest_label", "")
        if label not in ALLOWED_INTEREST_LABELS:
            errors.append(f"Message {i} invalid interest_label: {label!r}")

        if role == "business":
            if label != "not_applicable":
                errors.append(f"Message {i} (business) must have interest_label='not_applicable'")
            if msg.get("interest_score") is not None:
                errors.append(f"Message {i} (business) must have interest_score=null")
        elif role == "customer":
            if label == "not_applicable":
                errors.append(f"Message {i} (customer) cannot have interest_label='not_applicable'")

        ts = msg.get("timestamp", "")
        if not isinstance(ts, str) or not _TIMESTAMP_RE.match(ts):
            errors.append(f"Message {i} invalid timestamp format: {ts!r} (expected YYYY-MM-DDTHH:MM)")

        delay = msg.get("response_delay_minutes")
        if not isinstance(delay, int) or delay < 0:
            errors.append(f"Message {i} invalid response_delay_minutes: {delay!r}")

        if i > 0:
            prev_ts = messages[i - 1].get("timestamp", "")
            curr_ts = ts
            if _TIMESTAMP_RE.match(prev_ts) and _TIMESTAMP_RE.match(curr_ts) and delay is not None:
                try:
                    from datetime import datetime

                    prev_dt = datetime.strptime(prev_ts, "%Y-%m-%dT%H:%M")
                    curr_dt = datetime.strptime(curr_ts, "%Y-%m-%dT%H:%M")
                    actual_delay = int((curr_dt - prev_dt).total_seconds() / 60)
                    if actual_delay != delay:
                        errors.append(
                            f"Message {i} response_delay_minutes={delay} does not match "
                            f"timestamp diff={actual_delay}"
                        )
                except ValueError:
                    pass

        bf = msg.get("behavioral_features", {})
        bf_errors = _validate_behavioral_features(bf, text, i)
        errors.extend(bf_errors)

    return errors


def _validate_behavioral_features(bf: Any, text: str, msg_idx: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(bf, dict):
        errors.append(f"Message {msg_idx} behavioral_features is not a dict")
        return errors

    for field in REQUIRED_BEHAVIORAL_FIELDS:
        if field not in bf:
            errors.append(f"Message {msg_idx} behavioral_features missing: {field!r}")

    if "message_length_chars" in bf and isinstance(text, str):
        expected = len(text)
        actual = bf["message_length_chars"]
        if actual != expected:
            errors.append(
                f"Message {msg_idx} message_length_chars={actual} but len(text)={expected}"
            )

    for bool_field in [
        "contains_price",
        "contains_negotiation",
        "contains_objection",
        "contains_commitment",
        "contains_delay_signal",
    ]:
        val = bf.get(bool_field)
        if val is not None and not isinstance(val, bool):
            errors.append(f"Message {msg_idx} {bool_field} must be bool, got {type(val).__name__}")

    return errors


def _validate_metadata(meta: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(meta, dict):
        errors.append("metadata is not a dict")
        return errors
    for field in REQUIRED_METADATA_FIELDS:
        if field not in meta:
            errors.append(f"metadata missing field: {field!r}")
    if meta.get("channel") != "whatsapp":
        errors.append(f"metadata.channel must be 'whatsapp', got {meta.get('channel')!r}")
    return errors


def validate_dataset(conversations: list[dict[str, Any]], check_slash_gender: bool = True) -> dict[str, Any]:
    """Validate an entire dataset and return a report."""
    total = len(conversations)
    errors_by_id: dict[str, list[str]] = {}

    seen_ids: set[str] = set()
    for conv in conversations:
        cid = conv.get("conversation_id", "<missing>")
        if cid in seen_ids:
            errors_by_id.setdefault(cid, []).append(f"Duplicate conversation_id: {cid!r}")
        seen_ids.add(cid)
        errs = validate_conversation(conv, check_slash_gender=check_slash_gender)
        if errs:
            errors_by_id[cid] = errs

    return {
        "total_conversations": total,
        "valid_conversations": total - len(errors_by_id),
        "invalid_conversations": len(errors_by_id),
        "errors": errors_by_id,
    }
