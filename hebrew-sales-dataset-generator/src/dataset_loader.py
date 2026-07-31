"""Load and index the approved source dataset."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^he_sales_(\d{6})$")


def load_corpus(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    logger.info("Loaded %d conversations from %s", len(data), path)
    return data


def extract_id_number(conversation_id: str) -> int:
    m = _ID_RE.match(conversation_id)
    if not m:
        raise ValueError(f"Cannot parse conversation ID: {conversation_id!r}")
    return int(m.group(1))


def get_max_id(conversations: list[dict[str, Any]]) -> int:
    if not conversations:
        return 0
    return max(extract_id_number(c["conversation_id"]) for c in conversations)


def make_next_id(current_max: int) -> str:
    return f"he_sales_{current_max + 1:06d}"


def build_message_text_index(conversations: list[dict[str, Any]]) -> dict[str, str]:
    """Return {normalized_text: conversation_id} for fast exact-duplicate detection."""
    index: dict[str, str] = {}
    for c in conversations:
        cid = c["conversation_id"]
        for m in c.get("messages", []):
            norm = normalize_text(m["text"])
            if norm and norm not in index:
                index[norm] = cid
    return index


def normalize_text(text: str) -> str:
    """Normalize a Hebrew message text for duplicate detection."""
    import unicodedata

    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("״", '"').replace("׳", "'")
    text = re.sub(r"[.!,;:]+$", "", text)
    return text.strip()


def get_domain_distribution(conversations: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in conversations:
        domain = c.get("domain", "unknown")
        dist[domain] = dist.get(domain, 0) + 1
    return dist


def get_outcome_distribution(conversations: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in conversations:
        outcome = c.get("final_outcome", "unknown")
        dist[outcome] = dist.get(outcome, 0) + 1
    return dist


def get_trajectory_distribution(conversations: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in conversations:
        traj = c.get("interest_trajectory", "unknown")
        dist[traj] = dist.get(traj, 0) + 1
    return dist
