"""Atomic checkpoint save/restore for safe resume."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    last_accepted_id: str = ""
    requested_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    retry_count: int = 0
    random_seed: int = 0
    used_plan_signatures: list[str] = field(default_factory=list)
    outcome_distribution: dict[str, int] = field(default_factory=dict)
    domain_distribution: dict[str, int] = field(default_factory=dict)
    trajectory_distribution: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    api_request_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def load_checkpoint(path: Path) -> Checkpoint | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        cp = Checkpoint.from_dict(data)
        logger.info("Restored checkpoint: accepted=%d, last_id=%s", cp.accepted_count, cp.last_accepted_id)
        return cp
    except Exception as exc:
        logger.warning("Could not load checkpoint from %s: %s", path, exc)
        return None


def save_checkpoint(checkpoint: Checkpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint.to_dict(), path)
    logger.debug("Checkpoint saved to %s", path)


def load_generated(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Could not load generated conversations from %s: %s", path, exc)
        return []


def save_generated(conversations: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(conversations, path)


def save_combined(
    source_conversations: list[dict[str, Any]],
    generated_conversations: list[dict[str, Any]],
    path: Path,
) -> None:
    combined = source_conversations + generated_conversations
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(combined, path)
    logger.info("Saved combined dataset (%d conversations) to %s", len(combined), path)


def append_rejection_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_write_json(data: Any, path: Path) -> None:
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
