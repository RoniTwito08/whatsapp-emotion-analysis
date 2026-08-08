"""dataset.py

PyTorch Dataset for the Hebrew customer-interest text model.

Loads conversation records (JSONL / JSON / CSV), extracts and joins the
configured message roles into a single text per conversation, maps raw
outcome labels onto the final model classes, tokenizes everything once
during initialization, and exposes tensors ready for a Hugging Face
sequence-classification model.

This module only prepares model *inputs*. It does not train anything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("dataset")


class DatasetError(RuntimeError):
    """Raised for any recoverable, user-facing dataset-building error."""


# ---------------------------------------------------------------------------
# Corpus loading (JSONL / JSON / CSV -> list of dict records)
# ---------------------------------------------------------------------------

def load_corpus(input_path: Path) -> List[Dict[str, Any]]:
    """Load a corpus file and return a flat list of raw record dicts.

    Supports three formats, detected from the file extension:
      * .jsonl -- one JSON object per line.
      * .json  -- a JSON array of objects, an object wrapping the array
                  under a common key ("data", "records", "conversations",
                  "items"), or a single record object.
      * .csv   -- a standard CSV file with a header row.
    """
    if not input_path.exists():
        raise DatasetError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(input_path)
    if suffix == ".json":
        return _load_json(input_path)
    if suffix == ".csv":
        return _load_csv(input_path)
    raise DatasetError(
        f"Unsupported input file extension '{suffix}'. Expected .jsonl, .json, or .csv."
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    if not records:
        raise DatasetError(f"No records found in {path}")
    return records


def _load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "conversations", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise DatasetError(
        f"Unsupported JSON structure in {path}: expected a list or object at the top level."
    )


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # pandas raises several exception types
        raise DatasetError(f"Failed to read CSV file {path}: {exc}") from exc
    if df.empty:
        raise DatasetError(f"No records found in {path}")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Nested field access (dot notation)
# ---------------------------------------------------------------------------

def get_nested_value(record: Mapping[str, Any], dotted_path: Optional[str]) -> Any:
    """Fetch a possibly-nested value from a record using dot notation.

    Example: get_nested_value(record, "customer.contact.email") walks
    record["customer"]["contact"]["email"], returning None if any hop
    along the way is missing or is not a mapping.
    """
    if not dotted_path:
        return None
    current: Any = record
    for part in dotted_path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def invert_label_mapping(label_mapping: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    """Turn {mapped_label: [raw_label, ...]} into {raw_label: mapped_label}."""
    inverted: Dict[str, str] = {}
    for mapped_label, raw_labels in label_mapping.items():
        for raw_label in raw_labels:
            if raw_label in inverted:
                raise DatasetError(
                    f"Raw label '{raw_label}' appears under more than one mapped "
                    f"label in label_mapping ('{inverted[raw_label]}' and '{mapped_label}')."
                )
            inverted[raw_label] = mapped_label
    return inverted


def build_label_to_id(label_mapping: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    """Deterministically assign integer IDs to mapped label names.

    Labels are sorted alphabetically so the mapping is reproducible across
    runs regardless of key insertion order in config.json. For the default
    binary config this yields {"interested": 0, "losing_interest": 1}.
    """
    return {label: index for index, label in enumerate(sorted(label_mapping.keys()))}


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------

def _extract_conversation_text(
    record: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> str:
    """Extract and join the configured message roles for one conversation."""
    messages_field = data_config.get("messages_field", "messages")
    role_field = data_config.get("role_field", "role")
    text_field = data_config.get("text_field", "text")
    included_roles = data_config.get("included_roles")
    separator = data_config.get("message_separator", " [SEP] ")

    messages = get_nested_value(record, messages_field)
    if not isinstance(messages, list) or not messages:
        return ""

    pieces: List[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = get_nested_value(message, role_field)
        text = get_nested_value(message, text_field)
        if role is not None and not isinstance(role, str):
            role = str(role)
        if text is not None and not isinstance(text, str):
            text = str(text)
        if included_roles and role not in included_roles:
            continue
        if text:
            pieces.append(text)
    return separator.join(pieces)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HebrewConversationDataset(Dataset):
    """PyTorch Dataset over tokenized Hebrew customer conversations.

    Each retained record is reduced to: a joined customer-message string, a
    mapped label, and a record ID. Tokenization happens once here, during
    __init__, so it is not repeated on every epoch.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        base_dir: Optional[Path] = None,
        subset_ids: Optional[Set[str]] = None,
    ) -> None:
        """Build the dataset described by config["data"] / config["tokenizer"].

        Args:
            config: The full parsed config.json contents.
            tokenizer: A loaded Hugging Face tokenizer.
            base_dir: Directory that data.input_path is resolved relative to.
                Defaults to the current working directory.
            subset_ids: Optional set of conversation IDs to include. When
                provided, only records whose id_field value is in this set are
                retained. Pass None to include all records (original behavior).
        """
        self.tokenizer = tokenizer
        data_config = config.get("data", {})
        tokenizer_config = config.get("tokenizer", {})

        label_mapping = data_config.get("label_mapping")
        if not label_mapping:
            raise DatasetError("config['data']['label_mapping'] is required and cannot be empty.")
        self.label_to_id: Dict[str, int] = build_label_to_id(label_mapping)
        self.id_to_label: Dict[int, str] = {index: label for label, index in self.label_to_id.items()}
        raw_to_mapped = invert_label_mapping(label_mapping)

        input_path = Path(data_config.get("input_path", "data/sample_corpus.jsonl"))
        if base_dir is not None and not input_path.is_absolute():
            input_path = base_dir / input_path
        records = load_corpus(input_path)
        logger.info("Loaded %d raw record(s) from %s", len(records), input_path)

        texts, labels, record_ids = self._filter_and_extract(
            records, data_config, raw_to_mapped, self.label_to_id, subset_ids=subset_ids
        )
        self._validate_retained_records(texts, labels, record_ids)

        max_length = int(tokenizer_config.get("max_length", 256))
        padding = tokenizer_config.get("padding", "max_length")
        truncation = bool(tokenizer_config.get("truncation", True))
        add_special_tokens = bool(tokenizer_config.get("add_special_tokens", True))

        encodings = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

        self.input_ids: torch.Tensor = encodings["input_ids"].long()
        self.attention_mask: torch.Tensor = encodings["attention_mask"].long()
        self.token_type_ids: Optional[torch.Tensor] = (
            encodings["token_type_ids"].long() if "token_type_ids" in encodings else None
        )
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)
        self.record_ids: List[str] = record_ids
        self.texts: List[str] = texts

        logger.info(
            "Built dataset with %d usable record(s) across %d class(es): %s",
            len(self.record_ids),
            len(self.label_to_id),
            self.label_to_id,
        )

    # -- Filtering / extraction -------------------------------------------------

    @staticmethod
    def _filter_and_extract(
        records: Sequence[Mapping[str, Any]],
        data_config: Mapping[str, Any],
        raw_to_mapped: Mapping[str, str],
        label_to_id: Mapping[str, int],
        subset_ids: Optional[Set[str]] = None,
    ) -> tuple[List[str], List[int], List[str]]:
        """Turn raw records into parallel lists of (text, label_id, record_id)."""
        id_field = data_config.get("id_field", "conversation_id")
        label_field = data_config.get("label_field", "final_outcome")

        texts: List[str] = []
        labels: List[int] = []
        record_ids: List[str] = []
        seen_ids: set[str] = set()

        skipped_no_messages = 0
        skipped_empty_text = 0
        skipped_unmapped_label = 0
        skipped_missing_label = 0
        skipped_not_in_subset = 0

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                skipped_no_messages += 1
                continue

            if subset_ids is not None:
                record_id_check = get_nested_value(record, id_field)
                if record_id_check is None or str(record_id_check) not in subset_ids:
                    skipped_not_in_subset += 1
                    continue

            raw_label = get_nested_value(record, label_field)
            if raw_label is None:
                skipped_missing_label += 1
                continue
            raw_label = str(raw_label)
            if raw_label not in raw_to_mapped:
                skipped_unmapped_label += 1
                continue
            mapped_label = raw_to_mapped[raw_label]

            messages = get_nested_value(record, data_config.get("messages_field", "messages"))
            if not isinstance(messages, list) or not messages:
                skipped_no_messages += 1
                continue

            text = _extract_conversation_text(record, data_config)
            if not text.strip():
                skipped_empty_text += 1
                continue

            record_id = get_nested_value(record, id_field)
            if record_id is None:
                record_id = f"row_{index}"
            record_id = str(record_id)
            if record_id in seen_ids:
                raise DatasetError(f"Duplicate record ID found in corpus: {record_id!r}")
            seen_ids.add(record_id)

            texts.append(text)
            labels.append(label_to_id[mapped_label])
            record_ids.append(record_id)

        if skipped_not_in_subset:
            logger.info("Excluded %d record(s) not in subset_ids.", skipped_not_in_subset)
        if skipped_no_messages:
            logger.info("Ignored %d record(s) with a missing/empty message list.", skipped_no_messages)
        if skipped_missing_label:
            logger.info("Ignored %d record(s) with a missing label field.", skipped_missing_label)
        if skipped_unmapped_label:
            logger.info("Ignored %d record(s) with a label not present in label_mapping.", skipped_unmapped_label)
        if skipped_empty_text:
            logger.info("Ignored %d record(s) with empty text after role filtering.", skipped_empty_text)

        return texts, labels, record_ids

    # -- Validation ---------------------------------------------------------

    @staticmethod
    def _validate_retained_records(texts: Sequence[str], labels: Sequence[int], record_ids: Sequence[str]) -> None:
        """Validate invariants that must hold for the final retained set."""
        if not texts:
            raise DatasetError("No usable records remain after extraction and label filtering.")

        if len(texts) != len(labels) or len(texts) != len(record_ids):
            raise DatasetError("Internal error: texts/labels/record_ids length mismatch.")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise DatasetError("Encountered a retained record with empty or non-string text.")

        if len(set(record_ids)) != len(record_ids):
            raise DatasetError("Duplicate record IDs found among retained records.")

        if len(set(labels)) < 2:
            raise DatasetError(
                "Fewer than 2 classes remain after filtering. Need at least 2 classes to train a classifier. "
                f"Remaining label IDs: {sorted(set(labels))}"
            )

    # -- torch.utils.data.Dataset interface ----------------------------------

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "record_id": self.record_ids[index],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[index]
        return item
