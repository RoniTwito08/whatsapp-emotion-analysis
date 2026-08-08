"""splitter.py

Deterministic, stratified train / validation / test split at conversation level.

The split is performed at the conversation level so that no conversation_id
appears in more than one split (no data leakage across splits). The split is
stratified by binary label to preserve class balance.

The resulting split_ids dict can be saved to JSON so that the TF-IDF baseline
can reuse the same test set for a scientifically fair comparison.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("splitter")


class SplitError(RuntimeError):
    """Raised when the split cannot be constructed."""


def stratified_split(
    records: Sequence[Mapping[str, Any]],
    label_mapping: Mapping[str, Sequence[str]],
    id_field: str = "conversation_id",
    label_field: str = "final_outcome",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_ids, val_ids, test_ids) for a stratified conversation split.

    Args:
        records: List of raw conversation dicts from the corpus.
        label_mapping: {binary_label: [raw_outcome, ...]} mapping.
        id_field: Field name holding the conversation ID.
        label_field: Field name holding the raw outcome (e.g. 'final_outcome').
        train_ratio: Fraction of conversations for training.
        val_ratio: Fraction of conversations for validation.
        test_ratio: Fraction of conversations for test.
        seed: Random seed for reproducibility.

    Returns:
        Three lists of conversation IDs: train, val, test.

    Raises:
        SplitError: If ratios don't sum to 1, or a class has too few examples.
    """
    from sklearn.model_selection import train_test_split

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise SplitError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, "
            f"got {train_ratio + val_ratio + test_ratio:.6f}."
        )

    raw_to_mapped: Dict[str, str] = {}
    for mapped_label, raw_labels in label_mapping.items():
        for raw in raw_labels:
            raw_to_mapped[raw] = mapped_label

    ids: List[str] = []
    labels: List[str] = []
    skipped = 0

    for record in records:
        raw_label = record.get(label_field)
        if raw_label is None or str(raw_label) not in raw_to_mapped:
            skipped += 1
            continue
        record_id = record.get(id_field)
        if record_id is None:
            skipped += 1
            continue
        ids.append(str(record_id))
        labels.append(raw_to_mapped[str(raw_label)])

    if skipped:
        logger.info("Skipped %d record(s) with missing/unmapped label or id.", skipped)

    if len(ids) < 6:
        raise SplitError(
            f"Too few usable records ({len(ids)}) for a stratified 3-way split."
        )

    # First split off the test set, then split the remainder into train/val.
    test_size = test_ratio
    val_size_of_remainder = val_ratio / (train_ratio + val_ratio)

    try:
        ids_trainval, ids_test, labels_trainval, _ = train_test_split(
            ids, labels,
            test_size=test_size,
            stratify=labels,
            random_state=seed,
        )
        ids_train, ids_val, _, _ = train_test_split(
            ids_trainval, labels_trainval,
            test_size=val_size_of_remainder,
            stratify=labels_trainval,
            random_state=seed,
        )
    except ValueError as exc:
        raise SplitError(
            f"Stratified split failed — one or more classes may have too few examples. "
            f"Original error: {exc}"
        ) from exc

    _validate_no_overlap(ids_train, ids_val, ids_test)

    return ids_train, ids_val, ids_test


def _validate_no_overlap(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    """Assert zero ID overlap between all three splits. Raises SplitError if any found."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    if tv or tt or vt:
        raise SplitError(
            f"ID overlap detected after split: "
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}. "
            f"This should never happen — check the splitter logic."
        )


def print_split_statistics(
    records: Sequence[Mapping[str, Any]],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    label_mapping: Mapping[str, Sequence[str]],
    id_field: str = "conversation_id",
    label_field: str = "final_outcome",
) -> None:
    """Print label distribution for each split to stdout."""
    raw_to_mapped: Dict[str, str] = {}
    for mapped_label, raw_labels in label_mapping.items():
        for raw in raw_labels:
            raw_to_mapped[raw] = mapped_label

    id_to_label: Dict[str, str] = {}
    for record in records:
        record_id = str(record.get(id_field, ""))
        raw_label = record.get(label_field)
        if raw_label and str(raw_label) in raw_to_mapped:
            id_to_label[record_id] = raw_to_mapped[str(raw_label)]

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        counts = Counter(id_to_label[i] for i in split_ids if i in id_to_label)
        total = sum(counts.values())
        dist = ", ".join(f"{k}={v} ({v/total:.1%})" for k, v in sorted(counts.items()))
        print(f"  {split_name:>5}: {total:4d} conversations | {dist}")


def save_split_ids(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    output_path: Path,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist split IDs to JSON so the baseline can reuse the same test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }
    if config_snapshot:
        payload["config_snapshot"] = config_snapshot
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Saved split IDs to %s", output_path)


def load_split_ids(split_ids_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Load split IDs previously saved by save_split_ids."""
    if not split_ids_path.exists():
        raise SplitError(f"split_ids file not found: {split_ids_path}")
    with split_ids_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["train_ids"], payload["val_ids"], payload["test_ids"]
