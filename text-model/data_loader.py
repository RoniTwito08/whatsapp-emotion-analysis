"""data_loader.py

Config loading, reproducibility, tokenizer/model construction, and
Dataset/DataLoader factory functions for the Hebrew text-model pipeline.

This module only prepares model inputs (tokenizer, dataset, dataloader,
model with a fresh classification head). It does not train anything.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dataset import HebrewConversationDataset, DatasetError, load_corpus

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("data_loader")


class ConfigError(RuntimeError):
    """Raised for any recoverable, user-facing config error."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and minimally validate the JSON config file."""
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {config_path}: {exc}") from exc

    if not config.get("model_name"):
        raise ConfigError("Config is missing required key 'model_name'.")
    if not config.get("data", {}).get("input_path"):
        raise ConfigError("Config is missing required key 'data.input_path'.")
    if not config.get("data", {}).get("label_mapping"):
        raise ConfigError("Config is missing required key 'data.label_mapping'.")

    return config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible smoke tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Tokenizer / model loading
# ---------------------------------------------------------------------------

def load_tokenizer(config: Dict[str, Any]) -> PreTrainedTokenizerBase:
    """Load the Hugging Face tokenizer named in config['model_name']."""
    model_name = config["model_name"]
    logger.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    config: Dict[str, Any],
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> PreTrainedModel:
    """Load the pretrained checkpoint with a fresh classification head.

    The base AlephBERT checkpoint has no classification head, so
    Transformers will log a warning about newly initialized weights for
    the classifier layer. That warning is expected here and is NOT an
    error -- see README.md.
    """
    model_name = config["model_name"]
    num_labels = int(config.get("model", {}).get("num_labels", len(label_to_id)))
    if num_labels != len(label_to_id):
        raise ConfigError(
            f"config['model']['num_labels']={num_labels} does not match the "
            f"{len(label_to_id)} class(es) derived from data.label_mapping."
        )

    logger.info("Loading model: %s (num_labels=%d)", model_name, num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label=id_to_label,
    )

    device = get_device()
    logger.info("Selected device: %s", device)
    model.to(device)
    return model


def get_device() -> torch.device:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset / DataLoader factories
# ---------------------------------------------------------------------------

def load_corpus_records(config: Dict[str, Any], base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load raw corpus records from the path specified in config['data']['input_path']."""
    input_path = Path(config["data"]["input_path"])
    if base_dir is not None and not input_path.is_absolute():
        input_path = base_dir / input_path
    try:
        return load_corpus(input_path)
    except DatasetError as exc:
        raise ConfigError(str(exc)) from exc


def create_dataset(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    base_dir: Optional[Path] = None,
    subset_ids: Optional[Set[str]] = None,
) -> HebrewConversationDataset:
    """Build the HebrewConversationDataset described by config.

    Args:
        subset_ids: Optional set of conversation IDs to include. When provided,
            only records whose id matches this set are retained. Useful for
            creating train/val/test split datasets from a shared corpus file.
    """
    try:
        return HebrewConversationDataset(config, tokenizer, base_dir=base_dir, subset_ids=subset_ids)
    except DatasetError as exc:
        raise ConfigError(str(exc)) from exc


def create_dataloader(
    dataset: HebrewConversationDataset,
    config: Dict[str, Any],
    split: str = "train",
) -> DataLoader:
    """Build a DataLoader using dataloader.* settings from config.

    Args:
        split: One of 'train', 'val', 'test'. Controls batch_size key and
            whether shuffle is applied (only for train).
    """
    dataloader_config = config.get("dataloader", {})
    if split == "train":
        batch_size = int(dataloader_config.get("train_batch_size", dataloader_config.get("batch_size", 8)))
        shuffle = bool(dataloader_config.get("shuffle", True))
    else:
        batch_size = int(dataloader_config.get("eval_batch_size", dataloader_config.get("batch_size", 16)))
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(dataloader_config.get("num_workers", 0)),
        pin_memory=bool(dataloader_config.get("pin_memory", False)),
    )
