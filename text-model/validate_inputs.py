"""validate_inputs.py

Command-line smoke test for the Hebrew text-model input pipeline.

Loads config.json, builds the tokenizer/dataset/dataloader/model, pulls one
batch, validates every tensor property required by the Definition of Done,
runs a single no-grad forward pass, and prints a validation report.

This script does NOT train the model.

Usage:
    python validate_inputs.py --config config.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from data_loader import (
    ConfigError,
    create_dataloader,
    create_dataset,
    get_device,
    load_config,
    load_model,
    load_tokenizer,
    set_random_seed,
)


class ValidationError(RuntimeError):
    """Raised when a smoke-test validation check fails."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Hebrew text-model input pipeline.")
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json")
    return parser.parse_args(argv)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


# ---------------------------------------------------------------------------
# Batch-level validation
# ---------------------------------------------------------------------------

def validate_batch_tensors(
    batch: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_labels: int,
) -> None:
    """Validate structure, dtypes, shapes, and value ranges of one batch."""
    _check("input_ids" in batch, "Batch is missing 'input_ids'.")
    _check("attention_mask" in batch, "Batch is missing 'attention_mask'.")
    _check("labels" in batch, "Batch is missing 'labels'.")
    _check("record_id" in batch, "Batch is missing 'record_id'.")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    _check(input_ids.dtype == torch.long, f"input_ids dtype must be torch.long, got {input_ids.dtype}.")
    _check(attention_mask.dtype == torch.long, f"attention_mask dtype must be torch.long, got {attention_mask.dtype}.")
    _check(labels.dtype == torch.long, f"labels dtype must be torch.long, got {labels.dtype}.")

    _check(input_ids.dim() == 2, f"input_ids must be 2-D [batch, seq_len], got shape {tuple(input_ids.shape)}.")
    _check(attention_mask.dim() == 2, f"attention_mask must be 2-D [batch, seq_len], got shape {tuple(attention_mask.shape)}.")
    _check(labels.dim() == 1, f"labels must be 1-D [batch], got shape {tuple(labels.shape)}.")

    _check(
        input_ids.shape == attention_mask.shape,
        f"input_ids shape {tuple(input_ids.shape)} != attention_mask shape {tuple(attention_mask.shape)}.",
    )
    _check(
        input_ids.shape[0] == labels.shape[0],
        f"Batch size mismatch: input_ids has {input_ids.shape[0]} rows, labels has {labels.shape[0]}.",
    )
    _check(
        input_ids.shape[1] <= max_length,
        f"Sequence length {input_ids.shape[1]} exceeds configured max_length {max_length}.",
    )

    vocab_size = len(tokenizer)
    _check(
        bool(((input_ids >= 0) & (input_ids < vocab_size)).all()),
        f"Found token IDs outside the tokenizer vocabulary (size {vocab_size}).",
    )
    _check(
        bool(((attention_mask == 0) | (attention_mask == 1)).all()),
        "attention_mask must contain only 0 and 1.",
    )
    _check(
        bool(((labels >= 0) & (labels < num_labels)).all()),
        f"labels must be within [0, {num_labels - 1}].",
    )

    for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask), ("labels", labels)):
        floated = tensor.float()
        _check(not torch.isnan(floated).any().item(), f"{name} contains NaN values.")
        _check(not torch.isinf(floated).any().item(), f"{name} contains infinite values.")

    record_ids = batch["record_id"]
    _check(len(record_ids) == input_ids.shape[0], "record_id count does not match batch size.")
    for record_id in record_ids:
        _check(isinstance(record_id, str) and bool(record_id), "Found an empty or non-string record_id.")


def validate_model_outputs(outputs: Any, batch_size: int, num_labels: int) -> None:
    """Validate the forward-pass output object."""
    _check(hasattr(outputs, "logits") and outputs.logits is not None, "Model outputs are missing 'logits'.")
    logits = outputs.logits
    _check(
        tuple(logits.shape) == (batch_size, num_labels),
        f"logits shape must be ({batch_size}, {num_labels}), got {tuple(logits.shape)}.",
    )
    _check(not torch.isnan(logits).any().item(), "logits contain NaN values.")
    _check(not torch.isinf(logits).any().item(), "logits contain infinite values.")

    _check(
        hasattr(outputs, "loss") and outputs.loss is not None,
        "Model outputs are missing 'loss' even though labels were provided.",
    )
    loss_value = outputs.loss.item()
    _check(math.isfinite(loss_value), f"Forward-pass loss is not finite: {loss_value}.")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(config_path: Path) -> None:
    config = load_config(config_path)
    base_dir = config_path.resolve().parent
    set_random_seed(int(config.get("random_seed", 42)))

    tokenizer = load_tokenizer(config)
    dataset = create_dataset(config, tokenizer, base_dir=base_dir)
    dataloader = create_dataloader(dataset, config)
    model = load_model(config, dataset.label_to_id, dataset.id_to_label)
    device = get_device()

    num_labels = int(config.get("model", {}).get("num_labels", len(dataset.label_to_id)))
    _check(
        num_labels == len(dataset.label_to_id),
        f"config['model']['num_labels']={num_labels} does not match dataset classes {dataset.label_to_id}.",
    )

    batch = next(iter(dataloader))
    max_length = int(config.get("tokenizer", {}).get("max_length", 256))
    validate_batch_tensors(batch, tokenizer, max_length, num_labels)

    decoded_sample = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
    _check(bool(decoded_sample.strip()), "Decoded sample text is empty.")

    model.eval()
    with torch.no_grad():
        model_inputs: Dict[str, torch.Tensor] = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        if "token_type_ids" in batch:
            model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)
        outputs = model(**model_inputs)

    batch_size_actual = int(batch["input_ids"].shape[0])
    validate_model_outputs(outputs, batch_size_actual, num_labels)

    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Labels: {dataset.label_to_id}")
    print(f"Batch size: {batch_size_actual}")
    print(f"Input IDs shape: {tuple(batch['input_ids'].shape)}")
    print(f"Attention mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"Labels shape: {tuple(batch['labels'].shape)}")
    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(f"Forward-pass loss: {outputs.loss.item():.4f}")
    print(f"Decoded sample: {decoded_sample}")
    print("-" * 60)
    print("All input validations passed.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.config)
    except (ConfigError, ValidationError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure surface
        print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
