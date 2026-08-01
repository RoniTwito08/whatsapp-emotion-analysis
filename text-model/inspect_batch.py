"""inspect_batch.py

Manual inspection tool for one tokenized batch: record IDs, the joined
customer text, mapped label, per-record token count before padding, and the
first tokens/IDs -- so a human can eyeball that Hebrew tokenization looks
reasonable before any training happens.

Usage:
    python inspect_batch.py --config config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from data_loader import create_dataloader, create_dataset, load_config, load_tokenizer, set_random_seed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one tokenized batch for manual review.")
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json")
    return parser.parse_args(argv)


def run(config_path: Path) -> None:
    config = load_config(config_path)
    base_dir = config_path.resolve().parent
    set_random_seed(int(config.get("random_seed", 42)))

    tokenizer = load_tokenizer(config)
    dataset = create_dataset(config, tokenizer, base_dir=base_dir)
    dataloader = create_dataloader(dataset, config)

    record_id_to_index = {record_id: index for index, record_id in enumerate(dataset.record_ids)}

    tokenizer_config = config.get("tokenizer", {})
    truncation = bool(tokenizer_config.get("truncation", True))
    max_length = int(tokenizer_config.get("max_length", 256))
    add_special_tokens = bool(tokenizer_config.get("add_special_tokens", True))

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    record_ids = batch["record_id"]

    print("=" * 70)
    print("BATCH INSPECTION")
    print("=" * 70)
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")
    print()

    for position, record_id in enumerate(record_ids):
        dataset_index = record_id_to_index[record_id]
        text = dataset.texts[dataset_index]
        label_id = int(labels[position].item())
        label_name = dataset.id_to_label[label_id]

        # Re-tokenize without padding to measure the true (pre-padding) length.
        unpadded = tokenizer(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        token_count_before_padding = len(unpadded["input_ids"])

        row_ids = input_ids[position]
        first_30_ids = row_ids[:30].tolist()
        first_30_tokens = tokenizer.convert_ids_to_tokens(first_30_ids)
        decoded = tokenizer.decode(row_ids, skip_special_tokens=True)

        print("-" * 70)
        print(f"Record ID: {record_id}")
        print(f"Text: {text}")
        print(f"Label: {label_name} (id={label_id})")
        print(f"Token count before padding: {token_count_before_padding}")
        print(f"input_ids shape (this row): {tuple(row_ids.shape)}")
        print(f"attention_mask shape (this row): {tuple(attention_mask[position].shape)}")
        print(f"First 30 token IDs: {first_30_ids}")
        print(f"First 30 tokens: {first_30_tokens}")
        print(f"Decoded: {decoded}")

    print("-" * 70)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
