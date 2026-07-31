#!/usr/bin/env python3
"""CLI tool for validating a generated or combined dataset JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import setup_logging
from src.generator import REPORTS_DIR, VALIDATION_REPORT_PATH
from src.validator import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Hebrew sales conversation dataset JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_dataset.py output/combined_dataset.json
  python validate_dataset.py data/corpus_300.json
""",
    )
    parser.add_argument("dataset", type=Path, help="Path to the JSON dataset file to validate")
    parser.add_argument("--output", type=Path, default=None, help="Override report output path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    dataset_path: Path = args.dataset
    if not dataset_path.exists():
        print(f"Error: File not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {dataset_path} …")
    try:
        with dataset_path.open(encoding="utf-8") as f:
            conversations = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: Cannot parse JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(conversations, list):
        print("Error: Dataset must be a JSON array of conversations.", file=sys.stderr)
        sys.exit(1)

    print(f"Validating {len(conversations)} conversations …")
    report = validate_dataset(conversations)

    total = report["total_conversations"]
    valid = report["valid_conversations"]
    invalid = report["invalid_conversations"]

    print(f"\nValidation Summary")
    print(f"  Total       : {total}")
    print(f"  Valid       : {valid}")
    print(f"  Invalid     : {invalid}")

    if report["errors"]:
        print("\nFirst 10 errors:")
        for cid, errors in list(report["errors"].items())[:10]:
            print(f"  [{cid}]")
            for e in errors[:3]:
                print(f"    - {e}")

    out_path = args.output or VALIDATION_REPORT_PATH
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport written to: {out_path}")

    if invalid > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
