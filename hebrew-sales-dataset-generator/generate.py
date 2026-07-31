#!/usr/bin/env python3
"""CLI entry point for generating synthetic Hebrew WhatsApp business conversations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, setup_logging
from src.generator import (
    GENERATION_REPORT_PATH,
    REPORTS_DIR,
    run_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Hebrew WhatsApp sales conversations using OpenAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate exactly 20 new conversations
  python generate.py --count 20

  # Continue until dataset has 5000 total
  python generate.py --target-total 5000

  # Resume a previously interrupted run
  python generate.py --target-total 5000 --resume

  # Dry-run mode (no API calls)
  python generate.py --count 5 --dry-run

  # Reproducible generation
  python generate.py --count 20 --seed 42
""",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--count", type=int, metavar="N", help="Generate exactly N new conversations")
    group.add_argument(
        "--target-total",
        type=int,
        metavar="N",
        help="Continue until combined dataset has N conversations",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Create plans only, no API calls")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level",
    )
    return parser.parse_args()


_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    "gpt-5": (5.00, 20.00),
    "gpt-5-mini": (0.20, 0.80),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _COST_PER_1M.get(model)
    if rates is None:
        for key, r in _COST_PER_1M.items():
            if model.startswith(key):
                rates = r
                break
    if rates is None:
        return None
    input_cost = (input_tokens / 1_000_000) * rates[0]
    output_cost = (output_tokens / 1_000_000) * rates[1]
    return round(input_cost + output_cost, 6)


def main() -> None:
    args = parse_args()
    config = get_config()
    log_level = args.log_level or config.log_level
    setup_logging(log_level)

    logger = logging.getLogger(__name__)

    if args.count is None and args.target_total is None:
        print("Error: specify --count N or --target-total N", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and not config.has_api_key():
        print(
            "\nNo OPENAI_API_KEY found.\n"
            "Running in dry-run mode (no API calls will be made).\n"
            "To generate real conversations, add your key to .env:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Then run:\n"
            "  python generate.py --count 20\n",
        )
        args.dry_run = True

    logger.info(
        "Starting generation | count=%s | target_total=%s | dry_run=%s | seed=%s",
        args.count,
        args.target_total,
        args.dry_run,
        args.seed,
    )

    t_wall_start = time.monotonic()
    try:
        stats = run_generation(
            count=args.count,
            target_total=args.target_total,
            resume=args.resume,
            dry_run=args.dry_run,
            seed=args.seed,
            config=config,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved to output/checkpoint.json.", file=sys.stderr)
        sys.exit(0)

    elapsed_wall = time.monotonic() - t_wall_start
    estimated_cost = _estimate_cost(
        config.openai_model,
        stats.estimated_input_tokens,
        stats.estimated_output_tokens,
    )
    report = {
        "source_conversations": stats.source_conversations,
        "requested_new_conversations": stats.requested_new_conversations,
        "accepted_new_conversations": stats.accepted_new_conversations,
        "total_conversations": stats.total_conversations,
        "rejected_exact_duplicates": stats.rejected_exact_duplicates,
        "rejected_similar_messages": stats.rejected_similar_messages,
        "rejected_similar_conversations": stats.rejected_similar_conversations,
        "rejected_invalid_schema": stats.rejected_invalid_schema,
        "rejected_gender_inconsistency": stats.rejected_gender_inconsistency,
        "rejected_incoherent_outcome": stats.rejected_incoherent_outcome,
        "rejected_wrong_message_count": stats.rejected_wrong_message_count,
        "api_requests": stats.api_requests,
        "retry_count": stats.retry_count,
        "estimated_input_tokens": stats.estimated_input_tokens,
        "estimated_output_tokens": stats.estimated_output_tokens,
        "estimated_cost_usd": estimated_cost,
        "elapsed_seconds": round(elapsed_wall, 2),
        "dry_run": args.dry_run,
        "domain_distribution": stats.domain_distribution,
        "outcome_distribution": stats.outcome_distribution,
        "conversation_length_distribution": dict(stats.conversation_length_distribution),
        "most_repeated_normalized_phrases": stats.most_repeated_normalized_phrases,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with GENERATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    cost_str = f"${estimated_cost:.4f}" if estimated_cost is not None else "n/a"
    print(f"\n{'='*50}")
    print(f"Generation complete")
    print(f"  Accepted : {stats.accepted_new_conversations}")
    print(f"  Total    : {stats.total_conversations}")
    print(f"  API calls: {stats.api_requests}")
    print(f"  Retries  : {stats.retry_count}")
    print(f"  Elapsed  : {elapsed_wall:.1f}s")
    print(f"  Cost est : {cost_str}")
    if args.dry_run:
        print("\n[DRY-RUN] No API calls were made.")
        print("To run for real, configure OPENAI_API_KEY and remove --dry-run.")
    else:
        print(f"\nOutput    : output/combined_dataset.json")
        print(f"Report    : reports/generation_report.json")
        print(f"Rejected  : reports/rejected_conversations.jsonl")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
