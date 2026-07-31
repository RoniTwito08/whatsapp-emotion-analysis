#!/usr/bin/env python3
"""
Post-generation quality report for a batch of new synthetic conversations.

Usage:
    python quality_report.py --new-from 311          # analyse IDs >= he_sales_000311
    python quality_report.py --new-from 311 --top 20 # show top-20 repeated messages
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_SLASH_RE = re.compile(r"[א-ת]+/[א-ת]+", re.UNICODE)

# Generic questions that are inappropriate in non-physical-service domains
_GENERIC_SERVICE_PHRASES = [
    "ביטוח על העבודה",
    "צוות קבוע",
    "מי בדיוק מגיע",
    "צריך להיות בבית",
    "הגעת טכנאי",
    "עלות נוספת על הובלה",
    "אחריות על העבודה",
]

_PHYSICAL_DOMAINS = {
    "cleaning", "pest_control", "air_conditioning", "electrical", "plumbing",
    "construction", "solar", "home_renovation", "moving", "storage", "interior_design",
    "car_service",
}

_PRODUCT_DOMAINS = {"furniture", "electronics", "kitchens", "pet_store"}


def _id_num(cid: str) -> int:
    return int(cid.replace("he_sales_", ""))


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def analyse(conversations: list[dict], label: str, top_n: int = 15) -> None:
    print(f"\n{'='*60}")
    print(f" {label}  ({len(conversations)} conversations)")
    print(f"{'='*60}")

    # ── 1. Slash-gender violations ───────────────────────────────
    slash_hits: list[tuple[str, str, str]] = []
    for c in conversations:
        for m in c["messages"]:
            if _SLASH_RE.search(m["text"]):
                slash_hits.append((c["conversation_id"], m["role"], m["text"][:80]))

    print(f"\n[1] Slash-gender violations: {len(slash_hits)}")
    for cid, role, text in slash_hits[:10]:
        print(f"    {cid} [{role}]: {text}")
    if len(slash_hits) > 10:
        print(f"    … and {len(slash_hits) - 10} more")

    # ── 2. Repeated exact messages ───────────────────────────────
    msg_counter: Counter[str] = Counter()
    for c in conversations:
        for m in c["messages"]:
            msg_counter[m["text"]] += 1

    repeated = [(t, n) for t, n in msg_counter.most_common() if n > 1]
    print(f"\n[2] Repeated exact messages (appear >1× in this batch): {len(repeated)}")
    for text, cnt in repeated[:top_n]:
        print(f"    {cnt}×  {text[:80]}")

    # ── 3. Domain-inappropriate messages ─────────────────────────
    inappropriate: list[tuple[str, str, str, str]] = []
    for c in conversations:
        domain = c.get("domain", "")
        if domain in _PHYSICAL_DOMAINS or domain in _PRODUCT_DOMAINS:
            continue
        for m in c["messages"]:
            for phrase in _GENERIC_SERVICE_PHRASES:
                if phrase in m["text"]:
                    inappropriate.append((c["conversation_id"], domain, phrase, m["text"][:80]))
                    break

    print(f"\n[3] Domain-inappropriate messages: {len(inappropriate)}")
    for cid, domain, phrase, text in inappropriate[:10]:
        print(f"    {cid} [{domain}] trigger='{phrase}'")
        print(f"        {text}")
    if len(inappropriate) > 10:
        print(f"    … and {len(inappropriate) - 10} more")

    # ── 4. Outcome distribution ──────────────────────────────────
    outcomes = Counter(c["final_outcome"] for c in conversations)
    print(f"\n[4] Outcome distribution:")
    for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {outcome:<28} {count:3d}  {bar}")

    # ── 5. Domain distribution ───────────────────────────────────
    domains = Counter(c["domain"] for c in conversations)
    print(f"\n[5] Domain distribution (top 10):")
    for domain, count in domains.most_common(10):
        print(f"    {domain:<28} {count:3d}")

    # ── 6. Opening message uniqueness ───────────────────────────
    openings = [c["messages"][0]["text"] for c in conversations if c["messages"]]
    open_ctr = Counter(openings)
    repeated_openings = [(t, n) for t, n in open_ctr.most_common() if n > 1]
    print(f"\n[6] Repeated opening messages: {len(repeated_openings)}")
    for text, cnt in repeated_openings[:5]:
        print(f"    {cnt}×  {text[:90]}")

    # ── 7. Closing message uniqueness ───────────────────────────
    closings = [c["messages"][-1]["text"] for c in conversations if c["messages"]]
    close_ctr = Counter(closings)
    repeated_closings = [(t, n) for t, n in close_ctr.most_common() if n > 1]
    top_closing, top_count = close_ctr.most_common(1)[0] if close_ctr else ("—", 0)
    print(f"\n[7] Repeated closing messages: {len(repeated_closings)}")
    print(f"    Most common: {top_count}× \"{top_closing[:80]}\"")
    for text, cnt in repeated_closings[:5]:
        print(f"    {cnt}×  {text[:90]}")

    # ── Summary ──────────────────────────────────────────────────
    total_msgs = sum(len(c["messages"]) for c in conversations)
    unique_msg_pct = 100 * (len(msg_counter) / max(total_msgs, 1))
    print(f"\n── Summary ──")
    print(f"  Conversations analysed : {len(conversations)}")
    print(f"  Total messages         : {total_msgs}")
    print(f"  Unique messages        : {len(msg_counter)} ({unique_msg_pct:.1f}%)")
    print(f"  Slash-gender hits      : {len(slash_hits)}")
    print(f"  Domain-inappropriate   : {len(inappropriate)}")


def load_generation_report(reports_dir: Path) -> None:
    p = reports_dir / "generation_report.json"
    if not p.exists():
        print("\n[generation_report.json not found]")
        return
    with p.open(encoding="utf-8") as f:
        r = json.load(f)
    print(f"\n── Generation Report ──")
    print(f"  API requests   : {r.get('api_requests', '?')}")
    print(f"  Retries        : {r.get('retry_count', '?')}")
    print(f"  Accepted       : {r.get('accepted_new_conversations', '?')}")
    print(f"  Elapsed        : {r.get('elapsed_seconds', '?')}s")
    cost = r.get("estimated_cost_usd")
    print(f"  Est. cost      : {'${:.4f}'.format(cost) if cost else 'n/a'}")
    rej = {k: v for k, v in r.items() if k.startswith("rejected_") and v}
    if rej:
        print(f"  Rejections by type:")
        for k, v in sorted(rej.items(), key=lambda x: -x[1]):
            print(f"    {k:<40} {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined",
        default="output/combined_dataset.json",
        help="Path to combined_dataset.json",
    )
    parser.add_argument(
        "--new-from",
        type=int,
        required=True,
        help="First conversation ID number to treat as 'new batch' (e.g. 311)",
    )
    parser.add_argument("--top", type=int, default=15, help="Show top-N repeated messages")
    args = parser.parse_args()

    combined_path = Path(args.combined)
    if not combined_path.exists():
        print(f"Error: {combined_path} not found", file=sys.stderr)
        sys.exit(1)

    all_convs = _load(combined_path)
    new_convs = [c for c in all_convs if _id_num(c["conversation_id"]) >= args.new_from]

    if not new_convs:
        print(f"No conversations with ID >= he_sales_{args.new_from:06d} found.")
        sys.exit(0)

    analyse(new_convs, f"New batch (he_sales_{args.new_from:06d}+)", top_n=args.top)
    load_generation_report(Path("reports"))


if __name__ == "__main__":
    main()
