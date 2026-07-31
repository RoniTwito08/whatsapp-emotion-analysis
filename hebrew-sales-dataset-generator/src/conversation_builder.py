"""Build complete conversation objects from LLM output + plan."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from .config import (
    CHANNEL,
    LANGUAGE,
    LOCALE,
    QUALITY_STATUS,
    SOURCE_DATASET_NAME,
)
from .feature_calculator import calculate_behavioral_features
from .models import (
    INTEREST_LABEL_SCORE_RANGES,
    TRAJECTORY_TO_OUTCOME,
    ConversationPlan,
    LLMMessageItem,
    LLMResponse,
)

TRAJECTORY_LABEL_SEQUENCES: dict[str, list[str]] = {
    "high_to_conversion":        ["interested", "interested", "high_interest", "converted", "converted"],
    "high_to_price_rejection":   ["interested", "interested", "losing_interest", "rejected"],
    "high_to_ghosting":          ["interested", "interested", "high_interest"],
    "high_to_delivery_loss":     ["interested", "interested", "high_interest", "losing_interest", "rejected"],
    "high_to_trust_loss":        ["interested", "interested", "high_interest", "losing_interest", "rejected"],
    "medium_to_discount_conversion": ["interested", "interested", "uncertain", "reengaged", "converted", "converted"],
    "medium_to_competitor_loss": ["interested", "interested", "uncertain", "losing_interest"],
    "medium_to_pending":         ["interested", "interested", "uncertain", "uncertain", "uncertain"],
    "low_to_appointment":        ["interested", "interested", "uncertain", "reengaged", "converted"],
    "low_to_reengagement":       ["interested", "interested", "losing_interest", "reengaged", "interested"],
}

_WEEKDAY_HOUR_WEIGHTS = {
    7: 3, 8: 8, 9: 15, 10: 15, 11: 12, 12: 10,
    13: 8, 14: 10, 15: 12, 16: 12, 17: 10, 18: 8,
    19: 5, 20: 3, 21: 2, 22: 1,
}

_TIMING_DELAY_RANGES = {
    "החלפה חיה מהירה עם השהיות של דקות בודדות": (1, 8),
    "שיחה עם פסקות של שעות מספר": (30, 240),
    "המשך ביום שלמחרת": (480, 1440),
    "לקוח נעלם ומגיע בחזרה": (60, 360),
    "עסק מעקב אחרי יומיים": (60, 240),
    "פסקת סוף שבוע": (1440, 4320),
    "פסקה ארוכה לפני דחייה סופית": (120, 600),
    "שיחה חיה עם כמה שאלות ברצף": (1, 5),
}


def _assign_customer_labels(trajectory: str, customer_count: int) -> list[str]:
    template = TRAJECTORY_LABEL_SEQUENCES.get(trajectory, ["interested"] * 5)
    if customer_count <= 0:
        return []
    if customer_count == 1:
        return [template[0]]

    result: list[str] = []
    n_phases = len(template)

    phase_1 = max(1, int(customer_count * 0.3))
    phase_last = max(1, min(2, customer_count - phase_1))
    phase_middle = max(0, customer_count - phase_1 - phase_last)

    if customer_count <= n_phases:
        padded = list(template)
        while len(padded) < customer_count:
            padded.insert(-1, padded[-2])
        return padded[:customer_count]

    label_first = template[0]
    label_middle_group = template[len(template) // 2] if len(template) > 2 else template[0]
    label_last = template[-1]

    result.extend([label_first] * phase_1)
    if phase_middle > 0:
        mid_labels = template[1:-1] if len(template) > 2 else [template[0]]
        per_mid = max(1, phase_middle // max(len(mid_labels), 1))
        for i, lbl in enumerate(mid_labels):
            count = per_mid if i < len(mid_labels) - 1 else phase_middle - per_mid * (len(mid_labels) - 1)
            result.extend([lbl] * max(1, count))
        result = result[: phase_1 + phase_middle]
    result.extend([label_last] * phase_last)

    return result[:customer_count]


def _score_for_label(label: str, rng: random.Random, initial_score: float | None = None) -> float | None:
    if label == "not_applicable":
        return None
    if label == "interested" and initial_score is not None:
        return round(initial_score + rng.uniform(-0.02, 0.02), 2)
    lo, hi = INTEREST_LABEL_SCORE_RANGES.get(label, (0.5, 0.7))
    return round(rng.uniform(lo, hi), 2)


def _next_business_time(dt: datetime, rng: random.Random) -> datetime:
    """Advance dt to the next weekday morning if it's outside working hours."""
    hours = list(_WEEKDAY_HOUR_WEIGHTS.keys())
    weights = list(_WEEKDAY_HOUR_WEIGHTS.values())
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
        dt = dt.replace(hour=rng.choices(hours, weights=weights, k=1)[0], minute=rng.randint(0, 59))
    if dt.hour >= 23:
        dt += timedelta(days=1)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        dt = dt.replace(hour=rng.choices(hours, weights=weights, k=1)[0], minute=rng.randint(0, 59))
    return dt


def _generate_timestamps(
    messages: list[dict[str, Any]],
    timing_pattern: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    hours = list(_WEEKDAY_HOUR_WEIGHTS.keys())
    weights = list(_WEEKDAY_HOUR_WEIGHTS.values())
    start_hour = rng.choices(hours, weights=weights, k=1)[0]
    start_minute = rng.randint(0, 59)

    month = rng.randint(1, 12)
    # Pick a day that's valid for the month (use 28 as safe max)
    base_date = datetime(2026, month, rng.randint(1, 28), start_hour, start_minute)
    base_date = _next_business_time(base_date, rng)

    default_range = _TIMING_DELAY_RANGES.get(timing_pattern, (2, 30))
    prev_dt = base_date

    for i, msg in enumerate(messages):
        if i == 0:
            raw_delay = rng.randint(3, 20)
            curr_dt = prev_dt + timedelta(minutes=raw_delay)
        else:
            prev_role = messages[i - 1]["role"]
            curr_role = msg["role"]

            if curr_role == "business" and prev_role == "business":
                raw_delay = rng.randint(60, 1440)
            elif "נעלם" in timing_pattern and i > len(messages) * 0.6:
                raw_delay = rng.randint(240, 2880)
            else:
                lo, hi = default_range
                raw_delay = rng.randint(max(1, lo), hi)

            curr_dt = prev_dt + timedelta(minutes=raw_delay)
            curr_dt = _next_business_time(curr_dt, rng)

        # Ensure strictly forward
        if curr_dt <= prev_dt:
            curr_dt = prev_dt + timedelta(minutes=1)

        # Compute the ACTUAL delay from the previous timestamp
        actual_delay = int((curr_dt - prev_dt).total_seconds() / 60)
        if actual_delay < 1:
            actual_delay = 1
            curr_dt = prev_dt + timedelta(minutes=1)

        msg["response_delay_minutes"] = actual_delay
        msg["timestamp"] = curr_dt.strftime("%Y-%m-%dT%H:%M")
        prev_dt = curr_dt

    return messages


def build_conversation(
    plan: ConversationPlan,
    llm_response: LLMResponse,
    rng: random.Random,
    generator_seed: int,
) -> dict[str, Any]:
    raw_messages = llm_response.messages
    trajectory = plan.trajectory
    outcome = TRAJECTORY_TO_OUTCOME.get(trajectory, "pending")

    customer_messages = [m for m in raw_messages if m.role == "customer"]
    customer_count = len(customer_messages)

    initial_score = round(rng.uniform(0.60, 0.80), 2)
    customer_labels = _assign_customer_labels(trajectory, customer_count)

    customer_label_iter = iter(customer_labels)
    built_messages: list[dict[str, Any]] = []

    for i, msg in enumerate(raw_messages):
        if msg.role == "customer":
            label = next(customer_label_iter, "interested")
            score = _score_for_label(label, rng, initial_score if label == "interested" else None)
        else:
            label = "not_applicable"
            score = None

        features = calculate_behavioral_features(msg.text)

        built_messages.append(
            {
                "message_index": i,
                "role": msg.role,
                "text": msg.text,
                "response_delay_minutes": 0,
                "interest_label": label,
                "interest_score": score,
                "behavioral_features": features,
                "timestamp": "",
            }
        )

    built_messages = _generate_timestamps(built_messages, plan.timing_pattern, rng)

    customer_scores = [m["interest_score"] for m in built_messages if m["role"] == "customer"]
    final_score = customer_scores[-1] if customer_scores else initial_score

    conversation: dict[str, Any] = {
        "conversation_id": plan.conversation_id,
        "source_dataset": SOURCE_DATASET_NAME,
        "language": LANGUAGE,
        "locale": LOCALE,
        "synthetic": True,
        "domain": plan.domain,
        "domain_he": plan.domain_he,
        "product_or_service": plan.product_or_service,
        "customer_persona": plan.customer_persona,
        "business_style": plan.business_style,
        "interest_trajectory": plan.trajectory,
        "initial_interest_score": initial_score,
        "final_interest_score": round(final_score, 2) if final_score is not None else 0.0,
        "final_outcome": outcome,
        "messages": built_messages,
        "metadata": {
            "channel": CHANNEL,
            "contains_personal_data": False,
            "quality_status": QUALITY_STATUS,
            "generator_seed": generator_seed,
        },
    }

    return conversation
