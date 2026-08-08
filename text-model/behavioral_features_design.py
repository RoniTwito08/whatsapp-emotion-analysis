"""behavioral_features_design.py

Conversation-level behavioral feature extraction for Model C and Model D.

Extracts ~42 scalar features per conversation from the messages array,
using only information that is realistically observable at inference time
(no text content, no target-derived fields).

--- Leakage analysis ---

SAFE features (derived from message structure and text properties):
    All features in this module are computed from:
      - message role (customer/business)
      - message_length_chars  (count of characters, from text)
      - question_count        (count of "?", from text)
      - emoji_count           (count of emoji characters, from text)
      - contains_price        (Hebrew price regex on text)
      - contains_negotiation  (Hebrew negotiation regex on text)
      - contains_objection    (Hebrew objection regex on text)
      - contains_commitment   (Hebrew commitment regex on text)
      - contains_delay_signal (Hebrew delay regex on text)
      - response_delay_minutes (time between messages — observable)

UNSAFE — must never be used as model input:
    - interest_score       per message — assigned by the generator
                           KNOWING the final_outcome. Not derivable
                           from message text alone.
    - interest_label       per message — same; pre-assigned synthetic label.
    - final_interest_score conversation-level — encodes outcome.
    - initial_interest_score — nearly uniform across classes (mean ~0.703
                           for both interested and losing_interest);
                           excluded because it has a mild synthetic look-back
                           smell (the generator set it knowing the trajectory).
    - interest_trajectory  — directly encodes the full outcome path.
    - final_outcome        — the label itself.

--- Note on synthetic data ---

All behavioral_features fields are computed from message text via regex
(see feature_calculator.py in the generator). They are NOT assigned from
the label. However, the LLM generated conversations with the full scenario
(including the planned outcome) in mind, so the text naturally contains
very clean commitment/objection signals. Performance on synthetic data
will likely exceed real-world performance because these patterns are more
stereotyped than in natural conversations.

--- Feature groups ---

  Group 1: Conversation-level counts          (4 features)
  Group 2: Customer message length stats      (6 features)
  Group 3: Customer message content counts    (9 features)
  Group 4: Per-message rates                  (4 features)
  Group 5: Response timing                    (5 features)
  Group 6: Session structure                  (2 features)
  Group 7: First-half vs second-half change   (6 features)
  Group 8: Conversation-ending pattern        (6 features)
                                     Total: ~42 features
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Optional

# Fields that must never be used as model inputs (target-derived or leaky)
FORBIDDEN_FIELDS: frozenset = frozenset({
    "interest_score",
    "interest_label",
    "final_interest_score",
    "initial_interest_score",
    "interest_trajectory",
    "final_outcome",
})


def extract_behavioral_features(
    conversation: Mapping[str, Any],
    messages_field: str = "messages",
    role_field: str = "role",
    customer_role: str = "customer",
    business_role: str = "business",
) -> Dict[str, float]:
    """Extract conversation-level behavioral features.

    Args:
        conversation: A single conversation dict from the corpus.
        messages_field: Key holding the list of messages.
        role_field: Key inside each message holding the sender role.
        customer_role: Value indicating a customer message.
        business_role: Value indicating a business message.

    Returns:
        Ordered dict of feature_name -> float. Booleans are 0.0 / 1.0.
        Returns an empty dict if the messages list is missing or malformed.

    Note:
        interest_score and interest_label are intentionally excluded.
        See FORBIDDEN_FIELDS and the module docstring.
    """
    messages = conversation.get(messages_field, [])
    if not isinstance(messages, list) or not messages:
        return {}

    customer_msgs = [m for m in messages if isinstance(m, dict) and m.get(role_field) == customer_role]
    business_msgs = [m for m in messages if isinstance(m, dict) and m.get(role_field) == business_role]

    n_customer = len(customer_msgs)
    n_business = len(business_msgs)
    n_total = len(messages)

    def _bf(msg: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        val = (msg.get("behavioral_features") or {}).get(key, default)
        return float(val) if val is not None else default

    def _safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _safe_median(values: List[float]) -> float:
        return statistics.median(values) if values else 0.0

    def _safe_std(values: List[float]) -> float:
        return statistics.stdev(values) if len(values) >= 2 else 0.0

    def _rate(count: float, n: int) -> float:
        return count / n if n > 0 else 0.0

    # --- Customer message length aggregates ---
    customer_lengths = [_bf(m, "message_length_chars") for m in customer_msgs]

    # --- Customer behavioral feature totals ---
    total_questions = sum(_bf(m, "question_count") for m in customer_msgs)
    total_emojis = sum(_bf(m, "emoji_count") for m in customer_msgs)
    total_price = sum(_bf(m, "contains_price") for m in customer_msgs)
    total_negotiation = sum(_bf(m, "contains_negotiation") for m in customer_msgs)
    total_objection = sum(_bf(m, "contains_objection") for m in customer_msgs)
    total_commitment = sum(_bf(m, "contains_commitment") for m in customer_msgs)
    total_delay = sum(_bf(m, "contains_delay_signal") for m in customer_msgs)

    # --- Response delays ---
    customer_delays = [float(m.get("response_delay_minutes") or 0) for m in customer_msgs]
    business_delays = [float(m.get("response_delay_minutes") or 0) for m in business_msgs]
    all_delays = [float(m.get("response_delay_minutes") or 0) for m in messages]

    # --- Session count (consecutive messages with >60 min gap = new session) ---
    session_count = 1
    max_gap = 0.0
    for delay in all_delays[1:]:
        if delay > 60:
            session_count += 1
        max_gap = max(max_gap, delay)

    # --- First half / second half split ---
    midpoint = n_total // 2
    first_half = messages[:midpoint] if n_total > 1 else []
    second_half = messages[midpoint:] if n_total > 1 else messages

    first_half_customer = [m for m in first_half if m.get(role_field) == customer_role]
    second_half_customer = [m for m in second_half if m.get(role_field) == customer_role]

    # Structural change features (pure — no text/regex)
    first_half_cust_lengths = [_bf(m, "message_length_chars") for m in first_half_customer]
    second_half_cust_lengths = [_bf(m, "message_length_chars") for m in second_half_customer]
    first_half_delays = [float(m.get("response_delay_minutes") or 0) for m in first_half]
    second_half_delays = [float(m.get("response_delay_minutes") or 0) for m in second_half]
    fh_mean_len = _safe_mean(first_half_cust_lengths)
    sh_mean_len = _safe_mean(second_half_cust_lengths)
    fh_mean_delay = _safe_mean(first_half_delays)
    sh_mean_delay = _safe_mean(second_half_delays)

    first_half_commitment = sum(_bf(m, "contains_commitment") for m in first_half_customer)
    first_half_objection = sum(_bf(m, "contains_objection") for m in first_half_customer)
    second_half_commitment = any(_bf(m, "contains_commitment") for m in second_half_customer)
    second_half_objection = any(_bf(m, "contains_objection") for m in second_half_customer)
    second_half_price = any(_bf(m, "contains_price") for m in second_half_customer)
    second_half_negotiation = any(_bf(m, "contains_negotiation") for m in second_half_customer)

    # --- Last customer message ---
    last_cust = customer_msgs[-1] if customer_msgs else None

    features: Dict[str, float] = {
        # --- Group 1: Conversation-level counts ---
        "total_messages": float(n_total),
        "customer_messages": float(n_customer),
        "business_messages": float(n_business),
        "ratio_customer_to_business": _rate(n_customer, n_business),

        # --- Group 2: Customer message length stats ---
        "mean_customer_msg_length": _safe_mean(customer_lengths),
        "median_customer_msg_length": _safe_median(customer_lengths),
        "std_customer_msg_length": _safe_std(customer_lengths),
        "min_customer_msg_length": min(customer_lengths, default=0.0),
        "max_customer_msg_length": max(customer_lengths, default=0.0),
        "total_customer_chars": sum(customer_lengths),

        # --- Group 3: Customer message content counts ---
        "total_customer_questions": total_questions,
        "total_customer_emojis": total_emojis,
        "customer_price_mentions": total_price,
        "customer_negotiation_mentions": total_negotiation,
        "customer_objection_mentions": total_objection,
        "customer_commitment_mentions": total_commitment,
        "customer_delay_signals": total_delay,
        "total_customer_bool_signals": total_price + total_negotiation + total_objection + total_commitment + total_delay,

        # --- Group 4: Per-message rates ---
        "question_rate": _rate(total_questions, n_customer),
        "emoji_rate": _rate(total_emojis, n_customer),
        "commitment_rate": _rate(total_commitment, n_customer),
        "objection_rate": _rate(total_objection, n_customer),
        "negotiation_rate": _rate(total_negotiation, n_customer),
        "delay_signal_rate": _rate(total_delay, n_customer),

        # --- Group 5: Response timing ---
        "mean_response_delay_customer": _safe_mean(customer_delays),
        "median_response_delay_customer": _safe_median(customer_delays),
        "max_response_delay_customer": max(customer_delays, default=0.0),
        "mean_response_delay_business": _safe_mean(business_delays),
        "last_response_delay": float(messages[-1].get("response_delay_minutes") or 0),

        # --- Group 6: Session structure ---
        "session_count": float(session_count),
        "max_session_gap_minutes": float(max_gap),

        # --- Group 7: First-half vs second-half change ---
        "first_half_commitment_count": float(first_half_commitment),
        "first_half_objection_count": float(first_half_objection),
        "commitment_in_second_half": float(second_half_commitment),
        "objection_in_second_half": float(second_half_objection),
        "price_in_second_half": float(second_half_price),
        "negotiation_in_second_half": float(second_half_negotiation),

        # --- Group 8: Conversation-ending pattern ---
        "last_role_is_customer": float(
            messages[-1].get(role_field) == customer_role if messages else False
        ),
        "last_customer_has_commitment": float(_bf(last_cust, "contains_commitment") if last_cust else 0.0),
        "last_customer_has_objection": float(_bf(last_cust, "contains_objection") if last_cust else 0.0),
        "last_customer_has_question": float(_bf(last_cust, "question_count") > 0 if last_cust else 0.0),
        "last_customer_has_delay_signal": float(_bf(last_cust, "contains_delay_signal") if last_cust else 0.0),
        "last_customer_has_price": float(_bf(last_cust, "contains_price") if last_cust else 0.0),

        # --- Group 9: Structural change (pure — no text/regex) ---
        "first_half_mean_customer_length": fh_mean_len,
        "second_half_mean_customer_length": sh_mean_len,
        "customer_length_change": sh_mean_len - fh_mean_len,
        "first_half_mean_response_delay": fh_mean_delay,
        "second_half_mean_response_delay": sh_mean_delay,
        "response_delay_change": sh_mean_delay - fh_mean_delay,
        "first_half_customer_message_count": float(len(first_half_customer)),
        "second_half_customer_message_count": float(len(second_half_customer)),
    }

    return features


def feature_names() -> List[str]:
    """Return the ordered list of feature names produced by extract_behavioral_features."""
    dummy: Dict[str, Any] = {
        "messages": [
            {
                "role": "customer",
                "text": "x",
                "response_delay_minutes": 0,
                "behavioral_features": {k: 0 for k in [
                    "message_length_chars", "question_count", "emoji_count",
                    "contains_price", "contains_negotiation", "contains_objection",
                    "contains_commitment", "contains_delay_signal",
                ]},
            },
            {
                "role": "business",
                "text": "y",
                "response_delay_minutes": 0,
                "behavioral_features": {},
            },
        ]
    }
    return list(extract_behavioral_features(dummy).keys())


def feature_count() -> int:
    return len(feature_names())


# ---------------------------------------------------------------------------
# Pure behavioral feature set (structural/timing only — no lexical signals)
# ---------------------------------------------------------------------------

# Features that inspect message TEXT content via regex or character patterns.
# These are excluded from the pure-behavioral experiment.
LEXICAL_FEATURE_NAMES: frozenset = frozenset({
    "total_customer_questions",
    "total_customer_emojis",
    "customer_price_mentions",
    "customer_negotiation_mentions",
    "customer_objection_mentions",
    "customer_commitment_mentions",
    "customer_delay_signals",
    "total_customer_bool_signals",
    "question_rate",
    "emoji_rate",
    "commitment_rate",
    "objection_rate",
    "negotiation_rate",
    "delay_signal_rate",
    "first_half_commitment_count",
    "first_half_objection_count",
    "commitment_in_second_half",
    "objection_in_second_half",
    "price_in_second_half",
    "negotiation_in_second_half",
    "last_customer_has_commitment",
    "last_customer_has_objection",
    "last_customer_has_question",
    "last_customer_has_delay_signal",
    "last_customer_has_price",
})

# Ordered list of pure structural/timing features (no text inspection).
PURE_BEHAVIORAL_FEATURE_NAMES: List[str] = [
    f for f in feature_names() if f not in LEXICAL_FEATURE_NAMES
]


def pure_behavioral_feature_names() -> List[str]:
    """Return only structural/timing features — no lexical/regex signals."""
    return PURE_BEHAVIORAL_FEATURE_NAMES
