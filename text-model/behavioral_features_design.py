"""behavioral_features_design.py

Design sketch for conversation-level behavioral feature extraction.

This module describes (and partially implements) how to aggregate the
per-message behavioral features already present in the dataset into
conversation-level feature vectors suitable for Model C
(Behavioral Feature Classifier) and Model D (AlephBERT + Behavioral Features).

It is intentionally NOT connected to the training pipeline yet.
Include it when behavioral fusion experiments begin.

--- Research context ---

The dataset contains per-message behavioral features:
    message['behavioral_features'] = {
        "message_length_chars": int,
        "question_count": int,
        "emoji_count": int,
        "contains_price": bool,
        "contains_negotiation": bool,
        "contains_objection": bool,
        "contains_commitment": bool,
        "contains_delay_signal": bool,
    }

And per-message metadata:
    message['response_delay_minutes']: float
    message['interest_score']: float      *** LEAKS TARGET — do not use ***
    message['interest_label']: str        *** LEAKS TARGET — do not use ***
    message['timestamp']: str

--- Leakage analysis ---

SAFE to use as model inputs (available in a real-world deployment):
    - message_length_chars     (observable at inference time)
    - question_count           (observable)
    - emoji_count              (observable)
    - contains_price           (observable)
    - contains_negotiation     (observable)
    - contains_objection       (observable)
    - contains_commitment      (observable — this IS correlated with outcome
                                but it's a pattern in the text, not the label)
    - contains_delay_signal    (observable)
    - response_delay_minutes   (observable — time between messages)
    - message count per role   (observable)
    - who sent the last message (observable)
    - timestamp patterns       (time-of-day, session gaps — observable)

UNSAFE to use (directly encode or are correlated with the target label):
    - interest_score           *** LEAKS — synthetic score assigned by the
                                generator with full knowledge of final_outcome ***
    - interest_label           *** LEAKS — per-message classification
                                that already knows the final label ***
    - final_interest_score     *** LEAKS — top-level conversation field ***
    - initial_interest_score   *** MARGINAL — may encode outcome indirectly ***
    - interest_trajectory      *** LEAKS — encodes the full outcome path ***

--- Aggregation strategy ---

For a conversation with N messages, the following features can be computed
without any look-ahead or label leakage:

Group 1: Conversation-level counts
    - total_messages
    - customer_messages
    - business_messages
    - ratio_customer_to_business

Group 2: Customer message content aggregates
    - mean_customer_msg_length
    - max_customer_msg_length
    - total_customer_chars
    - total_customer_questions
    - total_customer_emojis
    - customer_price_mentions        (count)
    - customer_negotiation_mentions  (count)
    - customer_objection_mentions    (count)
    - customer_commitment_mentions   (count — correlated but observable)
    - customer_delay_signals         (count)

Group 3: Response timing
    - mean_response_delay_minutes_customer
    - max_response_delay_minutes_customer
    - mean_response_delay_minutes_business
    - last_response_delay_minutes
    - session_count                  (gaps > 1 hour → new session)

Group 4: Message trajectory (first half vs. second half of conversation)
    - commitment_in_second_half      (bool)
    - objection_in_second_half       (bool)
    - increasing_message_length      (bool — trend)

Group 5: Conversation-ending pattern
    - last_role_to_message           ('customer' or 'business')
    - last_customer_msg_has_question (bool)
    - last_customer_msg_has_commitment (bool)

--- Feature vector size ---
Approximately 25-30 scalar features per conversation, all computable from
the 'messages' array without using interest_score, interest_label, or
any outcome-correlated metadata field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


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
        Dict of feature_name -> float. All values are numeric;
        booleans are encoded as 0.0 / 1.0.

    Note:
        interest_score and interest_label are intentionally excluded to
        prevent target leakage. See module docstring for the full leakage
        analysis.
    """
    messages = conversation.get(messages_field, [])
    if not isinstance(messages, list):
        return {}

    customer_msgs = [m for m in messages if isinstance(m, dict) and m.get(role_field) == customer_role]
    business_msgs = [m for m in messages if isinstance(m, dict) and m.get(role_field) == business_role]

    def _bf(msg: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        val = (msg.get("behavioral_features") or {}).get(key, default)
        return float(val) if val is not None else default

    def _safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    customer_lengths = [_bf(m, "message_length_chars") for m in customer_msgs]
    customer_delays = [float(m.get("response_delay_minutes") or 0) for m in customer_msgs]
    business_delays = [float(m.get("response_delay_minutes") or 0) for m in business_msgs]

    n_msgs = len(messages)
    midpoint = n_msgs // 2
    second_half = messages[midpoint:] if n_msgs > 1 else messages

    features: Dict[str, float] = {
        # Counts
        "total_messages": float(len(messages)),
        "customer_messages": float(len(customer_msgs)),
        "business_messages": float(len(business_msgs)),
        "ratio_customer_to_business": (
            len(customer_msgs) / len(business_msgs) if business_msgs else float(len(customer_msgs))
        ),

        # Content aggregates
        "mean_customer_msg_length": _safe_mean(customer_lengths),
        "max_customer_msg_length": max(customer_lengths, default=0.0),
        "total_customer_chars": sum(customer_lengths),
        "total_customer_questions": sum(_bf(m, "question_count") for m in customer_msgs),
        "total_customer_emojis": sum(_bf(m, "emoji_count") for m in customer_msgs),
        "customer_price_mentions": sum(_bf(m, "contains_price") for m in customer_msgs),
        "customer_negotiation_mentions": sum(_bf(m, "contains_negotiation") for m in customer_msgs),
        "customer_objection_mentions": sum(_bf(m, "contains_objection") for m in customer_msgs),
        "customer_commitment_mentions": sum(_bf(m, "contains_commitment") for m in customer_msgs),
        "customer_delay_signals": sum(_bf(m, "contains_delay_signal") for m in customer_msgs),

        # Response timing
        "mean_response_delay_customer": _safe_mean(customer_delays),
        "max_response_delay_customer": max(customer_delays, default=0.0),
        "mean_response_delay_business": _safe_mean(business_delays),
        "last_response_delay": float(messages[-1].get("response_delay_minutes") or 0) if messages else 0.0,

        # Trajectory (second half)
        "commitment_in_second_half": float(
            any(_bf(m, "contains_commitment") for m in second_half if m.get(role_field) == customer_role)
        ),
        "objection_in_second_half": float(
            any(_bf(m, "contains_objection") for m in second_half if m.get(role_field) == customer_role)
        ),

        # Conversation-ending pattern
        "last_role_is_customer": float(
            messages[-1].get(role_field) == customer_role if messages else False
        ),
        "last_customer_has_commitment": float(
            _bf(customer_msgs[-1], "contains_commitment") if customer_msgs else 0.0
        ),
        "last_customer_has_objection": float(
            _bf(customer_msgs[-1], "contains_objection") if customer_msgs else 0.0
        ),
        "last_customer_has_question": float(
            _bf(customer_msgs[-1], "question_count") > 0 if customer_msgs else 0.0
        ),
    }

    return features


def feature_names() -> List[str]:
    """Return the ordered list of feature names produced by extract_behavioral_features."""
    dummy: Dict[str, Any] = {
        "messages": [
            {"role": "customer", "text": "x", "response_delay_minutes": 0,
             "behavioral_features": {k: 0 for k in [
                 "message_length_chars", "question_count", "emoji_count",
                 "contains_price", "contains_negotiation", "contains_objection",
                 "contains_commitment", "contains_delay_signal",
             ]}},
        ]
    }
    return list(extract_behavioral_features(dummy).keys())
