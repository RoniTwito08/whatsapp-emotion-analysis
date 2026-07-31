"""Pydantic models matching the exact corpus_300.json schema."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_OUTCOMES = frozenset(
    [
        "converted",
        "explicit_rejection",
        "ghosted",
        "appointment_set",
        "competitor_loss",
        "delivery_loss",
        "pending",
        "trust_loss",
        "reengaged_pending",
    ]
)

ALLOWED_TRAJECTORIES = frozenset(
    [
        "high_to_conversion",
        "high_to_price_rejection",
        "high_to_ghosting",
        "high_to_delivery_loss",
        "high_to_trust_loss",
        "medium_to_discount_conversion",
        "medium_to_competitor_loss",
        "medium_to_pending",
        "low_to_appointment",
        "low_to_reengagement",
    ]
)

TRAJECTORY_TO_OUTCOME: dict[str, str] = {
    "high_to_conversion": "converted",
    "high_to_price_rejection": "explicit_rejection",
    "high_to_ghosting": "ghosted",
    "high_to_delivery_loss": "delivery_loss",
    "high_to_trust_loss": "trust_loss",
    "medium_to_discount_conversion": "converted",
    "medium_to_competitor_loss": "competitor_loss",
    "medium_to_pending": "pending",
    "low_to_appointment": "appointment_set",
    "low_to_reengagement": "reengaged_pending",
}

ALLOWED_INTEREST_LABELS = frozenset(
    [
        "interested",
        "high_interest",
        "uncertain",
        "losing_interest",
        "reengaged",
        "converted",
        "rejected",
        "not_applicable",
    ]
)

ALLOWED_ROLES = frozenset(["customer", "business"])

INTEREST_LABEL_SCORE_RANGES: dict[str, tuple[float, float]] = {
    "interested": (0.60, 0.80),
    "high_interest": (0.82, 0.97),
    "uncertain": (0.40, 0.60),
    "losing_interest": (0.16, 0.34),
    "reengaged": (0.66, 0.85),
    "converted": (0.97, 1.00),
    "rejected": (0.00, 0.06),
}


class BehavioralFeatures(BaseModel):
    message_length_chars: int
    question_count: int
    emoji_count: int
    contains_price: bool
    contains_negotiation: bool
    contains_objection: bool
    contains_commitment: bool
    contains_delay_signal: bool


class Message(BaseModel):
    message_index: int
    role: Literal["customer", "business"]
    text: str
    response_delay_minutes: int
    interest_label: str
    interest_score: Optional[float]
    behavioral_features: BehavioralFeatures
    timestamp: str

    @field_validator("interest_label")
    @classmethod
    def validate_interest_label(cls, v: str) -> str:
        if v not in ALLOWED_INTEREST_LABELS:
            raise ValueError(f"Invalid interest_label: {v!r}")
        return v

    @model_validator(mode="after")
    def validate_score_consistency(self) -> "Message":
        if self.role == "business":
            if self.interest_label != "not_applicable":
                raise ValueError(
                    f"Business message must have interest_label='not_applicable', got {self.interest_label!r}"
                )
            if self.interest_score is not None:
                raise ValueError("Business message must have interest_score=null")
        else:
            if self.interest_label == "not_applicable":
                raise ValueError("Customer message cannot have interest_label='not_applicable'")
        return self


class Metadata(BaseModel):
    channel: Literal["whatsapp"] = "whatsapp"
    contains_personal_data: bool = False
    quality_status: str = "pilot_v2"
    generator_seed: int


class Conversation(BaseModel):
    conversation_id: str
    source_dataset: str
    language: Literal["he"] = "he"
    locale: Literal["he-IL"] = "he-IL"
    synthetic: Literal[True] = True
    domain: str
    domain_he: str
    product_or_service: str
    customer_persona: str
    business_style: str
    interest_trajectory: str
    initial_interest_score: float
    final_interest_score: float
    final_outcome: str
    messages: list[Message]
    metadata: Metadata

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str) -> str:
        import re

        if not re.match(r"^he_sales_\d{6}$", v):
            raise ValueError(f"Invalid conversation_id format: {v!r}")
        return v

    @field_validator("interest_trajectory")
    @classmethod
    def validate_trajectory(cls, v: str) -> str:
        if v not in ALLOWED_TRAJECTORIES:
            raise ValueError(f"Invalid interest_trajectory: {v!r}")
        return v

    @field_validator("final_outcome")
    @classmethod
    def validate_outcome(cls, v: str) -> str:
        if v not in ALLOWED_OUTCOMES:
            raise ValueError(f"Invalid final_outcome: {v!r}")
        return v

    def model_dump_ordered(self) -> dict[str, Any]:
        """Return dict with field ordering matching the original schema."""
        d = self.model_dump()
        ordered_keys = [
            "conversation_id",
            "source_dataset",
            "language",
            "locale",
            "synthetic",
            "domain",
            "domain_he",
            "product_or_service",
            "customer_persona",
            "business_style",
            "interest_trajectory",
            "initial_interest_score",
            "final_interest_score",
            "final_outcome",
            "messages",
            "metadata",
        ]
        result = {k: d[k] for k in ordered_keys if k in d}
        result["messages"] = [_order_message(m) for m in d["messages"]]
        return result


def _order_message(m: dict[str, Any]) -> dict[str, Any]:
    ordered_keys = [
        "message_index",
        "role",
        "text",
        "response_delay_minutes",
        "interest_label",
        "interest_score",
        "behavioral_features",
        "timestamp",
    ]
    result = {k: m[k] for k in ordered_keys if k in m}
    bf_keys = [
        "message_length_chars",
        "question_count",
        "emoji_count",
        "contains_price",
        "contains_negotiation",
        "contains_objection",
        "contains_commitment",
        "contains_delay_signal",
    ]
    bf = m.get("behavioral_features", {})
    result["behavioral_features"] = {k: bf[k] for k in bf_keys if k in bf}
    return result


class ConversationPlan(BaseModel):
    """Structured plan created before calling the LLM."""

    conversation_id: str
    domain: str
    domain_he: str
    specific_scenario: str
    product_or_service: str
    customer_gender: Literal["male", "female"]
    customer_persona: str
    customer_age_impression: str
    customer_technical_level: str
    business_style: str
    customer_goal: str
    main_concern: str
    secondary_concern: str
    starting_context: str
    trajectory: str
    turning_point: str
    ending: str
    message_count: int
    writing_style: str
    timing_pattern: str
    urgency: str
    budget_sensitivity: str
    trust_level: str
    objection_type: str


class LLMMessageItem(BaseModel):
    role: Literal["customer", "business"]
    text: str


class LLMResponse(BaseModel):
    """Flat response structure — easier for Structured Outputs validation."""

    messages: list[LLMMessageItem]
    actual_outcome: str
    trajectory_notes: str


class RejectionRecord(BaseModel):
    timestamp: str
    planned_id: str
    plan: dict[str, Any]
    rejection_reason: str
    similarity_score: Optional[float] = None
    matching_id: Optional[str] = None
    raw_response_truncated: Optional[str] = None
    retry_attempt: int = 0


class GenerationStats(BaseModel):
    source_conversations: int = 0
    requested_new_conversations: int = 0
    accepted_new_conversations: int = 0
    total_conversations: int = 0
    rejected_exact_duplicates: int = 0
    rejected_similar_messages: int = 0
    rejected_similar_conversations: int = 0
    rejected_invalid_schema: int = 0
    rejected_gender_inconsistency: int = 0
    rejected_incoherent_outcome: int = 0
    rejected_wrong_message_count: int = 0
    rejected_other: int = 0
    api_requests: int = 0
    retry_count: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    elapsed_seconds: float = 0.0
    domain_distribution: dict[str, int] = Field(default_factory=dict)
    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    conversation_length_distribution: dict[str, int] = Field(default_factory=dict)
    most_repeated_normalized_phrases: list[str] = Field(default_factory=list)
