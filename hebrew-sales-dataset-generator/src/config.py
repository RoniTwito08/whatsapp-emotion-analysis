"""Configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"
SOURCE_DATASET_NAME = "synthetic_hebrew_whatsapp_sales_interest_v2"
QUALITY_STATUS = "pilot_v2"
CHANNEL = "whatsapp"
LANGUAGE = "he"
LOCALE = "he-IL"
ID_FORMAT = "he_sales_{n:06d}"


@dataclass
class Config:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    target_total_count: int = field(
        default_factory=lambda: int(os.getenv("TARGET_TOTAL_COUNT", "5000"))
    )
    generation_batch_size: int = field(
        default_factory=lambda: int(os.getenv("GENERATION_BATCH_SIZE", "10"))
    )
    max_retries_per_conversation: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES_PER_CONVERSATION", "5"))
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    )
    message_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("MESSAGE_SIMILARITY_THRESHOLD", "88"))
    )
    conversation_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONVERSATION_SIMILARITY_THRESHOLD", "82"))
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def has_api_key(self) -> bool:
        provider = self.llm_provider.lower()
        if provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key.strip())
        return False

    def validate_for_generation(self) -> None:
        if not self.has_api_key():
            provider = self.llm_provider.upper()
            raise ValueError(
                f"{provider}_API_KEY is not set. Add it to your .env file or environment."
            )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
