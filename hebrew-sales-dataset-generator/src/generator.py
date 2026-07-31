"""Main orchestrator for dataset generation."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint_manager import (
    Checkpoint,
    append_rejection_record,
    load_checkpoint,
    load_generated,
    save_checkpoint,
    save_combined,
    save_generated,
)
from .config import Config, get_config
from .conversation_builder import build_conversation
from .dataset_loader import (
    build_message_text_index,
    get_domain_distribution,
    get_max_id,
    get_outcome_distribution,
    get_trajectory_distribution,
    load_corpus,
    make_next_id,
)
from .llm_client import LLMClient, LLMResult, MockLLMClient, create_client
from .models import ConversationPlan, GenerationStats, LLMResponse
from .prompt_builder import (
    FORBIDDEN_PHRASES,
    build_system_prompt,
    build_user_prompt,
    sample_recent_messages,
)
from .scenario_planner import create_conversation_plan
from .similarity_checker import SimilarityChecker
from .validator import validate_conversation

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"

SOURCE_PATH = DATA_DIR / "corpus_300.json"
GENERATED_PATH = OUTPUT_DIR / "generated_conversations.json"
COMBINED_PATH = OUTPUT_DIR / "combined_dataset.json"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
GENERATION_REPORT_PATH = REPORTS_DIR / "generation_report.json"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"
REJECTED_PATH = REPORTS_DIR / "rejected_conversations.jsonl"


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM output, handling markdown code fences."""
    text = raw.strip()
    for fence in ["```json", "```"]:
        if fence in text:
            start = text.index(fence) + len(fence)
            end = text.rfind("```", start)
            text = text[start:end].strip()
            break
    return json.loads(text)


def _validate_llm_output(data: dict[str, Any], expected_count: int) -> LLMResponse:
    """Validate and parse LLM output dict into LLMResponse."""
    if "messages" not in data:
        raise ValueError("LLM response missing 'messages' key")
    messages_raw = data["messages"]
    if not isinstance(messages_raw, list) or not messages_raw:
        raise ValueError("LLM 'messages' must be a non-empty list")

    from .models import LLMMessageItem

    items: list[LLMMessageItem] = []
    for i, m in enumerate(messages_raw):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] is not a dict")
        role = m.get("role", "")
        text = m.get("text", "")
        if role not in ("customer", "business"):
            raise ValueError(f"messages[{i}] invalid role: {role!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"messages[{i}] empty text")
        items.append(LLMMessageItem(role=role, text=text.strip()))

    actual_count = len(items)
    if actual_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} messages, LLM returned {actual_count}"
        )

    actual_outcome = str(data.get("actual_outcome", "")).strip()
    trajectory_notes = str(data.get("trajectory_notes", "")).strip()

    return LLMResponse(
        messages=items,
        actual_outcome=actual_outcome,
        trajectory_notes=trajectory_notes,
    )


class GenerationPipeline:
    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        source_conversations: list[dict[str, Any]],
        dry_run: bool = False,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.source = source_conversations
        self.dry_run = dry_run

        self.seed = seed if seed is not None else int(datetime.now(timezone.utc).timestamp())
        self.rng = random.Random(self.seed)

        self.checker = SimilarityChecker(
            message_threshold=config.message_similarity_threshold,
            conversation_threshold=config.conversation_similarity_threshold,
        )
        self.checker.load_existing(source_conversations)

        self.stats = GenerationStats(source_conversations=len(source_conversations))
        self.checkpoint = Checkpoint(random_seed=self.seed)
        self.generated: list[dict[str, Any]] = []
        self._used_signatures: set[str] = set()

    def resume_from_checkpoint(self) -> None:
        cp = load_checkpoint(CHECKPOINT_PATH)
        if cp is None:
            return
        self.checkpoint = cp
        self._used_signatures = set(cp.used_plan_signatures)
        self.generated = load_generated(GENERATED_PATH)
        for conv in self.generated:
            self.checker.accept_conversation(conv)
        logger.info(
            "Resumed: accepted=%d, rejected=%d",
            cp.accepted_count,
            cp.rejected_count,
        )

    def run(self, count: int) -> GenerationStats:
        t_start = time.monotonic()
        max_id = get_max_id(self.source + self.generated)
        accepted = self.checkpoint.accepted_count
        rejected_total = self.checkpoint.rejected_count

        domain_dist = get_domain_distribution(self.source + self.generated)
        traj_dist = get_trajectory_distribution(self.source + self.generated)
        outcome_dist = get_outcome_distribution(self.source + self.generated)

        system_prompt = build_system_prompt()

        while accepted < count:
            next_id = make_next_id(max_id + accepted)
            plan = create_conversation_plan(
                next_id=next_id,
                rng=self.rng,
                existing_domain_dist=domain_dist,
                existing_trajectory_dist=traj_dist,
                used_plan_signatures=self._used_signatures,
            )

            sig = f"{plan.domain}:{plan.product_or_service}:{plan.trajectory}"
            self._used_signatures.add(sig)

            if self.dry_run:
                logger.info("[DRY-RUN] Plan: %s", _format_plan(plan))
                accepted += 1
                continue

            conversation, rejection_reason = self._generate_one(plan, system_prompt, traj_dist)

            if conversation is not None:
                accepted += 1
                self.generated.append(conversation)
                self.checker.accept_conversation(conversation)
                domain_dist[conversation["domain"]] = domain_dist.get(conversation["domain"], 0) + 1
                traj_dist[conversation["interest_trajectory"]] = (
                    traj_dist.get(conversation["interest_trajectory"], 0) + 1
                )
                out = conversation["final_outcome"]
                outcome_dist[out] = outcome_dist.get(out, 0) + 1

                self.checkpoint.accepted_count = accepted
                self.checkpoint.last_accepted_id = conversation["conversation_id"]
                self.checkpoint.domain_distribution = domain_dist
                self.checkpoint.outcome_distribution = outcome_dist
                self.checkpoint.trajectory_distribution = traj_dist
                self.checkpoint.used_plan_signatures = list(self._used_signatures)

                _persist(self.source, self.generated, self.checkpoint)
                logger.info(
                    "[%d/%d] Accepted: %s (%s)",
                    accepted,
                    count,
                    conversation["conversation_id"],
                    conversation["final_outcome"],
                )
            else:
                rejected_total += 1
                self.checkpoint.rejected_count = rejected_total

        self.stats.accepted_new_conversations = accepted
        self.stats.total_conversations = len(self.source) + len(self.generated)
        self.stats.retry_count = self.checkpoint.retry_count
        self.stats.api_requests = self.checkpoint.api_request_count
        self.stats.estimated_input_tokens = self.checkpoint.input_tokens
        self.stats.estimated_output_tokens = self.checkpoint.output_tokens
        self.stats.elapsed_seconds = time.monotonic() - t_start
        self.stats.domain_distribution = domain_dist
        self.stats.outcome_distribution = outcome_dist
        self.stats.most_repeated_normalized_phrases = self.checker.most_repeated_phrases(10)
        self.stats.rejected_exact_duplicates = self.checkpoint.error_counts.get("exact_duplicate", 0)
        self.stats.rejected_similar_messages = self.checkpoint.error_counts.get("fuzzy_duplicate", 0)
        self.stats.rejected_similar_conversations = self.checkpoint.error_counts.get("similar_conversation", 0)
        self.stats.rejected_invalid_schema = self.checkpoint.error_counts.get("invalid_schema", 0)
        self.stats.rejected_gender_inconsistency = self.checkpoint.error_counts.get("gender_inconsistency", 0)
        self.stats.rejected_incoherent_outcome = self.checkpoint.error_counts.get("incoherent_outcome", 0)
        self.stats.rejected_wrong_message_count = self.checkpoint.error_counts.get("wrong_message_count", 0)

        return self.stats

    def _generate_one(
        self,
        plan: ConversationPlan,
        system_prompt: str,
        traj_dist: dict[str, int],
    ) -> tuple[dict[str, Any] | None, str | None]:
        max_retries = self.config.max_retries_per_conversation
        retry_feedback: str | None = None

        for attempt in range(max_retries):
            recent = sample_recent_messages(self.source[-30:] + self.generated[-20:], rng=self.rng)
            user_prompt = build_user_prompt(
                plan=plan,
                forbidden_samples=FORBIDDEN_PHRASES,
                recent_accepted_samples=recent,
                retry_feedback=retry_feedback,
            )

            try:
                result: LLMResult = self.llm_client.generate(system_prompt, user_prompt)
                self.checkpoint.api_request_count += 1
                self.checkpoint.input_tokens += result.usage.input_tokens
                self.checkpoint.output_tokens += result.usage.output_tokens

                if result.refusal:
                    rejection_reason = f"LLM refusal: {result.refusal[:100]}"
                    logger.warning("RETRY [attempt %d] reason: %s", attempt + 1, rejection_reason)
                    _log_rejection(plan, rejection_reason, None, attempt, REJECTED_PATH)
                    retry_feedback = "Model refused. Try different scenario framing."
                    self.checkpoint.retry_count += 1
                    continue

                if result.finish_reason == "length":
                    rejection_reason = "finish_reason=length: response truncated at token limit"
                    logger.warning("RETRY [attempt %d] reason: %s", attempt + 1, rejection_reason)
                    _log_rejection(plan, rejection_reason, result.content[:200], attempt, REJECTED_PATH)
                    retry_feedback = "Response was truncated. Return fewer messages (6-8 max)."
                    self.checkpoint.retry_count += 1
                    _increment_error(self.checkpoint, "invalid_schema")
                    continue

                raw_data = _parse_llm_response(result.content)
                llm_response = _validate_llm_output(raw_data, plan.message_count)

            except json.JSONDecodeError as exc:
                rejection_reason = f"Malformed JSON: {exc}"
                logger.warning("RETRY [attempt %d] reason: %s | content_preview=%r",
                               attempt + 1, rejection_reason,
                               (result.content[:120] if "result" in dir() else ""))
                _increment_error(self.checkpoint, "invalid_schema")
                _log_rejection(plan, rejection_reason, result.content if "result" in dir() else "", attempt, REJECTED_PATH)
                retry_feedback = "Return valid JSON only. No markdown, no explanation."
                self.checkpoint.retry_count += 1
                continue

            except ValueError as exc:
                rejection_reason = str(exc)
                logger.warning("RETRY [attempt %d] reason: %s", attempt + 1, rejection_reason)
                if "messages" in rejection_reason.lower():
                    _increment_error(self.checkpoint, "wrong_message_count")
                else:
                    _increment_error(self.checkpoint, "invalid_schema")
                _log_rejection(plan, rejection_reason, "", attempt, REJECTED_PATH)
                retry_feedback = f"Fix: {rejection_reason}"
                self.checkpoint.retry_count += 1
                continue

            except Exception as exc:
                rejection_reason = f"LLM error: {exc}"
                logger.warning("RETRY [attempt %d] reason: %s", attempt + 1, rejection_reason)
                _log_rejection(plan, rejection_reason, "", attempt, REJECTED_PATH)
                retry_feedback = "Try again."
                self.checkpoint.retry_count += 1
                continue

            conv = build_conversation(plan, llm_response, self.rng, self.seed)

            schema_errors = validate_conversation(conv)
            if schema_errors:
                rejection_reason = "Schema errors: " + "; ".join(schema_errors[:3])
                logger.warning("RETRY [attempt %d] reason: %s", attempt + 1, rejection_reason)
                gender_related = any("gender" in e.lower() or "slash" in e.lower() for e in schema_errors)
                if gender_related:
                    _increment_error(self.checkpoint, "gender_inconsistency")
                else:
                    _increment_error(self.checkpoint, "invalid_schema")
                _log_rejection(plan, rejection_reason, "", attempt, REJECTED_PATH)
                retry_feedback = "Fix schema issues: " + "; ".join(schema_errors[:2])
                self.checkpoint.retry_count += 1
                continue

            similarity_issues = self.checker.check_new_conversation(conv)
            if similarity_issues:
                issue_type, issue_desc, score = similarity_issues[0]
                rejection_reason = f"{issue_type}: {issue_desc}"
                logger.warning("RETRY [attempt %d] reason: %s", attempt + 1, rejection_reason)
                _increment_error(self.checkpoint, issue_type)
                _log_rejection(plan, rejection_reason, "", attempt, REJECTED_PATH, score)
                retry_feedback = "Too similar to existing content. Use completely different phrasing and scenario."
                self.checkpoint.retry_count += 1
                continue

            return conv, None

        _log_rejection(plan, f"Max retries ({max_retries}) exhausted", "", max_retries - 1, REJECTED_PATH)
        _increment_error(self.checkpoint, "max_retries")
        return None, "max_retries_exhausted"


def _format_plan(plan: ConversationPlan) -> str:
    return (
        f"{plan.conversation_id} | {plan.domain_he} | {plan.product_or_service} | "
        f"{plan.trajectory} | {plan.message_count} msgs | {plan.customer_persona}"
    )


def _increment_error(checkpoint: Checkpoint, key: str) -> None:
    checkpoint.error_counts[key] = checkpoint.error_counts.get(key, 0) + 1


def _log_rejection(
    plan: ConversationPlan,
    reason: str,
    raw_response: str,
    attempt: int,
    path: Path,
    similarity_score: float | None = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "planned_id": plan.conversation_id,
        "plan": plan.model_dump(),
        "rejection_reason": reason,
        "similarity_score": similarity_score,
        "matching_id": None,
        "raw_response_truncated": (raw_response or "")[:500],
        "retry_attempt": attempt,
    }
    try:
        append_rejection_record(record, path)
    except Exception as exc:
        logger.debug("Failed to log rejection: %s", exc)


def _persist(
    source: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    checkpoint: Checkpoint,
) -> None:
    try:
        save_generated(generated, GENERATED_PATH)
        save_combined(source, generated, COMBINED_PATH)
        save_checkpoint(checkpoint, CHECKPOINT_PATH)
    except Exception as exc:
        logger.error("Failed to persist state: %s", exc)


def run_generation(
    count: int | None = None,
    target_total: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    seed: int | None = None,
    config: Config | None = None,
    llm_client: LLMClient | None = None,
) -> GenerationStats:
    if config is None:
        config = get_config()

    source = load_corpus(SOURCE_PATH)
    n_source = len(source)

    if target_total is not None:
        count = max(0, target_total - n_source)
        if count == 0:
            logger.info("Dataset already has %d conversations. Nothing to generate.", n_source)
            return GenerationStats(
                source_conversations=n_source,
                total_conversations=n_source,
            )

    if count is None or count <= 0:
        raise ValueError("Must specify --count or --target-total")

    if llm_client is None:
        if not dry_run and not config.has_api_key():
            raise ValueError(
                "No API key configured. Set OPENAI_API_KEY in .env or use --dry-run."
            )
        if dry_run or not config.has_api_key():
            llm_client = MockLLMClient()
        else:
            llm_client = create_client(config)

    pipeline = GenerationPipeline(
        config=config,
        llm_client=llm_client,
        source_conversations=source,
        dry_run=dry_run,
        seed=seed,
    )

    if resume:
        pipeline.resume_from_checkpoint()

    pipeline.stats.requested_new_conversations = count
    stats = pipeline.run(count)
    return stats
