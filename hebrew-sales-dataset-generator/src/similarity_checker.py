"""Multi-level duplicate and similarity detection."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any

from rapidfuzz import fuzz

from .dataset_loader import normalize_text

logger = logging.getLogger(__name__)

VERY_SHORT_OK = frozenset(["כן", "לא", "תודה", "אוקיי", "אוק", "בסדר", "ממש לא"])


class SimilarityChecker:
    def __init__(
        self,
        message_threshold: float = 88.0,
        conversation_threshold: float = 82.0,
    ) -> None:
        self.message_threshold = message_threshold
        self.conversation_threshold = conversation_threshold

        self._exact_index: dict[str, str] = {}
        self._length_buckets: defaultdict[int, list[tuple[str, str]]] = defaultdict(list)
        self._prefix_buckets: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        self._conv_tfidf_corpus: list[str] = []
        self._conv_ids: list[str] = []
        self._vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._tfidf_dirty: bool = False

        self._structure_signature_counts: Counter[str] = Counter()
        self._phrase_counts: Counter[str] = Counter()

    def load_existing(self, conversations: list[dict[str, Any]]) -> None:
        for conv in conversations:
            cid = conv["conversation_id"]
            for msg in conv.get("messages", []):
                text = msg.get("text", "")
                self._index_message(text, cid)
            self._index_conversation(conv)
        self._tfidf_dirty = True
        logger.info("Loaded %d conversations into similarity index", len(conversations))

    def _index_message(self, text: str, conv_id: str) -> None:
        norm = normalize_text(text)
        if not norm or norm in VERY_SHORT_OK:
            return
        self._exact_index[norm] = conv_id
        bucket_key = len(norm) // 10
        self._length_buckets[bucket_key].append((norm, conv_id))
        prefix = norm[:8] if len(norm) >= 8 else norm
        self._prefix_buckets[prefix].append((norm, conv_id))
        self._phrase_counts[norm] += 1

    def _index_conversation(self, conv: dict[str, Any]) -> None:
        doc = _conversation_to_doc(conv)
        self._conv_tfidf_corpus.append(doc)
        self._conv_ids.append(conv["conversation_id"])
        self._tfidf_dirty = True

    def _rebuild_tfidf(self) -> None:
        if not self._tfidf_dirty or not self._conv_tfidf_corpus:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=8000,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(self._conv_tfidf_corpus)
            self._tfidf_dirty = False
        except Exception as exc:
            logger.warning("TF-IDF rebuild failed: %s", exc)
            self._vectorizer = None
            self._tfidf_matrix = None

    def check_exact_duplicate(self, text: str) -> str | None:
        """Return the ID of the conversation containing this exact message, or None."""
        norm = normalize_text(text)
        if not norm or norm in VERY_SHORT_OK:
            return None
        return self._exact_index.get(norm)

    def check_fuzzy_duplicate(self, text: str) -> tuple[float, str | None]:
        """Return (best_score, matching_conv_id) for fuzzy message similarity."""
        norm = normalize_text(text)
        if not norm or len(norm) < 8 or norm in VERY_SHORT_OK:
            return 0.0, None

        candidates: dict[str, str] = {}
        bucket_key = len(norm) // 10
        for delta in range(-1, 2):
            for cand_norm, cid in self._length_buckets.get(bucket_key + delta, []):
                candidates[cand_norm] = cid

        prefix = norm[:8]
        for cand_norm, cid in self._prefix_buckets.get(prefix, []):
            candidates[cand_norm] = cid

        best_score = 0.0
        best_id: str | None = None
        for cand_norm, cid in candidates.items():
            score = fuzz.token_sort_ratio(norm, cand_norm)
            if score > best_score:
                best_score = score
                best_id = cid

        return best_score, best_id

    def check_conversation_similarity(
        self, new_conv: dict[str, Any]
    ) -> tuple[float, str | None]:
        """Return (similarity_score, matching_conv_id) using TF-IDF cosine."""
        if len(self._conv_tfidf_corpus) == 0:
            return 0.0, None

        self._rebuild_tfidf()
        if self._vectorizer is None or self._tfidf_matrix is None:
            return 0.0, None

        try:
            import numpy as np

            doc = _conversation_to_doc(new_conv)
            vec = self._vectorizer.transform([doc])
            sims = (self._tfidf_matrix @ vec.T).toarray().flatten()
            idx = int(np.argmax(sims))
            score = float(sims[idx]) * 100
            matched_id = self._conv_ids[idx] if score > 0 else None
            return score, matched_id
        except Exception as exc:
            logger.debug("Conversation similarity check failed: %s", exc)
            return 0.0, None

    def make_structure_signature(self, conv: dict[str, Any]) -> str:
        messages = conv.get("messages", [])
        traj = conv.get("interest_trajectory", "unknown")
        outcome = conv.get("final_outcome", "unknown")
        domain = conv.get("domain", "unknown")
        msg_count_bucket = (len(messages) // 4) * 4
        first_role = messages[0]["role"] if messages else "?"
        last_role = messages[-1]["role"] if messages else "?"
        return f"{domain}|{traj}|{outcome}|{msg_count_bucket}|{first_role}-{last_role}"

    def signature_count(self, sig: str) -> int:
        return self._structure_signature_counts[sig]

    def accept_conversation(self, conv: dict[str, Any]) -> None:
        """Register an accepted conversation in all indexes."""
        cid = conv["conversation_id"]
        for msg in conv.get("messages", []):
            self._index_message(msg.get("text", ""), cid)
        self._index_conversation(conv)
        sig = self.make_structure_signature(conv)
        self._structure_signature_counts[sig] += 1

    def most_repeated_phrases(self, n: int = 10) -> list[str]:
        return [phrase for phrase, _ in self._phrase_counts.most_common(n)]

    def check_new_conversation(
        self, conv: dict[str, Any], message_threshold: float | None = None, conversation_threshold: float | None = None
    ) -> list[tuple[str, str, float | None]]:
        """
        Check a new conversation for all duplicate types.
        Returns a list of (check_name, reason, score) tuples — empty means no issues.
        """
        msg_thr = message_threshold if message_threshold is not None else self.message_threshold
        conv_thr = conversation_threshold if conversation_threshold is not None else self.conversation_threshold

        issues: list[tuple[str, str, float | None]] = []
        messages = conv.get("messages", [])
        seen_in_this_conv: set[str] = set()

        for msg in messages:
            text = msg.get("text", "")
            norm = normalize_text(text)

            if norm and norm in VERY_SHORT_OK:
                continue

            if norm and norm in seen_in_this_conv:
                issues.append(("exact_duplicate", f"Repeated within conversation: {text[:60]}", None))
                continue
            if norm:
                seen_in_this_conv.add(norm)

            existing_id = self.check_exact_duplicate(text)
            if existing_id:
                issues.append(("exact_duplicate", f"Exact match in {existing_id}: {text[:60]}", 100.0))
                continue

            score, matched_id = self.check_fuzzy_duplicate(text)
            if score >= msg_thr:
                issues.append(
                    ("fuzzy_duplicate", f"Fuzzy match ({score:.0f}%) in {matched_id}: {text[:60]}", score)
                )

        if conv_thr < 101:
            conv_score, conv_matched = self.check_conversation_similarity(conv)
            if conv_score >= conv_thr:
                issues.append(
                    ("similar_conversation", f"TF-IDF similarity {conv_score:.1f}% with {conv_matched}", conv_score)
                )

        return issues


def _conversation_to_doc(conv: dict[str, Any]) -> str:
    parts = [
        conv.get("domain", ""),
        conv.get("interest_trajectory", ""),
        conv.get("final_outcome", ""),
    ]
    for msg in conv.get("messages", []):
        parts.append(msg.get("text", ""))
    return " ".join(parts)
