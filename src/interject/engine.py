"""Recommendation engine — score, rank, and learn from feedback."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import get_engine_config, get_seed_topics
from .db import InterjectDB
from .embeddings import (
    EMBEDDING_DIM,
    EmbeddingClient,
    bytes_to_vector,
    vector_to_bytes,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Scores discoveries against an interest profile, learns from feedback."""

    def __init__(
        self,
        db: InterjectDB,
        embedder: EmbeddingClient,
        config: dict[str, Any],
    ):
        self.db = db
        self.embedder = embedder
        self.config = config
        self._engine_cfg = get_engine_config(config)

        # Thresholds (adaptive)
        self.high_threshold = self._engine_cfg.get("high_threshold", 0.8)
        self.medium_threshold = self._engine_cfg.get("medium_threshold", 0.5)
        self.low_threshold = self._engine_cfg.get("low_threshold", 0.2)

        # Weights
        self._priority_weights = self._engine_cfg.get(
            "priority_weights", {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}
        )
        # Normalize string keys from YAML to ints
        self._priority_weights = {
            int(k): v for k, v in self._priority_weights.items()
        }
        self.recency_floor = self._engine_cfg.get("recency_floor", 0.8)
        self.gap_bonus = self._engine_cfg.get("gap_bonus", 0.3)

        # Profile vector (cached)
        self._profile_vector: np.ndarray | None = None

    def ensure_profile(self) -> None:
        """Initialize profile with seed topics if no vector exists."""
        profile = self.db.get_profile()
        if profile.get("topic_vector") is not None:
            self._profile_vector = bytes_to_vector(profile["topic_vector"])
            return

        seed_topics = get_seed_topics(self.config)
        if not seed_topics:
            self._profile_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            return

        # Embed seed topics and average
        vectors = self.embedder.embed_batch(seed_topics)
        self._profile_vector = vectors.mean(axis=0)
        # Normalize
        norm = np.linalg.norm(self._profile_vector)
        if norm > 0:
            self._profile_vector /= norm

        self.db.update_profile(topic_vector=vector_to_bytes(self._profile_vector))
        logger.info("Interest profile seeded from %d topics", len(seed_topics))

    def score(
        self,
        text: str,
        source: str,
        discovered_at: datetime | None = None,
        gap_detected: bool = False,
    ) -> tuple[float, str]:
        """Score a discovery text. Returns (score, tier).

        The score combines:
        - Cosine similarity to interest profile
        - Keyword weight boost
        - Source weight multiplier
        - Recency boost
        - Gap bonus
        """
        self.ensure_profile()

        # 1. Embed and compute cosine similarity
        vec = self.embedder.embed(text)
        if self._profile_vector is not None:
            base_score = self.embedder.cosine_similarity(vec, self._profile_vector)
        else:
            base_score = 0.0

        # Clamp to [0, 1]
        base_score = max(0.0, min(1.0, (base_score + 1.0) / 2.0))

        # 2. Source weight
        profile = self.db.get_profile()
        source_weights = profile.get("source_weights", {})
        source_mult = source_weights.get(source, 1.0)
        base_score *= source_mult

        # 3. Keyword weight
        keyword_weights = profile.get("keyword_weights", {})
        for kw, weight in keyword_weights.items():
            if kw.lower() in text.lower():
                base_score += weight * 0.1

        # 4. Recency boost
        if discovered_at:
            now = datetime.utcnow()
            days_old = max(0, (now - discovered_at).total_seconds() / 86400)
            # Linear decay from 1.0 to recency_floor over 7 days
            recency = max(self.recency_floor, 1.0 - (1.0 - self.recency_floor) * days_old / 7.0)
            base_score *= recency

        # 5. Gap bonus
        if gap_detected:
            base_score += self.gap_bonus

        # Clamp final score
        final_score = max(0.0, min(1.0, base_score))

        # Determine tier
        tier = self._tier(final_score)
        return final_score, tier

    def _tier(self, score: float) -> str:
        if score >= self.high_threshold:
            return "high"
        elif score >= self.medium_threshold:
            return "medium"
        elif score >= self.low_threshold:
            return "low"
        return "discard"

    def learn_promotion(self, discovery_id: str, bead_priority: int) -> None:
        """Update interest profile from a promotion signal."""
        discovery = self.db.get_discovery(discovery_id)
        if not discovery or not discovery.get("embedding"):
            return

        vec = bytes_to_vector(discovery["embedding"])
        weight = self._priority_weights.get(bead_priority, 1.0)

        self.ensure_profile()
        if self._profile_vector is None:
            self._profile_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Weighted moving average
        self._profile_vector = (
            self._profile_vector * 0.9 + vec * 0.1 * weight
        )
        norm = np.linalg.norm(self._profile_vector)
        if norm > 0:
            self._profile_vector /= norm

        # Update source weight
        source = discovery["source"]
        profile = self.db.get_profile()
        source_weights = profile.get("source_weights", {})
        current = source_weights.get(source, 1.0)
        source_weights[source] = min(2.0, current + 0.05 * weight)

        self.db.update_profile(
            topic_vector=vector_to_bytes(self._profile_vector),
            source_weights=source_weights,
        )
        logger.info(
            "Learned from promotion: discovery=%s priority=P%d weight=%.1f",
            discovery_id, bead_priority, weight,
        )

    def learn_dismissal(self, discovery_id: str) -> None:
        """Update interest profile from a dismissal (negative signal)."""
        discovery = self.db.get_discovery(discovery_id)
        if not discovery or not discovery.get("embedding"):
            return

        vec = bytes_to_vector(discovery["embedding"])

        self.ensure_profile()
        if self._profile_vector is None:
            return

        # Move profile away from dismissed content
        self._profile_vector = self._profile_vector * 0.95 - vec * 0.05
        norm = np.linalg.norm(self._profile_vector)
        if norm > 0:
            self._profile_vector /= norm

        # Slightly reduce source weight
        source = discovery["source"]
        profile = self.db.get_profile()
        source_weights = profile.get("source_weights", {})
        current = source_weights.get(source, 1.0)
        source_weights[source] = max(0.3, current - 0.02)

        self.db.update_profile(
            topic_vector=vector_to_bytes(self._profile_vector),
            source_weights=source_weights,
        )
        logger.info("Learned from dismissal: discovery=%s", discovery_id)

    def adapt_thresholds(self) -> None:
        """Adjust confidence tier thresholds based on promotion/dismissal rates."""
        promotions = self.db.get_promotions(limit=100)
        if len(promotions) < 10:
            return  # Not enough data

        stats = self.db.get_stats()
        total = stats.get("total_discoveries", 0)
        promoted = stats.get("total_promotions", 0)
        if total == 0:
            return

        promotion_rate = promoted / total

        # If promoting a lot → lower thresholds to surface more
        # If ignoring a lot → raise thresholds to reduce noise
        if promotion_rate > 0.3:
            self.high_threshold = max(0.6, self.high_threshold - 0.02)
            self.medium_threshold = max(0.3, self.medium_threshold - 0.02)
        elif promotion_rate < 0.1:
            self.high_threshold = min(0.95, self.high_threshold + 0.02)
            self.medium_threshold = min(0.7, self.medium_threshold + 0.02)

        logger.info(
            "Adapted thresholds: high=%.2f medium=%.2f (promotion_rate=%.2f)",
            self.high_threshold, self.medium_threshold, promotion_rate,
        )

    def update_source_weights_from_feedback(self) -> None:
        """Adjust source weights based on conversion rates from feedback signals."""
        from .feedback import FeedbackCollector

        collector = FeedbackCollector(self.db)
        rates = collector.get_source_conversion_rates()

        profile = self.db.get_profile()
        source_weights = profile.get("source_weights", {})

        for source, data in rates.items():
            promoted = data.get("promoted", 0)
            shipped = data.get("shipped", 0)
            if promoted >= 3:  # Need enough data
                conversion = shipped / promoted if promoted > 0 else 0
                current = source_weights.get(source, 1.0)
                adjustment = 0.1 * (conversion - 0.3)  # 0.3 is baseline
                source_weights[source] = max(0.3, min(2.0, current + adjustment))

        self.db.update_profile(source_weights=source_weights)

    def get_embedding(self, text: str) -> bytes:
        """Get embedding bytes for a text string."""
        vec = self.embedder.embed(text)
        return vector_to_bytes(vec)
