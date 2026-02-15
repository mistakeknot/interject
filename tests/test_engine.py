"""Tests for the recommendation engine."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from interject.config import load_config
from interject.db import InterjectDB
from interject.embeddings import EmbeddingClient, vector_to_bytes
from interject.embeddings import EMBEDDING_DIM
from interject.engine import RecommendationEngine


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = InterjectDB(Path(tmpdir) / "test.db")
        db.connect()
        yield db
        db.close()


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic vectors."""
    embedder = MagicMock(spec=EmbeddingClient)

    def fake_embed(text):
        # Deterministic hash-based vector
        np.random.seed(hash(text) % 2**32)
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def fake_embed_batch(texts):
        return np.array([fake_embed(t) for t in texts])

    def fake_cosine(a, b):
        return float(np.dot(a, b))

    embedder.embed.side_effect = fake_embed
    embedder.embed_batch.side_effect = fake_embed_batch
    embedder.cosine_similarity.side_effect = fake_cosine
    return embedder


@pytest.fixture
def config():
    return {
        "engine": {
            "high_threshold": 0.8,
            "medium_threshold": 0.5,
            "low_threshold": 0.2,
            "decay_rate": 0.95,
            "decay_floor": 0.1,
            "priority_weights": {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0},
            "recency_floor": 0.8,
            "gap_bonus": 0.3,
        },
        "seed_topics": ["MCP servers", "agent coordination", "code analysis"],
    }


@pytest.fixture
def engine(db, mock_embedder, config):
    return RecommendationEngine(db, mock_embedder, config)


class TestScoring:
    def test_score_returns_tuple(self, engine):
        score, tier = engine.score("test text", source="test")
        assert isinstance(score, float)
        assert tier in ("high", "medium", "low", "discard")

    def test_score_range(self, engine):
        score, _ = engine.score("MCP server for code analysis", source="test")
        assert 0.0 <= score <= 1.0

    def test_gap_bonus(self, engine):
        score_no_gap, _ = engine.score("some text", source="test", gap_detected=False)
        score_gap, _ = engine.score("some text", source="test", gap_detected=True)
        assert score_gap > score_no_gap


class TestTiering:
    def test_high_tier(self, engine):
        assert engine._tier(0.9) == "high"

    def test_medium_tier(self, engine):
        assert engine._tier(0.6) == "medium"

    def test_low_tier(self, engine):
        assert engine._tier(0.3) == "low"

    def test_discard_tier(self, engine):
        assert engine._tier(0.1) == "discard"


class TestLearning:
    def test_learn_promotion_updates_profile(self, engine, db):
        # Insert a discovery with embedding
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="t1", title="Test",
            embedding=vector_to_bytes(vec),
        )

        engine.ensure_profile()
        old_profile = db.get_profile()
        engine.learn_promotion("ij-test-1", bead_priority=0)
        new_profile = db.get_profile()

        # Source weight should have increased
        old_weight = old_profile.get("source_weights", {}).get("test", 1.0)
        new_weight = new_profile.get("source_weights", {}).get("test", 1.0)
        assert new_weight > old_weight

    def test_learn_dismissal_updates_profile(self, engine, db):
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="t1", title="Test",
            embedding=vector_to_bytes(vec),
        )

        engine.ensure_profile()
        old_profile = db.get_profile()
        engine.learn_dismissal("ij-test-1")
        new_profile = db.get_profile()

        # Source weight should have decreased
        old_weight = old_profile.get("source_weights", {}).get("test", 1.0)
        new_weight = new_profile.get("source_weights", {}).get("test", 1.0)
        assert new_weight < old_weight


class TestAdaptiveThresholds:
    def test_no_adaptation_with_few_promotions(self, engine, db):
        old_high = engine.high_threshold
        engine.adapt_thresholds()
        assert engine.high_threshold == old_high  # Not enough data

    def test_adaptation_with_many_promotions(self, engine, db):
        # Insert many discoveries and promote most of them
        for i in range(20):
            db.insert_discovery(
                id=f"ij-test-{i}", source="test", source_id=f"t{i}", title=f"Test {i}",
                relevance_score=0.6,
            )
        for i in range(15):
            db.record_promotion(f"ij-test-{i}", f"iv-{i}", 2)

        old_high = engine.high_threshold
        engine.adapt_thresholds()
        # High promotion rate should lower threshold
        assert engine.high_threshold <= old_high


class TestEnsureProfile:
    def test_seeds_from_config(self, engine, db):
        engine.ensure_profile()
        profile = db.get_profile()
        assert profile.get("topic_vector") is not None


class TestFeedbackScoring:
    def test_source_weight_from_conversion(self, engine, db):
        """Sources with higher shipped conversion should score higher."""
        # Insert discoveries and promote them
        for i in range(5):
            db.insert_discovery(
                id=f"ij-gh-{i}", source="github", source_id=f"gh{i}",
                title=f"GH Tool {i}", relevance_score=0.7,
            )
            db.record_promotion(f"ij-gh-{i}", f"iv-{i}", 2)

        # Ship 3 of the 5
        for i in range(3):
            db.insert_feedback_signal(
                discovery_id=f"ij-gh-{i}",
                signal_type="bead_shipped",
                signal_data="{}",
            )

        engine.update_source_weights_from_feedback()
        profile = db.get_profile()
        gh_weight = profile.get("source_weights", {}).get("github", 1.0)
        # 3/5 = 60% conversion, above 30% baseline, so weight should increase
        assert gh_weight > 1.0
