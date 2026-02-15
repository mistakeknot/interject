"""Tests for the Interject database module."""

import tempfile
from pathlib import Path

import pytest

from interject.db import InterjectDB


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = InterjectDB(Path(tmpdir) / "test.db")
        db.connect()
        yield db
        db.close()


class TestDiscoveries:
    def test_insert_and_get(self, db):
        inserted = db.insert_discovery(
            id="ij-test-1",
            source="test",
            source_id="test-1",
            title="Test Discovery",
            summary="A test summary",
            url="https://example.com",
        )
        assert inserted is True

        disc = db.get_discovery("ij-test-1")
        assert disc is not None
        assert disc["title"] == "Test Discovery"
        assert disc["source"] == "test"
        assert disc["status"] == "new"

    def test_insert_duplicate(self, db):
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="test-1", title="First"
        )
        inserted = db.insert_discovery(
            id="ij-test-2", source="test", source_id="test-1", title="Duplicate"
        )
        assert inserted is False

    def test_list_with_filters(self, db):
        db.insert_discovery(
            id="ij-a-1", source="arxiv", source_id="a1", title="A1",
            relevance_score=0.9,
        )
        db.insert_discovery(
            id="ij-b-1", source="github", source_id="b1", title="B1",
            relevance_score=0.3,
        )

        # Filter by min score
        results = db.list_discoveries(min_score=0.5)
        assert len(results) == 1
        assert results[0]["id"] == "ij-a-1"

        # Filter by source
        results = db.list_discoveries(source="github")
        assert len(results) == 1
        assert results[0]["source"] == "github"

    def test_update_status(self, db):
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="t1", title="Test"
        )
        db.update_discovery_status("ij-test-1", "promoted")
        disc = db.get_discovery("ij-test-1")
        assert disc["status"] == "promoted"

    def test_decay(self, db):
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="t1", title="Test",
            relevance_score=0.5,
        )
        count = db.apply_decay(rate=0.5, floor=0.1)
        assert count == 1
        disc = db.get_discovery("ij-test-1")
        assert disc["relevance_score"] == pytest.approx(0.25, abs=0.01)

    def test_decay_below_floor(self, db):
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="t1", title="Test",
            relevance_score=0.15,
        )
        db.apply_decay(rate=0.5, floor=0.1)
        disc = db.get_discovery("ij-test-1")
        assert disc["status"] == "decayed"


class TestPromotions:
    def test_record_and_get(self, db):
        db.insert_discovery(
            id="ij-test-1", source="test", source_id="t1", title="Test"
        )
        db.record_promotion("ij-test-1", "iv-abc", 2)

        promos = db.get_promotions()
        assert len(promos) == 1
        assert promos[0]["bead_id"] == "iv-abc"
        assert promos[0]["bead_priority"] == 2

        disc = db.get_discovery("ij-test-1")
        assert disc["status"] == "promoted"


class TestProfile:
    def test_default_profile(self, db):
        profile = db.get_profile()
        assert profile["keyword_weights"] == {}
        assert profile["source_weights"] == {}

    def test_update_profile(self, db):
        db.update_profile(
            keyword_weights={"MCP": 1.5, "agent": 1.0},
            source_weights={"arxiv": 1.2},
        )
        profile = db.get_profile()
        assert profile["keyword_weights"]["MCP"] == 1.5
        assert profile["source_weights"]["arxiv"] == 1.2


class TestScanLog:
    def test_log_and_get(self, db):
        db.log_scan("arxiv", 50, 5)
        last = db.get_last_scan("arxiv")
        assert last is not None
        assert last["items_found"] == 50
        assert last["items_above_threshold"] == 5

    def test_stats(self, db):
        db.log_scan("arxiv", 50, 5)
        db.log_scan("github", 30, 3)
        stats = db.get_scan_stats()
        assert len(stats) == 2


class TestStats:
    def test_overall_stats(self, db):
        db.insert_discovery(
            id="ij-1", source="arxiv", source_id="a1", title="A"
        )
        db.insert_discovery(
            id="ij-2", source="github", source_id="g1", title="G"
        )
        db.record_promotion("ij-1", "iv-x", 1)

        stats = db.get_stats()
        assert stats["total_discoveries"] == 2
        assert stats["by_source"]["arxiv"] == 1
        assert stats["total_promotions"] == 1
