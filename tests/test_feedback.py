"""Tests for feedback collection and schema migration."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import pytest

from interject.db import InterjectDB
from interject.feedback import FeedbackCollector


@pytest.fixture
def db(tmp_path):
    instance = InterjectDB(tmp_path / "feedback.db")
    instance.connect()
    yield instance
    instance.close()


def test_collect_bead_lifecycle(db):
    db.insert_discovery(
        id="ij-a-1",
        source="arxiv",
        source_id="a1",
        title="A1",
    )
    db.record_promotion("ij-a-1", "iv-a-1", 2)

    collector = FeedbackCollector(db)
    collector.record_bead_outcome("iv-a-1", "shipped")

    signals = db.get_feedback_signals("ij-a-1")
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "bead_shipped"
    payload = json.loads(signals[0]["signal_data"])
    assert payload["bead_id"] == "iv-a-1"


def test_collect_query_pattern(db):
    collector = FeedbackCollector(db)
    collector.record_query("Find MCP servers", session_id="session-1")
    collector.record_query("  find   mcp   servers ", session_id="session-2")

    repeated = collector.get_repeated_queries(min_sessions=2)
    assert len(repeated) == 1
    assert repeated[0]["normalized_query"] == "find mcp servers"
    assert repeated[0]["session_count"] == 2


def test_source_conversion_rate(db):
    db.insert_discovery(
        id="ij-arxiv-1",
        source="arxiv",
        source_id="a1",
        title="A1",
    )
    db.insert_discovery(
        id="ij-arxiv-2",
        source="arxiv",
        source_id="a2",
        title="A2",
    )
    db.insert_discovery(
        id="ij-github-1",
        source="github",
        source_id="g1",
        title="G1",
    )

    db.record_promotion("ij-arxiv-1", "iv-arxiv-1", 2)
    db.record_promotion("ij-github-1", "iv-github-1", 3)

    collector = FeedbackCollector(db)
    collector.record_bead_outcome("iv-arxiv-1", "shipped")

    rates = collector.get_source_conversion_rates()
    assert rates["arxiv"] == {"total": 2, "promoted": 1, "shipped": 1}
    assert rates["github"] == {"total": 1, "promoted": 1, "shipped": 0}


def test_schema_migration(tmp_path):
    db_path = tmp_path / "migration.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE discoveries (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            raw_metadata TEXT NOT NULL DEFAULT '{}',
            embedding BLOB,
            relevance_score REAL NOT NULL DEFAULT 0.0,
            confidence_tier TEXT NOT NULL DEFAULT 'low',
            status TEXT NOT NULL DEFAULT 'new',
            discovered_at TEXT NOT NULL DEFAULT (datetime('now')),
            reviewed_at TEXT,
            UNIQUE(source, source_id)
        );

        CREATE TABLE promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovery_id TEXT NOT NULL REFERENCES discoveries(id),
            bead_id TEXT NOT NULL,
            bead_priority INTEGER NOT NULL,
            promoted_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE interest_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            topic_vector BLOB,
            keyword_weights TEXT NOT NULL DEFAULT '{}',
            source_weights TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            scanned_at TEXT NOT NULL DEFAULT (datetime('now')),
            items_found INTEGER NOT NULL DEFAULT 0,
            items_above_threshold INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE schema_info (
            version INTEGER NOT NULL
        );

        INSERT INTO schema_info (version) VALUES (1);
        """
    )
    conn.commit()
    conn.close()

    migrated = InterjectDB(db_path)
    migrated.connect()
    try:
        version = migrated.conn.execute(
            "SELECT version FROM schema_info LIMIT 1"
        ).fetchone()["version"]
        assert version == 2

        feedback_table = migrated.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_signals'"
        ).fetchone()
        query_log_table = migrated.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'"
        ).fetchone()
        assert feedback_table is not None
        assert query_log_table is not None
    finally:
        migrated.close()


def test_scan_bead_updates_records_shipped_once(db):
    db.insert_discovery(
        id="ij-a-1",
        source="arxiv",
        source_id="a1",
        title="A1",
    )
    db.record_promotion("ij-a-1", "iv-a-1", 2)

    collector = FeedbackCollector(db)
    done = subprocess.CompletedProcess(
        args=["bd", "show", "iv-a-1"],
        returncode=0,
        stdout="Status: done\n",
        stderr="",
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("interject.feedback.subprocess.run", lambda *args, **kwargs: done)
        inserted = collector.scan_bead_updates()
    assert inserted == 1

    signals = db.get_feedback_signals("ij-a-1")
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "bead_shipped"

    with pytest.MonkeyPatch.context() as monkeypatch:
        call_count = {"n": 0}

        def _run(*args, **kwargs):
            call_count["n"] += 1
            return done

        monkeypatch.setattr("interject.feedback.subprocess.run", _run)
        inserted = collector.scan_bead_updates()

    assert inserted == 0
    assert call_count["n"] == 0
