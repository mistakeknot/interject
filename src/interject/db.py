"""SQLite database schema and queries for Interject."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS discoveries (
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

CREATE TABLE IF NOT EXISTS promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_id TEXT NOT NULL REFERENCES discoveries(id),
    bead_id TEXT NOT NULL,
    bead_priority INTEGER NOT NULL,
    promoted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS interest_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    topic_vector BLOB,
    keyword_weights TEXT NOT NULL DEFAULT '{}',
    source_weights TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scan_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    scanned_at TEXT NOT NULL DEFAULT (datetime('now')),
    items_found INTEGER NOT NULL DEFAULT 0,
    items_above_threshold INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_discoveries_source ON discoveries(source);
CREATE INDEX IF NOT EXISTS idx_discoveries_status ON discoveries(status);
CREATE INDEX IF NOT EXISTS idx_discoveries_score ON discoveries(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_discoveries_tier ON discoveries(confidence_tier);
CREATE INDEX IF NOT EXISTS idx_discoveries_discovered ON discoveries(discovered_at DESC);
CREATE INDEX IF NOT EXISTS idx_scan_log_source ON scan_log(source, scanned_at DESC);
"""


class InterjectDB:
    """SQLite database for Interject discovery state."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    def _init_schema(self) -> None:
        cur = self.conn.executescript(SCHEMA_SQL)
        # Check/set schema version
        row = self.conn.execute(
            "SELECT version FROM schema_info LIMIT 1"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,)
            )
            self.conn.commit()
        # Seed empty interest profile if missing
        row = self.conn.execute(
            "SELECT id FROM interest_profile WHERE id = 1"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO interest_profile (id, keyword_weights, source_weights) "
                "VALUES (1, '{}', '{}')"
            )
            self.conn.commit()

    # ── Discovery CRUD ──────────────────────────────────────────────

    def insert_discovery(
        self,
        *,
        id: str,
        source: str,
        source_id: str,
        title: str,
        summary: str = "",
        url: str = "",
        raw_metadata: dict | None = None,
        embedding: bytes | None = None,
        relevance_score: float = 0.0,
        confidence_tier: str = "low",
    ) -> bool:
        """Insert a discovery. Returns True if inserted, False if duplicate."""
        try:
            self.conn.execute(
                """INSERT INTO discoveries
                   (id, source, source_id, title, summary, url, raw_metadata,
                    embedding, relevance_score, confidence_tier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    id,
                    source,
                    source_id,
                    title,
                    summary,
                    url,
                    json.dumps(raw_metadata or {}),
                    embedding,
                    relevance_score,
                    confidence_tier,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_discovery(self, discovery_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM discoveries WHERE id = ?", (discovery_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_discoveries(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        min_score: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        query = "SELECT * FROM discoveries WHERE relevance_score >= ?"
        params: list[Any] = [min_score]
        if status:
            query += " AND status = ?"
            params.append(status)
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY relevance_score DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def update_discovery_score(
        self, discovery_id: str, score: float, tier: str
    ) -> None:
        self.conn.execute(
            "UPDATE discoveries SET relevance_score = ?, confidence_tier = ? WHERE id = ?",
            (score, tier, discovery_id),
        )
        self.conn.commit()

    def update_discovery_status(self, discovery_id: str, status: str) -> None:
        reviewed = datetime.now(timezone.utc).isoformat() if status == "reviewed" else None
        if reviewed:
            self.conn.execute(
                "UPDATE discoveries SET status = ?, reviewed_at = ? WHERE id = ?",
                (status, reviewed, discovery_id),
            )
        else:
            self.conn.execute(
                "UPDATE discoveries SET status = ? WHERE id = ?",
                (status, discovery_id),
            )
        self.conn.commit()

    def update_discovery_embedding(self, discovery_id: str, embedding: bytes) -> None:
        self.conn.execute(
            "UPDATE discoveries SET embedding = ? WHERE id = ?",
            (embedding, discovery_id),
        )
        self.conn.commit()

    def get_new_discoveries(self, min_score: float = 0.0, limit: int = 10) -> list[dict]:
        """Get discoveries with status 'new' above threshold."""
        rows = self.conn.execute(
            """SELECT * FROM discoveries
               WHERE status = 'new' AND relevance_score >= ?
               ORDER BY relevance_score DESC LIMIT ?""",
            (min_score, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def apply_decay(self, rate: float = 0.95, floor: float = 0.1) -> int:
        """Apply decay to unreviewed discoveries. Returns count affected."""
        cur = self.conn.execute(
            """UPDATE discoveries
               SET relevance_score = relevance_score * ?,
                   status = CASE WHEN relevance_score * ? < ? THEN 'decayed' ELSE status END
               WHERE status IN ('new', 'low')
               AND relevance_score > ?""",
            (rate, rate, floor, floor),
        )
        self.conn.commit()
        return cur.rowcount

    # ── Promotions ──────────────────────────────────────────────────

    def record_promotion(
        self, discovery_id: str, bead_id: str, bead_priority: int
    ) -> None:
        self.conn.execute(
            "INSERT INTO promotions (discovery_id, bead_id, bead_priority) VALUES (?, ?, ?)",
            (discovery_id, bead_id, bead_priority),
        )
        self.conn.execute(
            "UPDATE discoveries SET status = 'promoted' WHERE id = ?",
            (discovery_id,),
        )
        self.conn.commit()

    def get_promotions(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM promotions ORDER BY promoted_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Interest Profile ────────────────────────────────────────────

    def get_profile(self) -> dict:
        row = self.conn.execute(
            "SELECT * FROM interest_profile WHERE id = 1"
        ).fetchone()
        if row is None:
            return {"topic_vector": None, "keyword_weights": {}, "source_weights": {}}
        result = dict(row)
        result["keyword_weights"] = json.loads(result["keyword_weights"])
        result["source_weights"] = json.loads(result["source_weights"])
        return result

    def update_profile(
        self,
        *,
        topic_vector: bytes | None = None,
        keyword_weights: dict | None = None,
        source_weights: dict | None = None,
    ) -> None:
        updates = []
        params: list[Any] = []
        if topic_vector is not None:
            updates.append("topic_vector = ?")
            params.append(topic_vector)
        if keyword_weights is not None:
            updates.append("keyword_weights = ?")
            params.append(json.dumps(keyword_weights))
        if source_weights is not None:
            updates.append("source_weights = ?")
            params.append(json.dumps(source_weights))
        if not updates:
            return
        updates.append("updated_at = datetime('now')")
        self.conn.execute(
            f"UPDATE interest_profile SET {', '.join(updates)} WHERE id = 1",
            params,
        )
        self.conn.commit()

    # ── Scan Log ────────────────────────────────────────────────────

    def log_scan(self, source: str, items_found: int, items_above: int) -> None:
        self.conn.execute(
            "INSERT INTO scan_log (source, items_found, items_above_threshold) VALUES (?, ?, ?)",
            (source, items_found, items_above),
        )
        self.conn.commit()

    def get_last_scan(self, source: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM scan_log WHERE source = ? ORDER BY scanned_at DESC LIMIT 1",
            (source,),
        ).fetchone()
        return dict(row) if row else None

    def get_last_scan_time(self, source: str | None = None) -> datetime | None:
        """Get the most recent scan timestamp, optionally filtered by source."""
        if source:
            row = self.conn.execute(
                "SELECT MAX(scanned_at) as t FROM scan_log WHERE source = ?",
                (source,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT MAX(scanned_at) as t FROM scan_log"
            ).fetchone()
        if row and row["t"]:
            return datetime.fromisoformat(row["t"])
        return None

    def get_scan_stats(self) -> list[dict]:
        """Get last scan time and counts per source."""
        rows = self.conn.execute(
            """SELECT source,
                      MAX(scanned_at) as last_scan,
                      SUM(items_found) as total_found,
                      SUM(items_above_threshold) as total_above
               FROM scan_log GROUP BY source"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get overall discovery stats."""
        total = self.conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
        by_status = {}
        for row in self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM discoveries GROUP BY status"
        ).fetchall():
            by_status[row["status"]] = row["cnt"]
        by_source = {}
        for row in self.conn.execute(
            "SELECT source, COUNT(*) as cnt FROM discoveries GROUP BY source"
        ).fetchall():
            by_source[row["source"]] = row["cnt"]
        promotions = self.conn.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
        return {
            "total_discoveries": total,
            "by_status": by_status,
            "by_source": by_source,
            "total_promotions": promotions,
        }
