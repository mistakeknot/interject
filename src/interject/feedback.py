"""Feedback collection and analysis utilities."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from interject.db import InterjectDB


class FeedbackCollector:
    """Collects feedback signals and query patterns for Interject."""

    def __init__(self, db: InterjectDB):
        self.db = db

    def record_bead_outcome(self, bead_id: str, outcome: str) -> None:
        """Record an outcome for the discovery linked to a promoted bead."""
        row = self.db.conn.execute(
            """SELECT discovery_id FROM promotions
               WHERE bead_id = ?
               ORDER BY promoted_at DESC, id DESC
               LIMIT 1""",
            (bead_id,),
        ).fetchone()
        if row is None:
            return

        signal_type = f"bead_{outcome.strip().lower()}"
        signal_data = json.dumps({"bead_id": bead_id, "outcome": outcome})
        self.db.insert_feedback_signal(
            row["discovery_id"],
            signal_type,
            signal_data,
        )

    def record_query(self, query_text: str, session_id: str = "") -> None:
        """Persist a query event for cross-session pattern tracking."""
        normalized_session = session_id.strip() or None
        self.db.insert_query_log(query_text, session_id=normalized_session)

    def get_repeated_queries(self, min_sessions: int = 2) -> list[dict]:
        """Return normalized queries seen in at least min_sessions distinct sessions."""
        rows = self.db.conn.execute(
            "SELECT query_text, session_id FROM query_log"
        ).fetchall()

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            query_text = row["query_text"]
            normalized = self._normalize_query(query_text)
            if not normalized:
                continue

            entry = grouped.setdefault(
                normalized,
                {
                    "query_text": query_text,
                    "count": 0,
                    "sessions": set(),
                },
            )
            entry["count"] += 1
            session_id = (row["session_id"] or "").strip()
            if session_id:
                entry["sessions"].add(session_id)

        results: list[dict] = []
        for normalized, entry in grouped.items():
            session_count = len(entry["sessions"])
            if session_count >= min_sessions:
                results.append(
                    {
                        "query_text": entry["query_text"],
                        "normalized_query": normalized,
                        "count": entry["count"],
                        "session_count": session_count,
                    }
                )

        results.sort(key=lambda item: (-item["session_count"], -item["count"], item["normalized_query"]))
        return results

    def get_source_conversion_rates(self) -> dict[str, dict[str, int]]:
        """Return conversion totals by source: total, promoted, and shipped."""
        totals = {
            row["source"]: row["cnt"]
            for row in self.db.conn.execute(
                "SELECT source, COUNT(*) AS cnt FROM discoveries GROUP BY source"
            ).fetchall()
        }

        promoted = {
            row["source"]: row["cnt"]
            for row in self.db.conn.execute(
                """SELECT d.source, COUNT(DISTINCT d.id) AS cnt
                   FROM discoveries d
                   JOIN promotions p ON p.discovery_id = d.id
                   GROUP BY d.source"""
            ).fetchall()
        }

        shipped = {
            row["source"]: row["cnt"]
            for row in self.db.conn.execute(
                """SELECT d.source, COUNT(DISTINCT d.id) AS cnt
                   FROM discoveries d
                   JOIN feedback_signals f ON f.discovery_id = d.id
                   WHERE f.signal_type = 'bead_shipped'
                   GROUP BY d.source"""
            ).fetchall()
        }

        sources = set(totals) | set(promoted) | set(shipped)
        return {
            source: {
                "total": int(totals.get(source, 0)),
                "promoted": int(promoted.get(source, 0)),
                "shipped": int(shipped.get(source, 0)),
            }
            for source in sources
        }

    def scan_bead_updates(self) -> int:
        """Scan promoted beads and record shipped outcomes for closed/done beads."""
        promoted_rows = self.db.conn.execute(
            "SELECT DISTINCT discovery_id, bead_id FROM promotions"
        ).fetchall()

        new_signals = 0
        for row in promoted_rows:
            discovery_id = row["discovery_id"]
            bead_id = row["bead_id"]

            existing = self.db.conn.execute(
                """SELECT 1 FROM feedback_signals
                   WHERE discovery_id = ? AND signal_type = 'bead_shipped'
                   LIMIT 1""",
                (discovery_id,),
            ).fetchone()
            if existing:
                continue

            try:
                result = subprocess.run(
                    ["bd", "show", bead_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

            if result.returncode != 0:
                continue

            output = f"{result.stdout}\n{result.stderr}".lower()
            if self._is_closed_or_done(output):
                signal_data = json.dumps({"bead_id": bead_id, "scan": "bd_show"})
                self.db.insert_feedback_signal(
                    discovery_id,
                    "bead_shipped",
                    signal_data,
                )
                new_signals += 1

        return new_signals

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        return re.sub(r"\s+", " ", query_text.strip().lower())

    @staticmethod
    def _is_closed_or_done(text: str) -> bool:
        return bool(re.search(r"\b(status|state)\s*:\s*(closed|done)\b", text))
