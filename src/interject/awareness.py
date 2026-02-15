"""Internal awareness helpers for project context signals."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


class InternalAwareness:
    """Reads internal project state to provide context for discovery scoring."""

    def __init__(self, interverse_root: Path | None = None):
        self.root = interverse_root or Path("/root/projects/Interverse")

    def get_active_topics(self) -> list[str]:
        """Combine active topics from plans and brainstorms."""
        return self._topics_from_plans() + self._topics_from_brainstorms()

    def get_bead_topics(self) -> list[str]:
        """Return titles for open beads, or [] when bead data is unavailable."""
        return self._topics_from_beads(status="open")

    def get_suppression_topics(self) -> list[str]:
        """Return titles for closed beads, or [] when bead data is unavailable."""
        return self._topics_from_beads(status="closed")

    def _topics_from_plans(self) -> list[str]:
        """Extract plan titles and explicit goal lines from recent plan docs."""
        topics: list[str] = []
        goal_re = re.compile(r"^\*\*Goal:\*\*\s*(.+?)\s*$", re.IGNORECASE)

        for path in self._recent_markdown_files(self.root / "docs" / "plans"):
            try:
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("# "):
                        title = stripped[2:].strip()
                        if title:
                            topics.append(title)
                    match = goal_re.match(stripped)
                    if match:
                        goal = match.group(1).strip()
                        if goal:
                            topics.append(goal)
            except OSError:
                continue

        return topics

    def _topics_from_brainstorms(self) -> list[str]:
        """Extract brainstorm titles from recent brainstorm docs."""
        topics: list[str] = []

        for path in self._recent_markdown_files(self.root / "docs" / "brainstorms"):
            try:
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("# "):
                        title = stripped[2:].strip()
                        if title:
                            topics.append(title)
            except OSError:
                continue

        return topics

    def _recent_markdown_files(self, directory: Path, limit: int = 10) -> list[Path]:
        """Return up to ``limit`` markdown files sorted by mtime descending."""
        if not directory.exists() or not directory.is_dir():
            return []

        try:
            files = [p for p in directory.glob("*.md") if p.is_file()]
        except OSError:
            return []

        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    def _topics_from_beads(self, status: str) -> list[str]:
        """Read bead titles for a given status via ``bd`` CLI."""
        try:
            result = subprocess.run(
                ["bd", "list", f"--status={status}", "--format=json"],
                capture_output=True,
                text=True,
                cwd=str(self.root),
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []

        if result.returncode != 0:
            return []

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        records: list[dict] = []
        if isinstance(payload, list):
            records = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            if isinstance(payload.get("issues"), list):
                records = [
                    item for item in payload["issues"] if isinstance(item, dict)
                ]
            elif isinstance(payload.get("items"), list):
                records = [
                    item for item in payload["items"] if isinstance(item, dict)
                ]

        topics: list[str] = []
        for record in records:
            title = record.get("title")
            if isinstance(title, str) and title.strip():
                topics.append(title.strip())
        return topics

