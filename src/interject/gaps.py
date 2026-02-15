"""Capability gap detection — scans installed plugins and beads to find gaps."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Taxonomy of common engineering capabilities
# Each entry: (category, description, indicator_keywords)
CAPABILITY_TAXONOMY = [
    ("testing", "Automated testing and test generation", ["test", "testing", "coverage", "assertion", "mock"]),
    ("ci-cd", "Continuous integration and deployment", ["ci", "cd", "pipeline", "deploy", "github-actions"]),
    ("monitoring", "Observability, metrics, and alerting", ["monitor", "metrics", "alert", "observability", "tracing"]),
    ("documentation", "Doc generation and maintenance", ["docs", "documentation", "readme", "api-docs"]),
    ("coordination", "Multi-agent coordination and orchestration", ["coordinate", "orchestrate", "agent", "dispatch"]),
    ("code-analysis", "Static analysis, linting, formatting", ["lint", "format", "static-analysis", "ast"]),
    ("security", "Security scanning and vulnerability detection", ["security", "vulnerability", "cve", "audit"]),
    ("performance", "Performance profiling and optimization", ["performance", "profiling", "benchmark", "optimization"]),
    ("data-pipeline", "Data processing and ETL", ["data", "etl", "pipeline", "transform"]),
    ("search", "Code and content search", ["search", "index", "query", "find"]),
    ("version-control", "Git workflows and branching strategies", ["git", "branch", "merge", "rebase"]),
    ("dependency-management", "Package and dependency management", ["dependency", "package", "upgrade", "audit"]),
    ("visualization", "Charts, diagrams, and visual output", ["chart", "diagram", "visualization", "render"]),
    ("notification", "Alerts, messages, and notifications", ["notification", "alert", "message", "slack"]),
    ("database", "Database management and migration", ["database", "migration", "schema", "sql"]),
    ("api-client", "API integration and client generation", ["api", "client", "rest", "graphql"]),
    ("embedding", "Text and code embedding models", ["embedding", "vector", "semantic", "similarity"]),
    ("workflow-automation", "Automated workflows and task chains", ["workflow", "automation", "chain", "trigger"]),
]


class GapDetector:
    """Detects capability gaps by comparing installed plugins against a taxonomy."""

    def __init__(self, interverse_root: Path | None = None):
        self.interverse_root = interverse_root or Path("/root/projects/Interverse")
        self._capabilities: dict[str, list[str]] = {}  # category -> [plugin names that cover it]

    def scan(self) -> list[dict[str, Any]]:
        """Scan plugins and return detected gaps.

        Returns list of {category, description, search_queries}
        """
        self._scan_plugins()
        gaps = []
        for category, description, keywords in CAPABILITY_TAXONOMY:
            if category not in self._capabilities or not self._capabilities[category]:
                gaps.append({
                    "category": category,
                    "description": description,
                    "search_queries": self._generate_queries(category, keywords),
                })
        return gaps

    def _scan_plugins(self) -> None:
        """Build capability map from installed plugins."""
        plugins_dir = self.interverse_root / "plugins"
        if not plugins_dir.exists():
            return

        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            plugin_name = plugin_dir.name
            # Read plugin.json if it exists
            plugin_json = plugin_dir / ".claude-plugin" / "plugin.json"
            if plugin_json.exists():
                try:
                    data = json.loads(plugin_json.read_text())
                    description = data.get("description", "").lower()
                    keywords = [k.lower() for k in data.get("keywords", [])]
                    text = f"{description} {' '.join(keywords)} {plugin_name}"
                    self._classify_plugin(plugin_name, text)
                except (json.JSONDecodeError, OSError):
                    pass

            # Also check CLAUDE.md
            claude_md = plugin_dir / "CLAUDE.md"
            if claude_md.exists():
                try:
                    text = claude_md.read_text().lower()[:2000]
                    self._classify_plugin(plugin_name, text)
                except OSError:
                    pass

    def _classify_plugin(self, plugin_name: str, text: str) -> None:
        """Classify a plugin into capability categories based on its text."""
        for category, _, keywords in CAPABILITY_TAXONOMY:
            if any(kw in text for kw in keywords):
                if category not in self._capabilities:
                    self._capabilities[category] = []
                if plugin_name not in self._capabilities[category]:
                    self._capabilities[category].append(plugin_name)

    def _generate_queries(
        self, category: str, keywords: list[str]
    ) -> list[str]:
        """Generate search queries for adapter sources based on a gap."""
        base_queries = [
            f"MCP server {category}",
            f"Claude Code plugin {category}",
            f"AI agent {category} tool",
        ]
        keyword_queries = [f"{kw} automation tool" for kw in keywords[:3]]
        return base_queries + keyword_queries

    def get_gap_topics(self) -> list[str]:
        """Get gap categories as topic strings for adapter search."""
        gaps = self.scan()
        return [g["category"] for g in gaps]

    def is_gap_filling(self, text: str, gaps: list[dict] | None = None) -> bool:
        """Check if a discovery text fills a detected gap."""
        if gaps is None:
            gaps = self.scan()
        text_lower = text.lower()
        for gap in gaps:
            category = gap["category"]
            # Check if discovery text mentions the gap category keywords
            for _, _, keywords in CAPABILITY_TAXONOMY:
                if _ == category and any(kw in text_lower for kw in keywords):
                    return True
        return False
