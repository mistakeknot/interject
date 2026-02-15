"""Output pipeline — creates beads, briefing docs, and draft plans."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OutputPipeline:
    """Generates outputs based on confidence tier."""

    def __init__(
        self,
        docs_root: Path | None = None,
        interverse_root: Path | None = None,
    ):
        self.docs_root = docs_root or Path("/root/projects/Interverse/docs")
        self.interverse_root = interverse_root or Path("/root/projects/Interverse")

    def process(self, discovery: dict, tier: str) -> dict[str, Any]:
        """Process a scored discovery through the output pipeline.

        Args:
            discovery: Discovery dict from the database
            tier: Confidence tier ('high', 'medium', 'low')

        Returns:
            Dict with keys: bead_id, briefing_path, plan_path (if applicable)
        """
        result: dict[str, Any] = {"tier": tier}

        if tier == "low":
            # Record only — no external output
            return result

        if tier in ("medium", "high"):
            # Create bead
            bead_id = self._create_bead(discovery, tier)
            result["bead_id"] = bead_id

            # Write briefing doc
            briefing_path = self._write_briefing(discovery)
            result["briefing_path"] = str(briefing_path)

        if tier == "high":
            # Draft implementation plan
            plan_path = self._write_plan(discovery)
            result["plan_path"] = str(plan_path)

        return result

    def _create_bead(self, discovery: dict, tier: str) -> str | None:
        """Create a bead via bd CLI."""
        title = f"[interject] {discovery['title'][:80]}"
        priority = 2 if tier == "high" else 3
        description = (
            f"Source: {discovery['source']} | {discovery['url']}\n\n"
            f"{discovery.get('summary', '')}\n\n"
            f"Relevance score: {discovery.get('relevance_score', 0):.2f}\n"
            f"Discovered: {discovery.get('discovered_at', '')}"
        )

        try:
            result = subprocess.run(
                [
                    "bd", "create",
                    f"--title={title}",
                    "--type=task",
                    f"--priority={priority}",
                    f"--description={description}",
                ],
                capture_output=True,
                text=True,
                cwd=str(self.interverse_root),
                timeout=10,
            )
            if result.returncode == 0:
                # Parse bead ID from output like "Created issue: iv-abc"
                for line in result.stdout.splitlines():
                    if "Created issue:" in line or "Created" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p.endswith(":") and i + 1 < len(parts):
                                return parts[i + 1]
                            # Also handle "iv-xxx" pattern
                            if p.startswith("iv-"):
                                return p
            else:
                logger.warning("bd create failed: %s", result.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Failed to create bead: %s", e)
        return None

    def _write_briefing(self, discovery: dict) -> Path:
        """Write a briefing doc to docs/research/."""
        research_dir = self.docs_root / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug = _slugify(discovery["title"])[:60]
        filename = f"{date_str}-interject-{slug}.md"
        filepath = research_dir / filename

        metadata = discovery.get("raw_metadata", "{}")
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        content = f"""# {discovery['title']}

**Source:** {discovery['source']}
**URL:** {discovery['url']}
**Relevance Score:** {discovery.get('relevance_score', 0):.2f}
**Confidence Tier:** {discovery.get('confidence_tier', 'unknown')}
**Discovered:** {discovery.get('discovered_at', 'unknown')}

## Summary

{discovery.get('summary', 'No summary available.')}

## Source Details

"""
        # Add source-specific details
        if discovery["source"] == "arxiv":
            authors = metadata.get("authors", [])
            if authors:
                content += f"**Authors:** {', '.join(authors[:5])}\n"
            categories = metadata.get("categories", [])
            if categories:
                content += f"**Categories:** {', '.join(categories)}\n"
        elif discovery["source"] == "github":
            content += f"**Stars:** {metadata.get('stars', 'N/A')}\n"
            content += f"**Language:** {metadata.get('language', 'N/A')}\n"
            topics = metadata.get("topics", [])
            if topics:
                content += f"**Topics:** {', '.join(topics[:10])}\n"
        elif discovery["source"] == "hackernews":
            content += f"**Points:** {metadata.get('points', 'N/A')}\n"
            content += f"**Comments:** {metadata.get('num_comments', 'N/A')}\n"
            hn_url = metadata.get("hn_url", "")
            if hn_url:
                content += f"**Discussion:** {hn_url}\n"

        content += f"""
## Relevance Analysis

This discovery was identified by Interject's ambient scanning engine.
It scored {discovery.get('relevance_score', 0):.2f} against the current interest profile.

---
*Generated by Interject on {date_str}*
"""
        filepath.write_text(content)
        logger.info("Wrote briefing: %s", filepath)
        return filepath

    def _write_plan(self, discovery: dict) -> Path:
        """Write a draft implementation plan to docs/plans/."""
        plans_dir = self.docs_root / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug = _slugify(discovery["title"])[:60]
        filename = f"{date_str}-interject-plan-{slug}.md"
        filepath = plans_dir / filename

        content = f"""# Draft Plan: {discovery['title']}

**Status:** Draft (auto-generated by Interject)
**Source:** {discovery['source']} — {discovery['url']}
**Relevance Score:** {discovery.get('relevance_score', 0):.2f}
**Generated:** {date_str}

## What

{discovery.get('summary', 'No summary available.')}

## Why

This capability was identified as high-relevance by Interject's recommendation engine,
scoring {discovery.get('relevance_score', 0):.2f} against the current interest profile.

## Integration Approach

> This is an auto-generated draft. Review and refine before implementation.

### Option 1: Direct Integration
- Evaluate the tool/paper for direct use in the current workflow
- Identify which existing plugins/services would benefit

### Option 2: New Plugin/Module
- If the capability is substantial enough, consider a new inter-module
- Follow the existing plugin scaffold pattern

### Option 3: Enhancement to Existing Module
- Check if this capability fits as an extension to an existing plugin

## Next Steps

1. Review this draft plan
2. If promising, promote via `/interject:promote` with appropriate priority
3. Create implementation beads if proceeding
4. Assign to a sprint cycle

---
*Auto-generated by Interject — review before acting*
"""
        filepath.write_text(content)
        logger.info("Wrote draft plan: %s", filepath)
        return filepath


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
