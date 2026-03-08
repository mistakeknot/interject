"""Output pipeline — creates beads, briefing docs, brainstorms, and digests."""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
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

    def process(
        self, discovery: dict, tier: str, discovery_id: str | None = None
    ) -> dict[str, Any]:
        """Process a scored discovery through the output pipeline.

        Args:
            discovery: Discovery dict from the database
            tier: Confidence tier ('high', 'medium', 'low')
            discovery_id: Optional discovery ID for kernel linkage

        Returns:
            Dict with keys: tier, kernel_discovery_id, bead_id,
            briefing_path, brainstorm_path (if applicable)
        """
        result: dict[str, Any] = {"tier": tier}

        # Submit to kernel for all tiers (durable record + events)
        kernel_id = self._submit_to_kernel(discovery)
        if kernel_id:
            result["kernel_discovery_id"] = kernel_id

        if tier == "low":
            return result

        if tier == "medium":
            bead_id = self._create_bead(discovery, tier, priority=4, labels=["pending_triage"])
            result["bead_id"] = bead_id
            briefing_path = self._write_briefing(discovery)
            result["briefing_path"] = str(briefing_path)

        elif tier == "high":
            bead_id = self._create_bead(discovery, tier, priority=2)
            result["bead_id"] = bead_id
            briefing_path = self._write_briefing(discovery)
            result["briefing_path"] = str(briefing_path)
            brainstorm_path = self._write_brainstorm(discovery)
            result["brainstorm_path"] = str(brainstorm_path)

        if kernel_id and result.get("bead_id"):
            self._promote_in_kernel(kernel_id, result["bead_id"])

        return result

    def _create_bead(
        self, discovery: dict, tier: str,
        priority: int = 2, labels: list[str] | None = None,
    ) -> str | None:
        """Create a bead via bd CLI."""
        title = f"[interject] {discovery['title'][:80]}"
        description = (
            f"Source: {discovery['source']} | {discovery['url']}\n\n"
            f"{discovery.get('summary', '')}\n\n"
            f"Relevance score: {discovery.get('relevance_score', 0):.2f}\n"
            f"Discovered: {discovery.get('discovered_at', '')}"
        )

        cmd = [
            "bd", "create",
            f"--title={title}",
            "--type=task",
            f"--priority={priority}",
            f"--description={description}",
        ]
        if labels:
            cmd.append(f"--labels={','.join(labels)}")

        try:
            result = subprocess.run(
                cmd,
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

    def _submit_to_kernel(self, discovery: dict) -> str | None:
        """Submit discovery to kernel via ic CLI. Returns kernel discovery ID or None."""
        import json as _json
        import os
        import tempfile

        cmd = [
            "ic", "discovery", "submit",
            f"--source={discovery['source']}",
            f"--source-id={discovery.get('id', '')}",
            f"--title={discovery['title'][:200]}",
            f"--score={discovery.get('relevance_score', 0):.4f}",
        ]

        if discovery.get("summary"):
            cmd.append(f"--summary={discovery['summary'][:500]}")
        if discovery.get("url"):
            cmd.append(f"--url={discovery['url']}")

        embedding_path = None
        metadata_path = None
        try:
            if discovery.get("embedding"):
                fd, embedding_path = tempfile.mkstemp(suffix=".bin")
                with os.fdopen(fd, "wb") as f:
                    if isinstance(discovery["embedding"], bytes):
                        f.write(discovery["embedding"])
                    else:
                        f.write(bytes(discovery["embedding"]))
                cmd.append(f"--embedding={embedding_path}")

            metadata = discovery.get("raw_metadata")
            if metadata:
                fd, metadata_path = tempfile.mkstemp(suffix=".json")
                with os.fdopen(fd, "w") as f:
                    if isinstance(metadata, str):
                        f.write(metadata)
                    else:
                        _json.dump(metadata, f)
                cmd.append(f"--metadata={metadata_path}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )

            if result.returncode == 0:
                kernel_id = result.stdout.strip()
                logger.info("Submitted to kernel: %s -> %s", discovery.get("id"), kernel_id)
                return kernel_id
            else:
                logger.warning("ic discovery submit failed (rc=%d): %s", result.returncode, result.stderr)

        except FileNotFoundError:
            logger.warning("ic CLI not found — kernel submit skipped (install intercore)")
        except subprocess.TimeoutExpired:
            logger.warning("ic discovery submit timed out")
        finally:
            if embedding_path:
                try:
                    os.unlink(embedding_path)
                except OSError:
                    pass
            if metadata_path:
                try:
                    os.unlink(metadata_path)
                except OSError:
                    pass

        return None

    def _promote_in_kernel(self, kernel_discovery_id: str, bead_id: str) -> bool:
        """Link kernel discovery record to bead via ic discovery promote."""
        try:
            result = subprocess.run(
                ["ic", "discovery", "promote", kernel_discovery_id, f"--bead-id={bead_id}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info("Kernel promote: %s -> %s", kernel_discovery_id, bead_id)
                return True
            else:
                logger.warning("ic discovery promote failed (rc=%d): %s", result.returncode, result.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("ic discovery promote skipped: %s", e)
        return False

    def _write_briefing(self, discovery: dict) -> Path:
        """Write a briefing doc to docs/research/."""
        research_dir = self.docs_root / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug = _slugify(discovery["title"])[:60]
        filename = f"{date_str}-interject-{slug}.md"
        filepath = research_dir / filename

        metadata = _metadata_dict(discovery)

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

    def _write_brainstorm(self, discovery: dict) -> Path:
        """Write a Clavain-format brainstorm doc to docs/brainstorms/."""
        brainstorms_dir = self.docs_root / "brainstorms"
        brainstorms_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug = _slugify(discovery["title"])[:60]
        filename = f"{date_str}-interject-{slug}-brainstorm.md"
        filepath = brainstorms_dir / filename

        source_details = _format_source_details(
            discovery.get("source", "unknown"),
            _metadata_dict(discovery),
            bullet_list=True,
        )
        content = f"""# {discovery['title']}

**Date:** {date_str}
**Status:** Auto-generated by Interject — review before acting

## What We're Building

{discovery.get('summary', 'No summary available.')}

## Why This Approach

This was identified by Interject's ambient scanning engine with a relevance score of {discovery.get('relevance_score', 0):.2f}.

Source: {discovery.get('source', 'unknown')} — {discovery.get('url', '')}

## Key Decisions

> Review needed — auto-generated from discovery metadata.

- **Integration approach:** TBD (confirm implementation shape after review)
- **Priority:** {_priority_hint(discovery.get('confidence_tier', 'high'))}

## Source Details

{source_details}
"""
        filepath.write_text(content)
        logger.info("Wrote brainstorm: %s", filepath)
        return filepath

    def generate_digest(self, discoveries: list[dict]) -> Path:
        """Generate a digest doc grouping discoveries by source."""
        research_dir = self.docs_root / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        filepath = research_dir / f"{date_str}-interject-digest.md"

        grouped: dict[str, list[dict]] = defaultdict(list)
        for discovery in discoveries:
            grouped[discovery.get("source", "unknown")].append(discovery)

        lines = [
            f"# Interject Digest — {date_str}",
            "",
            f"Total discoveries: {len(discoveries)}",
            "",
            "## Summary Table",
            "",
            "| Source | Count | Avg Score | High | Medium | Low |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for source in sorted(grouped):
            source_discoveries = grouped[source]
            scores = [float(d.get("relevance_score", 0) or 0) for d in source_discoveries]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            tiers = {
                "high": sum(1 for d in source_discoveries if d.get("confidence_tier") == "high"),
                "medium": sum(1 for d in source_discoveries if d.get("confidence_tier") == "medium"),
                "low": sum(1 for d in source_discoveries if d.get("confidence_tier") == "low"),
            }
            lines.append(
                f"| {source} | {len(source_discoveries)} | {avg_score:.2f} | "
                f"{tiers['high']} | {tiers['medium']} | {tiers['low']} |"
            )

        lines.extend(["", "## Discoveries by Source", ""])

        for source in sorted(grouped):
            source_discoveries = sorted(
                grouped[source],
                key=lambda d: float(d.get("relevance_score", 0) or 0),
                reverse=True,
            )
            lines.append(f"### {source} ({len(source_discoveries)})")
            lines.append("")

            for discovery in source_discoveries:
                title = discovery.get("title", "Untitled")
                url = discovery.get("url", "")
                score = float(discovery.get("relevance_score", 0) or 0)
                tier = discovery.get("confidence_tier", "unknown")

                if url:
                    lines.append(f"- **[{title}]({url})** — score {score:.2f}, tier {tier}")
                else:
                    lines.append(f"- **{title}** — score {score:.2f}, tier {tier}")

                summary = (discovery.get("summary") or "").strip()
                if summary:
                    lines.append(f"  Summary: {summary}")
            lines.append("")

        filepath.write_text("\n".join(lines))
        logger.info("Wrote digest: %s", filepath)
        return filepath


def _metadata_dict(discovery: dict) -> dict:
    """Return discovery metadata as a dict."""
    import json

    metadata = discovery.get("raw_metadata", "{}")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def _format_source_details(source: str, metadata: dict, bullet_list: bool = False) -> str:
    """Format source-specific metadata as markdown."""
    lines: list[str] = []
    prefix = "- " if bullet_list else ""

    if source == "arxiv":
        authors = metadata.get("authors", [])
        if authors:
            lines.append(f"{prefix}**Authors:** {', '.join(authors[:5])}")
        categories = metadata.get("categories", [])
        if categories:
            lines.append(f"{prefix}**Categories:** {', '.join(categories)}")
    elif source == "github":
        if "stars" in metadata:
            lines.append(f"{prefix}**Stars:** {metadata.get('stars', 'N/A')}")
        if "language" in metadata:
            lines.append(f"{prefix}**Language:** {metadata.get('language', 'N/A')}")
        topics = metadata.get("topics", [])
        if topics:
            lines.append(f"{prefix}**Topics:** {', '.join(topics[:10])}")
    elif source == "hackernews":
        if "points" in metadata:
            lines.append(f"{prefix}**Points:** {metadata.get('points', 'N/A')}")
        if "num_comments" in metadata:
            lines.append(
                f"{prefix}**Comments:** {metadata.get('num_comments', 'N/A')}"
            )
        hn_url = metadata.get("hn_url", "")
        if hn_url:
            lines.append(f"{prefix}**Discussion:** {hn_url}")

    if not lines:
        return "No additional source metadata available."
    return "\n".join(lines)


def _priority_hint(tier: str) -> str:
    """Return a priority suggestion based on confidence tier."""
    hints = {
        "high": "High (P2 bead by default)",
        "medium": "Medium (P4 bead by default)",
        "low": "Low (record-only unless manually promoted)",
    }
    return hints.get(tier, "Unspecified")


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
