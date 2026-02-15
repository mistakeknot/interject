"""Anthropic docs source adapter — monitors changelog and docs for updates."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

DOCS_URL = "https://docs.anthropic.com"
CHANGELOG_URL = f"{DOCS_URL}/en/docs/about-claude/models"


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.check_changelog: bool = cfg.get("check_changelog", True)
        self.check_docs: bool = cfg.get("check_docs", True)
        self.watch_pages: list[str] = cfg.get("watch_pages", [
            "https://docs.anthropic.com/en/docs/about-claude/models",
            "https://docs.anthropic.com/en/docs/build-with-claude/tool-use",
            "https://docs.anthropic.com/en/docs/agents-and-tools/mcp",
        ])

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Check Anthropic docs pages for changes.

        Since Anthropic doesn't have a structured changelog API, we fetch
        pages and hash them to detect changes. The hash comparison happens
        against previously stored metadata.
        """
        discoveries = []

        async with aiohttp.ClientSession() as session:
            for url in self.watch_pages:
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        html = await resp.text()

                    # Extract text content (rough — strip HTML tags)
                    text = re.sub(r"<[^>]+>", " ", html)
                    text = re.sub(r"\s+", " ", text).strip()

                    content_hash = hashlib.sha256(text[:5000].encode()).hexdigest()[:16]
                    page_name = url.split("/")[-1] or "index"

                    # Create a discovery keyed by URL + hash
                    # The db's UNIQUE(source, source_id) handles dedup —
                    # same URL+hash won't insert twice
                    discoveries.append(
                        RawDiscovery(
                            source=self.name,
                            source_id=f"{page_name}-{content_hash}",
                            title=f"Anthropic docs update: {page_name}",
                            summary=text[:300],
                            url=url,
                            metadata={
                                "page": page_name,
                                "content_hash": content_hash,
                                "content_preview": text[:1000],
                            },
                        )
                    )
                except Exception:
                    continue

        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=discovery.metadata.get("content_preview", discovery.summary),
            tags=["anthropic", "docs", discovery.metadata.get("page", "")],
        )
