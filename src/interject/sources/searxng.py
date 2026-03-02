"""SearXNG source adapter — searches a self-hosted SearXNG metasearch instance."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery


class SearXNGAdapter:
    name = "searxng"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.base_url: str = os.environ.get("SEARXNG_URL", "")
        self.max_results: int = cfg.get("max_results", 20)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Search SearXNG for each topic and return discoveries."""
        if not self.base_url:
            return []

        discoveries: list[RawDiscovery] = []
        queries = topics[:10] if topics else ["AI agent tooling"]
        search_url = f"{self.base_url.rstrip('/')}/search"

        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    params = {
                        "q": query,
                        "format": "json",
                        "categories": "general",
                    }
                    async with session.get(search_url, params=params) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    for item in data.get("results", [])[:self.max_results]:
                        url = item.get("url", "")
                        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]
                        title = item.get("title", "")
                        content = item.get("content", "")

                        discoveries.append(
                            RawDiscovery(
                                source=self.name,
                                source_id=source_id,
                                title=title,
                                summary=content[:500],
                                url=url,
                                metadata={
                                    "full_content": content,
                                    "matched_query": query,
                                },
                            )
                        )
                except Exception:
                    continue

        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with full content for embedding."""
        full_content = discovery.metadata.get("full_content", discovery.summary)
        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=f"{discovery.title}. {full_content}",
            tags=[discovery.metadata.get("matched_query", "")],
        )
