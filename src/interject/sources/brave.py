"""Brave Search source adapter — privacy-focused web search API."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

BRAVE_API = "https://api.search.brave.com/res/v1/web/search"


class BraveAdapter:
    name = "brave"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.api_key: str = os.environ.get("BRAVE_API_KEY", "")
        self.max_results: int = cfg.get("max_results", 20)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Search Brave for each topic and return discoveries."""
        if not self.api_key:
            return []

        discoveries: list[RawDiscovery] = []
        queries = topics[:10] if topics else ["AI agent tooling"]
        headers = {"X-Subscription-Token": self.api_key}

        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    params = {"q": query, "count": self.max_results}
                    async with session.get(
                        BRAVE_API, params=params, headers=headers
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    web_results = data.get("web", {}).get("results", [])
                    for item in web_results:
                        url = item.get("url", "")
                        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]
                        title = item.get("title", "")
                        description = item.get("description", "")

                        discoveries.append(
                            RawDiscovery(
                                source=self.name,
                                source_id=source_id,
                                title=title,
                                summary=description[:500],
                                url=url,
                                metadata={
                                    "description": description,
                                    "matched_query": query,
                                },
                            )
                        )
                except Exception:
                    continue

        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with full description for embedding."""
        description = discovery.metadata.get("description", discovery.summary)
        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=f"{discovery.title}. {description}",
            tags=[discovery.metadata.get("matched_query", "")],
        )
