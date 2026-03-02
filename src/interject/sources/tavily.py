"""Tavily source adapter — AI-powered web search API."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

TAVILY_API = "https://api.tavily.com/search"


class TavilyAdapter:
    name = "tavily"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.api_key: str = os.environ.get("TAVILY_API_KEY", "")
        self.max_results: int = cfg.get("max_results", 20)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Search Tavily for each topic and return discoveries."""
        if not self.api_key:
            return []

        discoveries: list[RawDiscovery] = []
        queries = topics[:10] if topics else ["AI agent tooling"]

        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    payload = {
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": self.max_results,
                    }
                    async with session.post(TAVILY_API, json=payload) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    for item in data.get("results", []):
                        url = item.get("url", "")
                        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]
                        title = item.get("title", "")
                        content = item.get("content", "")
                        score = item.get("score", 0.0)

                        discoveries.append(
                            RawDiscovery(
                                source=self.name,
                                source_id=source_id,
                                title=title,
                                summary=content[:500],
                                url=url,
                                metadata={
                                    "tavily_score": score,
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
