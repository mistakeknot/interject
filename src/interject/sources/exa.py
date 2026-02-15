"""Exa search source adapter — semantic web search for high-quality research."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

EXA_API = "https://api.exa.ai/search"


class ExaAdapter:
    name = "exa"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.max_results: int = cfg.get("max_results", 30)
        self.use_autoprompt: bool = cfg.get("use_autoprompt", True)
        self.search_queries: list[str] = cfg.get("search_queries", [
            "new MCP server tools for AI agents",
            "Claude Code plugin development",
            "multi-agent orchestration framework",
            "code analysis LLM tooling",
            "developer workflow automation AI",
        ])

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Fetch results from Exa semantic search."""
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            return []

        # Combine configured queries with dynamic topics
        queries = list(self.search_queries)
        for topic in topics[:5]:
            queries.append(f"{topic} tools and frameworks 2025 2026")

        discoveries = []
        seen_urls = set()

        async with aiohttp.ClientSession() as session:
            for query in queries:
                payload = {
                    "query": query,
                    "numResults": min(self.max_results, 10),
                    "useAutoprompt": self.use_autoprompt,
                    "startPublishedDate": since.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "contents": {
                        "text": {"maxCharacters": 1000},
                        "highlights": {"numSentences": 3},
                    },
                }

                try:
                    async with session.post(
                        EXA_API,
                        json=payload,
                        headers={
                            "x-api-key": api_key,
                            "Content-Type": "application/json",
                        },
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                except Exception:
                    continue

                for result in data.get("results", []):
                    url = result.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = result.get("title", "") or ""
                    text = result.get("text", "") or ""
                    highlights = result.get("highlights", [])
                    published = result.get("publishedDate", "")
                    author = result.get("author", "") or ""
                    score = result.get("score", 0)

                    # Use URL as source_id (unique per result)
                    source_id = url.split("//")[-1][:80].replace("/", "_")

                    summary = text[:300] if text else " ".join(highlights)[:300]

                    discoveries.append(
                        RawDiscovery(
                            source=self.name,
                            source_id=source_id,
                            title=title,
                            summary=summary,
                            url=url,
                            metadata={
                                "exa_score": score,
                                "author": author,
                                "published_date": published,
                                "highlights": highlights,
                                "full_text": text[:2000],
                                "matched_query": query,
                            },
                        )
                    )

        return discoveries[:self.max_results]

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with full text content for better embedding."""
        full_text = discovery.metadata.get("full_text", "")
        highlights = discovery.metadata.get("highlights", [])

        embed_text = f"{discovery.title}. {full_text}" if full_text else discovery.title
        if highlights:
            embed_text += " " + " ".join(highlights)

        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=embed_text[:3000],
            tags=[discovery.metadata.get("matched_query", "")],
        )
