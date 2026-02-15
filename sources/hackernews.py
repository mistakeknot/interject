"""Hacker News source adapter — searches via Algolia API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

HN_ALGOLIA_API = "https://hn.algolia.com/api/v1/search"


class HackerNewsAdapter:
    name = "hackernews"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.keywords: list[str] = cfg.get(
            "keywords", ["MCP", "Claude", "agent", "LLM tooling", "code analysis"]
        )
        self.min_points: int = cfg.get("min_points", 10)
        self.max_results: int = cfg.get("max_results", 50)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Fetch recent HN stories matching keywords and topics."""
        all_terms = list(set(self.keywords + topics[:10]))
        discoveries = []

        async with aiohttp.ClientSession() as session:
            for term in all_terms:
                params = {
                    "query": term,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{int(since.timestamp())},points>{self.min_points}",
                    "hitsPerPage": min(self.max_results, 20),
                }
                async with session.get(HN_ALGOLIA_API, params=params) as resp:
                    data = await resp.json()

                for hit in data.get("hits", []):
                    source_id = str(hit.get("objectID", ""))
                    # Deduplicate across keyword searches
                    if any(d.source_id == source_id for d in discoveries):
                        continue

                    title = hit.get("title", "") or ""
                    url = hit.get("url", "") or f"https://news.ycombinator.com/item?id={source_id}"
                    points = hit.get("points", 0) or 0
                    num_comments = hit.get("num_comments", 0) or 0
                    author = hit.get("author", "") or ""
                    created_at = hit.get("created_at", "")

                    discoveries.append(
                        RawDiscovery(
                            source=self.name,
                            source_id=source_id,
                            title=title,
                            summary=f"{points} points, {num_comments} comments by {author}",
                            url=url,
                            metadata={
                                "points": points,
                                "num_comments": num_comments,
                                "author": author,
                                "created_at": created_at,
                                "hn_url": f"https://news.ycombinator.com/item?id={source_id}",
                                "matched_term": term,
                            },
                        )
                    )

        return discoveries[:self.max_results]

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with HN discussion context."""
        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=discovery.title,
            tags=[discovery.metadata.get("matched_term", "")],
        )
