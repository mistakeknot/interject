"""Semantic Scholar source adapter — searches academic papers via the S2 API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarAdapter:
    name = "semantic_scholar"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.max_results: int = cfg.get("max_results", 20)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Search Semantic Scholar for each topic and return discoveries."""
        queries = topics[:10] if topics else ["AI agents"]
        discoveries: list[RawDiscovery] = []

        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    params = {
                        "query": query,
                        "limit": min(self.max_results, 100),
                        "fields": "title,abstract,url,year,citationCount",
                    }
                    async with session.get(S2_API, params=params) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    for paper in data.get("data", []):
                        paper_id = paper.get("paperId", "")
                        if not paper_id:
                            continue

                        title = paper.get("title", "")
                        abstract = paper.get("abstract", "") or ""
                        url = paper.get("url", "") or f"https://www.semanticscholar.org/paper/{paper_id}"
                        year = paper.get("year")
                        citation_count = paper.get("citationCount", 0)

                        discoveries.append(
                            RawDiscovery(
                                source=self.name,
                                source_id=paper_id,
                                title=title,
                                summary=abstract[:500],
                                url=url,
                                metadata={
                                    "year": year,
                                    "citation_count": citation_count,
                                    "full_abstract": abstract,
                                    "matched_query": query,
                                },
                            )
                        )
                except Exception:
                    continue

        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with full abstract for embedding."""
        full_abstract = discovery.metadata.get("full_abstract", discovery.summary)
        year = discovery.metadata.get("year")
        citations = discovery.metadata.get("citation_count", 0)

        tags = [discovery.metadata.get("matched_query", "")]
        if year:
            tags.append(str(year))
        if citations and citations > 100:
            tags.append("highly-cited")

        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=f"{discovery.title}. {full_abstract}",
            tags=tags,
        )
