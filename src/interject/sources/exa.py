"""Exa source adapter — uses intersearch shared Exa client."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from intersearch.exa import multi_search as exa_multi_search

from .base import EnrichedDiscovery, RawDiscovery


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

    async def fetch(self, since: datetime, topics: list[str]) -> list[RawDiscovery]:
        queries = list(self.search_queries)
        for topic in topics[:5]:
            queries.append(f"{topic} tools and frameworks 2025 2026")

        results = await exa_multi_search(
            queries,
            num_results=min(self.max_results, 10),
            use_autoprompt=self.use_autoprompt,
            start_date=since,
        )

        discoveries = []
        for r in results[:self.max_results]:
            source_id = r.url.split("//")[-1][:80].replace("/", "_")
            summary = r.text[:300] if r.text else " ".join(r.highlights)[:300]
            discoveries.append(
                RawDiscovery(
                    source=self.name,
                    source_id=source_id,
                    title=r.title,
                    summary=summary,
                    url=r.url,
                    metadata={
                        "exa_score": r.score,
                        "author": r.author,
                        "published_date": r.published_date,
                        "highlights": r.highlights,
                        "full_text": r.text[:2000],
                        "matched_query": r.matched_query,
                    },
                )
            )
        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
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
