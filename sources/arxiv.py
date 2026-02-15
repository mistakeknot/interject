"""arXiv source adapter — searches for papers by category and keywords."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivAdapter:
    name = "arxiv"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.categories: list[str] = cfg.get("categories", ["cs.AI", "cs.SE", "cs.MA", "cs.CL"])
        self.max_results: int = cfg.get("max_results", 50)
        self.courtesy_delay: float = cfg.get("courtesy_delay_seconds", 3.0)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Fetch recent papers matching categories and topics."""
        # Build query: categories OR'd, with topic keywords
        cat_query = " OR ".join(f"cat:{c}" for c in self.categories)
        if topics:
            topic_query = " OR ".join(f'all:"{t}"' for t in topics[:10])
            query = f"({cat_query}) AND ({topic_query})"
        else:
            query = cat_query

        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(ARXIV_API, params=params) as resp:
                text = await resp.text()

        return self._parse_feed(text, since)

    def _parse_feed(self, xml_text: str, since: datetime) -> list[RawDiscovery]:
        root = ET.fromstring(xml_text)
        discoveries = []

        for entry in root.findall(f"{ATOM_NS}entry"):
            published_str = _text(entry, f"{ATOM_NS}published")
            if not published_str:
                continue
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            if published.replace(tzinfo=None) < since:
                continue

            arxiv_id = _text(entry, f"{ATOM_NS}id", "").split("/abs/")[-1]
            title = _text(entry, f"{ATOM_NS}title", "").strip().replace("\n", " ")
            summary = _text(entry, f"{ATOM_NS}summary", "").strip().replace("\n", " ")

            authors = [
                _text(a, f"{ATOM_NS}name", "")
                for a in entry.findall(f"{ATOM_NS}author")
            ]

            categories = [
                c.get("term", "")
                for c in entry.findall("{http://arxiv.org/schemas/atom}category")
            ]

            # Get PDF link
            pdf_url = ""
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            discoveries.append(
                RawDiscovery(
                    source=self.name,
                    source_id=arxiv_id,
                    title=title,
                    summary=summary[:500],
                    url=pdf_url or f"https://arxiv.org/abs/{arxiv_id}",
                    metadata={
                        "authors": authors,
                        "categories": categories,
                        "published": published_str,
                        "full_abstract": summary,
                    },
                    discovered_at=published.replace(tzinfo=None),
                )
            )

        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with full abstract for better embedding."""
        full_abstract = discovery.metadata.get("full_abstract", discovery.summary)
        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=f"{discovery.title}. {full_abstract}",
            tags=discovery.metadata.get("categories", []),
        )


def _text(el: ET.Element, tag: str, default: str = "") -> str:
    child = el.find(tag)
    return child.text if child is not None and child.text else default
