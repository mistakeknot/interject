"""PubMed source adapter — searches biomedical literature via NCBI E-utilities."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

ESEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedAdapter:
    name = "pubmed"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.max_results: int = cfg.get("max_results", 30)
        self.courtesy_delay: float = cfg.get("courtesy_delay_seconds", 0.4)

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Search PubMed for each topic, fetch article details, return discoveries."""
        queries = topics[:10] if topics else ["artificial intelligence agents"]
        discoveries: list[RawDiscovery] = []

        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    pmids = await self._search_ids(session, query)
                    if not pmids:
                        continue
                    articles = await self._fetch_details(session, pmids)
                    for article in articles:
                        article.metadata["matched_query"] = query
                    discoveries.extend(articles)
                except Exception:
                    continue

        return discoveries

    async def _search_ids(
        self, session: aiohttp.ClientSession, query: str
    ) -> list[str]:
        """Search PubMed and return a list of PMIDs."""
        await asyncio.sleep(self.courtesy_delay)
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
        }
        async with session.get(ESEARCH_API, params=params) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

        return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_details(
        self, session: aiohttp.ClientSession, pmids: list[str]
    ) -> list[RawDiscovery]:
        """Fetch article details for a list of PMIDs."""
        await asyncio.sleep(self.courtesy_delay)
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        async with session.get(EFETCH_API, params=params) as resp:
            if resp.status != 200:
                return []
            text = await resp.text()

        return self._parse_articles(text)

    def _parse_articles(self, xml_text: str) -> list[RawDiscovery]:
        """Parse PubMed XML response into discoveries."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        discoveries = []
        for article_el in root.findall(".//PubmedArticle"):
            try:
                medline = article_el.find(".//MedlineCitation")
                if medline is None:
                    continue

                pmid_el = medline.find("PMID")
                pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
                if not pmid:
                    continue

                article = medline.find("Article")
                if article is None:
                    continue

                title_el = article.find("ArticleTitle")
                title = title_el.text if title_el is not None and title_el.text else ""

                abstract_el = article.find(".//AbstractText")
                abstract = (
                    abstract_el.text
                    if abstract_el is not None and abstract_el.text
                    else ""
                )

                # Extract authors
                authors = []
                for author_el in article.findall(".//Author"):
                    last = author_el.find("LastName")
                    first = author_el.find("ForeName")
                    if last is not None and last.text:
                        name = last.text
                        if first is not None and first.text:
                            name = f"{first.text} {name}"
                        authors.append(name)

                # Extract MeSH terms as tags
                mesh_terms = []
                for mesh_el in medline.findall(".//MeshHeading/DescriptorName"):
                    if mesh_el.text:
                        mesh_terms.append(mesh_el.text)

                discoveries.append(
                    RawDiscovery(
                        source=self.name,
                        source_id=pmid,
                        title=title,
                        summary=abstract[:500],
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        metadata={
                            "authors": authors,
                            "mesh_terms": mesh_terms,
                            "full_abstract": abstract,
                        },
                    )
                )
            except Exception:
                continue

        return discoveries

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with full abstract for embedding."""
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
            tags=discovery.metadata.get("mesh_terms", []),
        )
