"""GitHub source adapter — searches repos via REST API."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import aiohttp

from .base import EnrichedDiscovery, RawDiscovery

GITHUB_API = "https://api.github.com"


class GitHubAdapter:
    name = "github"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.topics: list[str] = cfg.get(
            "topics", ["mcp-server", "claude-code", "ai-agent", "code-analysis", "llm-tools"]
        )
        self.min_stars: int = cfg.get("min_stars", 5)
        self.max_results: int = cfg.get("max_results", 50)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Fetch recently created/updated repos matching topics."""
        all_topics = list(set(self.topics + topics[:10]))
        discoveries = []
        since_str = since.strftime("%Y-%m-%d")

        async with aiohttp.ClientSession() as session:
            for topic in all_topics:
                query = f"{topic} in:name,description,topics created:>{since_str} stars:>={self.min_stars}"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(self.max_results, 30),
                }
                async with session.get(
                    f"{GITHUB_API}/search/repositories",
                    params=params,
                    headers=self._headers(),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()

                for repo in data.get("items", []):
                    full_name = repo.get("full_name", "")
                    # Deduplicate across topic searches
                    if any(d.source_id == full_name for d in discoveries):
                        continue

                    name = repo.get("name", "")
                    description = repo.get("description", "") or ""
                    stars = repo.get("stargazers_count", 0)
                    language = repo.get("language", "") or ""
                    html_url = repo.get("html_url", "")
                    created = repo.get("created_at", "")
                    updated = repo.get("updated_at", "")
                    repo_topics = repo.get("topics", [])

                    discoveries.append(
                        RawDiscovery(
                            source=self.name,
                            source_id=full_name,
                            title=f"{full_name}: {description[:100]}",
                            summary=f"{stars} stars, {language}, topics: {', '.join(repo_topics[:5])}",
                            url=html_url,
                            metadata={
                                "full_name": full_name,
                                "name": name,
                                "description": description,
                                "stars": stars,
                                "language": language,
                                "topics": repo_topics,
                                "created_at": created,
                                "updated_at": updated,
                                "matched_topic": topic,
                            },
                        )
                    )

        # Sort by stars descending, cap at max_results
        discoveries.sort(key=lambda d: d.metadata.get("stars", 0), reverse=True)
        return discoveries[:self.max_results]

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich with README content for better embedding."""
        full_name = discovery.metadata.get("full_name", "")
        readme_text = ""

        if full_name:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{GITHUB_API}/repos/{full_name}/readme",
                        headers={**self._headers(), "Accept": "application/vnd.github.raw+json"},
                    ) as resp:
                        if resp.status == 200:
                            readme_text = (await resp.text())[:2000]
            except Exception:
                pass

        return EnrichedDiscovery(
            source=discovery.source,
            source_id=discovery.source_id,
            title=discovery.title,
            summary=discovery.summary,
            url=discovery.url,
            metadata=discovery.metadata,
            discovered_at=discovery.discovered_at,
            full_text=f"{discovery.title}. {discovery.metadata.get('description', '')}. {readme_text}",
            tags=discovery.metadata.get("topics", []),
        )
