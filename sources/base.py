"""Base protocol and data classes for source adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class RawDiscovery:
    """A raw finding from a source adapter before scoring."""

    source: str
    source_id: str
    title: str
    summary: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EnrichedDiscovery(RawDiscovery):
    """A discovery enriched with additional context from the source."""

    full_text: str = ""  # for better embedding (abstract, full description, etc.)
    tags: list[str] = field(default_factory=list)


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol for pluggable source adapters."""

    name: str

    async def fetch(
        self, since: datetime, topics: list[str]
    ) -> list[RawDiscovery]:
        """Fetch new items from the source since the given datetime.

        Args:
            since: Only return items newer than this
            topics: Interest topics to filter/search by

        Returns:
            List of raw discoveries
        """
        ...

    async def enrich(self, discovery: RawDiscovery) -> EnrichedDiscovery:
        """Enrich a raw discovery with additional context.

        Default implementation just copies fields. Adapters can override
        to fetch abstracts, README content, etc.
        """
        ...
