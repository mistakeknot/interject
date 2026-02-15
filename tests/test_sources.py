"""Tests for source adapters."""

from datetime import datetime, timedelta

import pytest

from sources.base import EnrichedDiscovery, RawDiscovery, SourceAdapter


class TestRawDiscovery:
    def test_creation(self):
        d = RawDiscovery(
            source="test",
            source_id="123",
            title="Test Discovery",
            summary="A summary",
            url="https://example.com",
        )
        assert d.source == "test"
        assert d.source_id == "123"
        assert d.title == "Test Discovery"

    def test_defaults(self):
        d = RawDiscovery(source="test", source_id="1", title="T")
        assert d.summary == ""
        assert d.url == ""
        assert d.metadata == {}


class TestEnrichedDiscovery:
    def test_inherits_raw(self):
        d = EnrichedDiscovery(
            source="test",
            source_id="1",
            title="Test",
            full_text="Full text content",
            tags=["ai", "mcp"],
        )
        assert d.full_text == "Full text content"
        assert d.tags == ["ai", "mcp"]


class TestSourceAdapterProtocol:
    def test_protocol_check(self):
        """Verify that a class implementing the right methods satisfies the protocol."""

        class MockAdapter:
            name = "mock"

            async def fetch(self, since, topics):
                return []

            async def enrich(self, discovery):
                return EnrichedDiscovery(
                    source=discovery.source,
                    source_id=discovery.source_id,
                    title=discovery.title,
                )

        adapter = MockAdapter()
        assert isinstance(adapter, SourceAdapter)


class TestAdapterImports:
    """Verify all adapters can be imported."""

    def test_import_arxiv(self):
        from sources.arxiv import ArxivAdapter
        adapter = ArxivAdapter()
        assert adapter.name == "arxiv"

    def test_import_hackernews(self):
        from sources.hackernews import HackerNewsAdapter
        adapter = HackerNewsAdapter()
        assert adapter.name == "hackernews"

    def test_import_github(self):
        from sources.github import GitHubAdapter
        adapter = GitHubAdapter()
        assert adapter.name == "github"

    def test_import_anthropic(self):
        from sources.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()
        assert adapter.name == "anthropic"

    def test_import_exa(self):
        from sources.exa import ExaAdapter
        adapter = ExaAdapter()
        assert adapter.name == "exa"


class TestAdapterConfig:
    def test_arxiv_default_categories(self):
        from sources.arxiv import ArxivAdapter
        adapter = ArxivAdapter()
        assert "cs.AI" in adapter.categories

    def test_arxiv_custom_config(self):
        from sources.arxiv import ArxivAdapter
        adapter = ArxivAdapter(config={"categories": ["cs.LG"], "max_results": 10})
        assert adapter.categories == ["cs.LG"]
        assert adapter.max_results == 10

    def test_hackernews_default_keywords(self):
        from sources.hackernews import HackerNewsAdapter
        adapter = HackerNewsAdapter()
        assert "MCP" in adapter.keywords

    def test_github_default_topics(self):
        from sources.github import GitHubAdapter
        adapter = GitHubAdapter()
        assert "mcp-server" in adapter.topics
