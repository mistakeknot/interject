"""Interject MCP server — exposes discovery, recommendation, and profile tools."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import get_db_path, get_session_config, load_config
from .db import InterjectDB
from .embeddings import EmbeddingClient
from .engine import RecommendationEngine
from .scanner import Scanner

logger = logging.getLogger(__name__)


def create_server(config: dict | None = None) -> tuple[Server, dict]:
    """Create and configure the MCP server with all tools."""
    if config is None:
        config = load_config()

    server = Server("interject")

    # Shared state
    db = InterjectDB(get_db_path(config))
    db.connect()
    embedder = EmbeddingClient()
    engine = RecommendationEngine(db, embedder, config)
    scanner = Scanner(db, engine, embedder, config)
    ctx = {"db": db, "engine": engine, "scanner": scanner, "config": config}

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="interject_scan",
                description="Trigger a full or per-source scan for new discoveries. Scans arXiv, HN, GitHub, Anthropic docs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Scan only this source (arxiv, hackernews, github, anthropic). Omit for all.",
                        },
                        "topic": {
                            "type": "string",
                            "description": "Additional topic to search for.",
                        },
                        "hours": {
                            "type": "integer",
                            "description": "Look back this many hours (default 24).",
                            "default": 24,
                        },
                    },
                },
            ),
            Tool(
                name="interject_inbox",
                description="Get discoveries above threshold since last review. Use for session-start checks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                        "min_score": {"type": "number", "default": 0.5},
                        "source": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="interject_detail",
                description="Get full details on a specific discovery.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "discovery_id": {"type": "string"},
                    },
                    "required": ["discovery_id"],
                },
            ),
            Tool(
                name="interject_promote",
                description="Promote a discovery to a bead with briefing and optional plan. Updates the recommendation model.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "discovery_id": {"type": "string"},
                        "priority": {
                            "type": "integer",
                            "description": "Bead priority 0-4 (0=critical, 4=backlog).",
                            "default": 2,
                        },
                    },
                    "required": ["discovery_id"],
                },
            ),
            Tool(
                name="interject_dismiss",
                description="Dismiss a discovery as irrelevant. Negative signal for the recommendation model.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "discovery_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["discovery_id"],
                },
            ),
            Tool(
                name="interject_profile",
                description="View or edit the interest profile (topics, weights, learned preferences).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["view", "add_topic", "remove_topic", "reset"],
                            "default": "view",
                        },
                        "topic": {
                            "type": "string",
                            "description": "Topic for add/remove actions.",
                        },
                    },
                },
            ),
            Tool(
                name="interject_status",
                description="Health check — last scan times, queue depth, profile stats, adapter health.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="interject_search",
                description="Semantic search across all stored discoveries.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source": {"type": "string"},
                        "min_score": {"type": "number", "default": 0.0},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="interject_session_context",
                description="Get discoveries relevant to the current session topic via embedding similarity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Current session topic text to match against new discoveries.",
                        },
                    },
                },
            ),
            Tool(
                name="interject_record_query",
                description="Record a query for cross-session pattern detection and topic boosting.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "interject_scan":
                result = await _handle_scan(arguments)
            elif name == "interject_inbox":
                result = await _handle_inbox(arguments)
            elif name == "interject_detail":
                result = await _handle_detail(arguments)
            elif name == "interject_promote":
                result = await _handle_promote(arguments)
            elif name == "interject_dismiss":
                result = await _handle_dismiss(arguments)
            elif name == "interject_profile":
                result = await _handle_profile(arguments)
            elif name == "interject_status":
                result = await _handle_status(arguments)
            elif name == "interject_search":
                result = await _handle_search(arguments)
            elif name == "interject_session_context":
                result = await _handle_session_context(arguments)
            elif name == "interject_record_query":
                result = await _handle_record_query(arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            result = {"error": str(e)}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    # ── Tool handlers ───────────────────────────────────────────────

    async def _handle_scan(args: dict) -> dict:
        source = args.get("source")
        hours = args.get("hours", 24)
        since = datetime.utcnow() - timedelta(hours=hours)

        if source:
            return await scanner.scan_source(source, since)
        return await scanner.scan_all(since)

    async def _handle_inbox(args: dict) -> dict:
        limit = args.get("limit", 10)
        min_score = args.get("min_score", 0.5)
        source = args.get("source")

        discoveries = db.list_discoveries(
            status="new", source=source, min_score=min_score, limit=limit
        )
        return {
            "count": len(discoveries),
            "discoveries": [
                {
                    "id": d["id"],
                    "title": d["title"],
                    "source": d["source"],
                    "score": d["relevance_score"],
                    "tier": d["confidence_tier"],
                    "url": d["url"],
                    "discovered_at": d["discovered_at"],
                }
                for d in discoveries
            ],
        }

    async def _handle_detail(args: dict) -> dict:
        disc_id = args["discovery_id"]
        discovery = db.get_discovery(disc_id)
        if not discovery:
            return {"error": f"Discovery not found: {disc_id}"}
        # Remove embedding blob from output
        result = {k: v for k, v in discovery.items() if k != "embedding"}
        if result.get("raw_metadata"):
            try:
                result["raw_metadata"] = json.loads(result["raw_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    async def _handle_promote(args: dict) -> dict:
        disc_id = args["discovery_id"]
        priority = args.get("priority", 2)

        discovery = db.get_discovery(disc_id)
        if not discovery:
            return {"error": f"Discovery not found: {disc_id}"}

        from .outputs import OutputPipeline
        outputs = OutputPipeline()

        # Force high-tier output (bead + briefing + plan)
        result = outputs.process(discovery, "high")

        # Record promotion and learn
        if result.get("bead_id"):
            db.record_promotion(disc_id, result["bead_id"], priority)
        engine.learn_promotion(disc_id, priority)
        db.update_discovery_status(disc_id, "promoted")

        return {
            "promoted": disc_id,
            "priority": priority,
            "bead_id": result.get("bead_id"),
            "briefing": result.get("briefing_path"),
            "plan": result.get("plan_path"),
        }

    async def _handle_dismiss(args: dict) -> dict:
        disc_id = args["discovery_id"]
        reason = args.get("reason", "")

        discovery = db.get_discovery(disc_id)
        if not discovery:
            return {"error": f"Discovery not found: {disc_id}"}

        engine.learn_dismissal(disc_id)
        db.update_discovery_status(disc_id, "dismissed")

        return {"dismissed": disc_id, "reason": reason}

    async def _handle_profile(args: dict) -> dict:
        action = args.get("action", "view")

        if action == "view":
            profile = db.get_profile()
            profile.pop("topic_vector", None)  # Don't send raw bytes
            return {
                "profile": profile,
                "thresholds": {
                    "high": engine.high_threshold,
                    "medium": engine.medium_threshold,
                    "low": engine.low_threshold,
                },
            }
        elif action == "add_topic":
            topic = args.get("topic", "")
            if not topic:
                return {"error": "Topic required for add_topic action"}
            profile = db.get_profile()
            kw = profile.get("keyword_weights", {})
            kw[topic] = kw.get(topic, 0) + 1.0
            db.update_profile(keyword_weights=kw)
            # Re-embed profile with new topic
            engine.ensure_profile()
            return {"added": topic, "weight": kw[topic]}
        elif action == "remove_topic":
            topic = args.get("topic", "")
            if not topic:
                return {"error": "Topic required for remove_topic action"}
            profile = db.get_profile()
            kw = profile.get("keyword_weights", {})
            kw.pop(topic, None)
            db.update_profile(keyword_weights=kw)
            return {"removed": topic}
        elif action == "reset":
            from .embeddings import vector_to_bytes
            import numpy as np
            from .config import get_seed_topics

            seed_topics = get_seed_topics(config)
            if seed_topics:
                vectors = embedder.embed_batch(seed_topics)
                vec = vectors.mean(axis=0)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                db.update_profile(
                    topic_vector=vector_to_bytes(vec),
                    keyword_weights={},
                    source_weights={},
                )
            return {"reset": True, "seed_topics": len(seed_topics)}
        return {"error": f"Unknown action: {action}"}

    async def _handle_status(args: dict) -> dict:
        stats = db.get_stats()
        scan_stats = db.get_scan_stats()
        profile = db.get_profile()
        profile.pop("topic_vector", None)

        return {
            "discoveries": stats,
            "scans": scan_stats,
            "profile": {
                "keyword_count": len(profile.get("keyword_weights", {})),
                "source_weights": profile.get("source_weights", {}),
            },
            "thresholds": {
                "high": engine.high_threshold,
                "medium": engine.medium_threshold,
                "low": engine.low_threshold,
            },
        }

    async def _handle_search(args: dict) -> dict:
        query = args["query"]
        source = args.get("source")
        min_score = args.get("min_score", 0.0)
        limit = args.get("limit", 20)

        # Embed query
        from .embeddings import bytes_to_vector
        query_vec = embedder.embed(query)

        # Get all discoveries and rank by similarity
        discoveries = db.list_discoveries(
            source=source, min_score=min_score, limit=200
        )

        scored = []
        for d in discoveries:
            if d.get("embedding"):
                d_vec = bytes_to_vector(d["embedding"])
                sim = float(query_vec @ d_vec)
                scored.append((sim, d))

        scored.sort(key=lambda x: x[0], reverse=True)

        return {
            "query": query,
            "count": min(limit, len(scored)),
            "results": [
                {
                    "id": d["id"],
                    "title": d["title"],
                    "source": d["source"],
                    "relevance_score": d["relevance_score"],
                    "search_similarity": round(sim, 3),
                    "url": d["url"],
                    "tier": d["confidence_tier"],
                }
                for sim, d in scored[:limit]
            ],
        }

    async def _handle_session_context(args: dict) -> str:
        topic = args.get("topic", "")
        if not topic:
            return "No topic provided. Use interject_inbox for global top discoveries."

        from .embeddings import bytes_to_vector

        embedding = embedder.embed(topic)
        discoveries = db.list_discoveries(status="new", limit=100)
        scored = []
        for discovery in discoveries:
            if discovery.get("embedding"):
                vec = bytes_to_vector(discovery["embedding"])
                sim = embedder.cosine_similarity(embedding, vec)
                if sim > 0.3:
                    scored.append((sim, discovery))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return "No relevant discoveries for this topic."

        lines = [f"Discoveries relevant to: {topic[:80]}"]
        for sim, discovery in scored[:5]:
            lines.append(
                f"- [{discovery['source']}] {discovery['title']} (relevance: {sim:.2f})"
            )
            lines.append(f"  {discovery.get('url', '')}")
        return "\n".join(lines)

    async def _handle_record_query(args: dict) -> str:
        query = args["query"]
        session_id = args.get("session_id", "")

        from .feedback import FeedbackCollector

        collector = FeedbackCollector(db)
        collector.record_query(query, session_id=session_id)
        return f"Recorded query: {query[:80]}"

    return server, ctx


async def run_server() -> None:
    """Run the MCP server on stdio."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    server, ctx = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

    # Cleanup
    ctx["db"].close()


def main() -> None:
    """Entry point for interject-mcp command."""
    asyncio.run(run_server())
