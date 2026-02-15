"""Scanner/daemon — orchestrates source adapters, scoring, and output pipeline."""

from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import get_daemon_config, get_seed_topics, get_source_config, load_config
from .db import InterjectDB
from .embeddings import EmbeddingClient, vector_to_bytes
from .engine import RecommendationEngine
from .gaps import GapDetector
from .outputs import OutputPipeline

logger = logging.getLogger(__name__)

# Map adapter module names to their classes
ADAPTER_CLASSES = {
    "arxiv": ("interject.sources.arxiv", "ArxivAdapter"),
    "hackernews": ("interject.sources.hackernews", "HackerNewsAdapter"),
    "github": ("interject.sources.github", "GitHubAdapter"),
    "anthropic": ("interject.sources.anthropic", "AnthropicAdapter"),
    "exa": ("interject.sources.exa", "ExaAdapter"),
}


class Scanner:
    """Orchestrates source scanning, scoring, and output generation."""

    def __init__(
        self,
        db: InterjectDB,
        engine: RecommendationEngine,
        embedder: EmbeddingClient,
        config: dict[str, Any],
    ):
        self.db = db
        self.engine = engine
        self.embedder = embedder
        self.config = config
        self.outputs = OutputPipeline()
        self.gap_detector = GapDetector()
        self._daemon_cfg = get_daemon_config(config)

    def _load_adapters(self) -> list[Any]:
        """Load enabled source adapters."""
        adapters = []
        sources_dir = Path(__file__).parent.parent.parent / "sources"

        for name, (module_path, class_name) in ADAPTER_CLASSES.items():
            source_cfg = get_source_config(self.config, name)
            if not source_cfg.get("enabled", True):
                logger.info("Adapter %s disabled, skipping", name)
                continue

            try:
                # Import relative to the sources package
                mod = importlib.import_module(f"interject.sources.{name}")
                adapter_cls = getattr(mod, class_name)
                adapter = adapter_cls(config=source_cfg)
                adapters.append(adapter)
                logger.info("Loaded adapter: %s", name)
            except Exception as e:
                logger.warning("Failed to load adapter %s: %s", name, e)

        return adapters

    async def scan_all(self, since: datetime | None = None) -> dict[str, Any]:
        """Run a full scan across all enabled adapters.

        Returns summary stats.
        """
        if since is None:
            # Default: look back 24 hours, or since last scan
            since = datetime.utcnow() - timedelta(hours=24)

        adapters = self._load_adapters()
        if not adapters:
            logger.warning("No adapters loaded")
            return {"error": "No adapters available"}

        # Detect gaps for bonus scoring
        gaps = self.gap_detector.scan()
        gap_topics = [g["category"] for g in gaps]

        # Get topics (seed + gap-derived)
        topics = get_seed_topics(self.config) + gap_topics

        # Ensure engine has a profile
        self.engine.ensure_profile()

        stagger = self._daemon_cfg.get("stagger_seconds", 30)
        results = {"adapters": {}, "total_found": 0, "total_scored": 0}

        for adapter in adapters:
            try:
                logger.info("Scanning %s since %s", adapter.name, since.isoformat())
                raw_discoveries = await adapter.fetch(since, topics)
                logger.info("%s returned %d items", adapter.name, len(raw_discoveries))

                scored = 0
                for raw in raw_discoveries:
                    # Enrich
                    enriched = await adapter.enrich(raw)

                    # Generate ID
                    disc_id = f"ij-{adapter.name}-{raw.source_id[:30]}"

                    # Check for duplicate
                    existing = self.db.get_discovery(disc_id)
                    if existing:
                        continue

                    # Embed
                    embed_text = enriched.full_text or f"{enriched.title}. {enriched.summary}"
                    embedding = self.engine.get_embedding(embed_text)

                    # Score
                    gap_detected = self.gap_detector.is_gap_filling(embed_text, gaps)
                    score, tier = self.engine.score(
                        embed_text,
                        source=adapter.name,
                        discovered_at=raw.discovered_at,
                        gap_detected=gap_detected,
                    )

                    if tier == "discard":
                        continue

                    # Insert into db
                    self.db.insert_discovery(
                        id=disc_id,
                        source=adapter.name,
                        source_id=raw.source_id,
                        title=raw.title,
                        summary=raw.summary,
                        url=raw.url,
                        raw_metadata=raw.metadata,
                        embedding=embedding,
                        relevance_score=score,
                        confidence_tier=tier,
                    )

                    # Process through output pipeline
                    output_result = self.outputs.process(
                        self.db.get_discovery(disc_id), tier
                    )

                    # Record promotion if bead was created
                    if output_result.get("bead_id"):
                        priority = 2 if tier == "high" else 3
                        self.db.record_promotion(
                            disc_id, output_result["bead_id"], priority
                        )

                    scored += 1

                # Log scan
                self.db.log_scan(adapter.name, len(raw_discoveries), scored)
                results["adapters"][adapter.name] = {
                    "found": len(raw_discoveries),
                    "scored": scored,
                }
                results["total_found"] += len(raw_discoveries)
                results["total_scored"] += scored

            except Exception as e:
                logger.error("Error scanning %s: %s", adapter.name, e)
                results["adapters"][adapter.name] = {"error": str(e)}

            # Stagger between adapters
            if stagger > 0 and adapter != adapters[-1]:
                await asyncio.sleep(stagger)

        # Apply decay to old discoveries
        decayed = self.db.apply_decay(
            rate=self.engine._engine_cfg.get("decay_rate", 0.95),
            floor=self.engine._engine_cfg.get("decay_floor", 0.1),
        )
        results["decayed"] = decayed

        # Adapt thresholds based on accumulated feedback
        self.engine.adapt_thresholds()

        return results

    async def scan_source(
        self, source_name: str, since: datetime | None = None
    ) -> dict[str, Any]:
        """Scan a single source adapter."""
        if source_name not in ADAPTER_CLASSES:
            return {"error": f"Unknown source: {source_name}"}

        module_path, class_name = ADAPTER_CLASSES[source_name]
        source_cfg = get_source_config(self.config, source_name)

        try:
            mod = importlib.import_module(f"sources.{source_name}")
            adapter_cls = getattr(mod, class_name)
            adapter = adapter_cls(config=source_cfg)
        except Exception as e:
            return {"error": f"Failed to load adapter: {e}"}

        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)

        topics = get_seed_topics(self.config)
        self.engine.ensure_profile()

        try:
            raw = await adapter.fetch(since, topics)
            scored = 0
            for item in raw:
                enriched = await adapter.enrich(item)
                disc_id = f"ij-{adapter.name}-{item.source_id[:30]}"
                if self.db.get_discovery(disc_id):
                    continue

                embed_text = enriched.full_text or f"{enriched.title}. {enriched.summary}"
                embedding = self.engine.get_embedding(embed_text)
                score, tier = self.engine.score(embed_text, source=adapter.name)

                if tier != "discard":
                    self.db.insert_discovery(
                        id=disc_id,
                        source=adapter.name,
                        source_id=item.source_id,
                        title=item.title,
                        summary=item.summary,
                        url=item.url,
                        raw_metadata=item.metadata,
                        embedding=embedding,
                        relevance_score=score,
                        confidence_tier=tier,
                    )
                    scored += 1

            self.db.log_scan(adapter.name, len(raw), scored)
            return {"source": source_name, "found": len(raw), "scored": scored}
        except Exception as e:
            return {"error": str(e)}


async def run_once(config: dict | None = None) -> dict:
    """Run a single scan across all sources and exit."""
    if config is None:
        config = load_config()

    db = InterjectDB(config.get("db_path", "~/.interject/interject.db"))
    db.connect()
    embedder = EmbeddingClient()
    engine = RecommendationEngine(db, embedder, config)
    scanner = Scanner(db, engine, embedder, config)

    logger.info("Interject scan starting (one-shot)")
    try:
        results = await scanner.scan_all()
        logger.info(
            "Scan complete: %d found, %d scored",
            results.get("total_found", 0),
            results.get("total_scored", 0),
        )
        return results
    except Exception as e:
        logger.error("Scan failed: %s", e)
        return {"error": str(e)}
    finally:
        db.close()


def main() -> None:
    """CLI entry point — runs a single scan and exits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(run_once())
