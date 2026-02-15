# Interject

Ambient discovery and research engine. Scans arXiv, Hacker News, GitHub, Anthropic docs, and Exa for new capabilities, workflows, and tools. Creates beads with briefings and brainstorm docs based on a learned recommendation model with closed-loop feedback.

## MCP Server

Python MCP server at `src/interject/server.py`. Run with `uv run interject-mcp`.

## Key Files

- `src/interject/server.py` — MCP server entrypoint (FastMCP, 10 tools)
- `src/interject/db.py` — SQLite schema v2 (discoveries, promotions, feedback_signals, query_log)
- `src/interject/engine.py` — Recommendation engine (score, rank, learn, feedback-driven weights)
- `src/interject/embeddings.py` — Re-exports from intersearch shared library
- `src/interject/scanner.py` — One-shot scanner with adapter orchestration
- `src/interject/feedback.py` — Bead lifecycle tracking, query log, conversion rates
- `src/interject/awareness.py` — Internal state reader (plans, beads, brainstorms)
- `src/interject/gaps.py` — Capability gap detection
- `src/interject/outputs.py` — Tiered output: brainstorm docs (high), briefings (medium), digest
- `src/interject/config.py` — YAML config loader
- `src/interject/sources/` — Pluggable source adapters (arXiv, HN, GitHub, Anthropic, Exa)

## Dependencies

- `intersearch` — shared Exa and embedding infrastructure (path dependency)

## Plugin Publishing

Use `/interpub:release <version>` or `scripts/bump-version.sh <version>`.
