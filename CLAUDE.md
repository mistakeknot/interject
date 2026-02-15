# Interject

Ambient discovery and research engine. Scans arXiv, Hacker News, GitHub, and Anthropic docs for new capabilities, workflows, and tools. Creates beads with briefings and draft plans based on a learned recommendation model.

## MCP Server

Python MCP server at `src/interject/server.py`. Run with `uv run interject-mcp`.

## Key Files

- `src/interject/server.py` — MCP server entrypoint (FastMCP)
- `src/interject/db.py` — SQLite schema and queries
- `src/interject/engine.py` — Recommendation engine (score, rank, learn)
- `src/interject/embeddings.py` — Shared embedding client (tldr-swinton)
- `src/interject/scanner.py` — Daemon loop and adapter orchestration
- `src/interject/gaps.py` — Capability gap detection
- `src/interject/outputs.py` — Bead creation, briefing/plan generation
- `src/interject/config.py` — YAML config loader
- `sources/` — Pluggable source adapters (arXiv, HN, GitHub, Anthropic)

## Plugin Publishing

Use `/interpub:release <version>` or `scripts/bump-version.sh <version>`.
