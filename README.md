# interject

Ambient discovery and research engine for Claude Code.

## What This Does

interject continuously scans arXiv, Hacker News, GitHub, Anthropic docs, and Exa for new capabilities, tools, and research relevant to your projects. It builds a learned interest profile from your promote/dismiss signals and gets better at surfacing relevant discoveries over time.

When something interesting turns up, it creates a briefing with context about why it matters for your specific projects — not just "here's a new paper" but "here's a new paper and here's how it relates to the embedding infrastructure you're building in intersearch."

High-relevance discoveries get promoted to brainstorm docs and beads. Medium relevance gets briefings. Everything else goes in the digest. The confidence tiering means you're not drowning in noise.

## Installation

```bash
/plugin install interject
```

Requires `intersearch` as a dependency (shared embedding infrastructure).

## Usage

Scan for new discoveries:

```
/interject:scan
```

Check your inbox:

```
/interject:inbox
```

View and manage your interest profile:

```
/interject:profile
```

## Architecture

```
src/interject/       Python library with pluggable source adapters
skills/              scan, discover, inbox, profile, status
server/              MCP server (Python/FastMCP, launched via uv run)
```

SQLite database tracks discoveries, promotions, feedback signals, and query history. The recommendation engine uses scored ranking with feedback-driven weight adjustments.
