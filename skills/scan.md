---
name: scan
description: Run a full scan across all sources or a specific source for new discoveries
user_invocable: true
---

# /interject:scan

Run an Interject scan to discover new capabilities, tools, papers, and discussions.

## Usage

When the user invokes `/interject:scan`, use the `interject_scan` MCP tool.

**Arguments:**
- `source` (optional) — scan only one source: `arxiv`, `hackernews`, `github`, `anthropic`, `exa`
- `topic` (optional) — add a specific topic to search for
- `hours` (optional) — look back this many hours (default 24)

## Behavior

1. Call `interject_scan` with the provided arguments
2. Report the results:
   - How many items found per source
   - How many scored above threshold
   - Any high-confidence discoveries worth immediate attention
3. If high-confidence discoveries exist, summarize them with titles and one-line descriptions
4. Suggest `/interject:inbox` if there are items to review
