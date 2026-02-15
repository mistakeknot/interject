---
name: inbox
description: Review pending discoveries — promote or dismiss interactively
user_invocable: true
---

# /interject:inbox

Review your Interject discovery inbox — pending items above the confidence threshold.

## Usage

When the user invokes `/interject:inbox`, use the `interject_inbox` MCP tool.

## Behavior

1. Call `interject_inbox` with default parameters (or specify `limit` and `min_score` if the user provides them)
2. Present discoveries in a numbered list with:
   - Title
   - Source (arxiv/hackernews/github/anthropic/exa)
   - Relevance score
   - One-line summary
   - URL
3. Ask the user which items to act on:
   - **Promote**: Call `interject_promote` with the discovery_id and ask for priority (P0-P4)
   - **Dismiss**: Call `interject_dismiss` with the discovery_id
   - **Detail**: Call `interject_detail` to see more about a specific item
   - **Skip**: Leave for later
4. After processing, report how many items remain in the inbox
