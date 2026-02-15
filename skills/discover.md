---
name: discover
description: Deep dive on a specific topic across all sources
user_invocable: true
---

# /interject:discover

Deep dive research on a specific topic across all Interject sources.

## Usage

When the user invokes `/interject:discover <topic>`, use the `interject_scan` tool with the topic argument.

**Arguments:**
- First argument is the topic to research (required)

## Behavior

1. Call `interject_scan` with `topic` set to the user's query and `hours` set to 168 (7 days) for deeper reach
2. Then call `interject_search` with the same topic as the query to find any existing discoveries that match
3. Present a combined view:
   - New discoveries from the scan
   - Existing relevant discoveries from the database
   - Recommended actions (promote, dismiss, or investigate further)
4. If the user wants to promote any discovery, use `interject_promote`
