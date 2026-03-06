---
name: triage
description: Batch review pending_triage beads — promote or dismiss interject discoveries
user_invocable: true
---

# /interject:triage

Batch review beads created by interject with `pending_triage` label. Promote worthy items (raise priority to P2) or dismiss stale ones.

## Usage

When the user invokes `/interject:triage`, follow the behavior below.

## Arguments

- `--limit=N` — Maximum items per batch (default: 5, max: 20)
- `--source=NAME` — Filter to a specific source (arxiv, github, hackernews, etc.)

## Behavior

1. **List pending items:**
   Run via Bash: `BEADS_DIR=.beads bd list --labels=pending_triage --status=open --json`
   Parse the JSON output to get bead IDs, titles, priorities, and descriptions.

2. **Enrich with context:**
   For each bead, extract from the description:
   - Source and URL (format: `Source: <source> | <url>`)
   - Relevance score (format: `Relevance score: <score>`)
   - Summary (text between URL line and score line)

3. **Present batch via AskUserQuestion:**
   Present up to `limit` items. For each item show:
   - Title (without `[interject]` prefix)
   - Source and URL
   - Relevance score
   - One-line summary

   Use multiSelect with options:
   - Individual items labeled "Promote: <title>" for each item
   - "Dismiss all shown" — Close all items in this batch
   - "Skip batch" — Leave all for later

4. **Process selections:**

   For promoted items:
   ```bash
   BEADS_DIR=.beads bd update <id> --priority=2
   BEADS_DIR=.beads bd update <id> --remove-label=pending_triage
   ```
   If a kernel discovery ID is referenced in the bead:
   ```bash
   ic discovery feedback <kernel_id> --signal=promote --actor=human
   ```

   For dismissed items:
   ```bash
   BEADS_DIR=.beads bd close <id> --reason="triage-dismissed"
   ```
   If a kernel discovery ID is referenced:
   ```bash
   ic discovery feedback <kernel_id> --signal=dismiss --actor=human
   ```

5. **Report summary:**
   ```
   Triage complete: N promoted, M dismissed, K skipped
   Remaining pending: <count from bd list>
   ```

6. **Loop:** If items remain and user didn't choose "Skip batch", present the next batch.
