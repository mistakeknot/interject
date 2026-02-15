#!/usr/bin/env bash
# Interject session hook — lightweight inbox check at session start.
# Only surfaces high-confidence discoveries above the learned threshold.
# Silent when nothing important.

set -euo pipefail

# This hook is called by Claude Code at session start.
# It checks the Interject inbox for high-priority discoveries.

# Read session info from stdin (Claude Code hook protocol)
SESSION_JSON=$(cat)

# Check if interject database exists
INTERJECT_DB="$HOME/.interject/interject.db"
if [[ ! -f "$INTERJECT_DB" ]]; then
    exit 0
fi

# Query for high-confidence new discoveries
# Uses sqlite3 directly for speed (no MCP server needed)
COUNT=$(sqlite3 "$INTERJECT_DB" "SELECT COUNT(*) FROM discoveries WHERE status = 'new' AND relevance_score >= 0.8" 2>/dev/null || echo "0")

if [[ "$COUNT" -gt 0 ]]; then
    # Get top discoveries
    ITEMS=$(sqlite3 -separator '|' "$INTERJECT_DB" \
        "SELECT title, source, relevance_score FROM discoveries WHERE status = 'new' AND relevance_score >= 0.8 ORDER BY relevance_score DESC LIMIT 5" \
        2>/dev/null || echo "")

    if [[ -n "$ITEMS" ]]; then
        echo "Interject: $COUNT high-relevance discoveries since last session"
        while IFS='|' read -r title source score; do
            echo "  - [$source] $title (score: $score)"
        done <<< "$ITEMS"
        echo "Run /interject:inbox to review"
    fi
fi
