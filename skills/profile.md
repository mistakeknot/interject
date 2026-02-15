---
name: profile
description: View and tune your Interject interest profile
user_invocable: true
---

# /interject:profile

View and manage your Interject interest profile — the recommendation engine's learned preferences.

## Usage

When the user invokes `/interject:profile`, use the `interject_profile` MCP tool.

## Behavior

1. Call `interject_profile` with `action: view`
2. Present the profile:
   - Keyword weights (topics and their learned importance)
   - Source weights (which sources produce the best recommendations)
   - Current adaptive thresholds (high/medium/low confidence tiers)
3. Ask if the user wants to:
   - **Add topic**: Call `interject_profile` with `action: add_topic` and `topic`
   - **Remove topic**: Call `interject_profile` with `action: remove_topic` and `topic`
   - **Reset**: Call `interject_profile` with `action: reset` to re-seed from defaults
