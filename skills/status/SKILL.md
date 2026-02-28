---
name: status
description: Dashboard of scan health, recommendation stats, and adapter status
user_invocable: true
---

# /interject:status

Show the Interject system health dashboard.

## Usage

When the user invokes `/interject:status`, use the `interject_status` MCP tool.

## Behavior

1. Call `interject_status`
2. Present a dashboard showing:
   - **Discovery stats**: total, by status (new/reviewed/promoted/dismissed/decayed), by source
   - **Scan history**: last scan time per adapter, items found, items above threshold
   - **Profile health**: keyword count, source weights, current thresholds
   - **Promotion rate**: total promotions vs total discoveries (affects adaptive thresholds)
3. Flag any issues:
   - Adapters that haven't scanned recently
   - Very low or very high promotion rates (may indicate threshold miscalibration)
   - Empty profile (recommendation quality will be poor)
