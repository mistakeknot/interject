#!/usr/bin/env bash
# Thin wrapper around Interverse's shared interbump.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")/scripts/interbump.sh" "$@"
