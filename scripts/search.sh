#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${1:-checkpoints/jackbot_best.pt}"
shift || true

uv run jackbot-search "${CHECKPOINT}" --rollouts "${JACKBOT_SEARCH_ROLLOUTS:-64}" "$@"
