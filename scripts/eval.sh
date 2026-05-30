#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${1:-checkpoints/jackbot_best.pt}"
shift || true

uv run jackbot-eval "${CHECKPOINT}" --games "${JACKBOT_EVAL_GAMES:-64}" "$@"
