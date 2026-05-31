#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${1:-checkpoints/jackbot_best.pt}"
shift || true

uv run jackbot-gauntlet "${CHECKPOINT}" --games "${JACKBOT_GAUNTLET_GAMES:-512}" "$@"
