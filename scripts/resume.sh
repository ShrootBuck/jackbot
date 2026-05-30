#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${1:-checkpoints/jackbot_latest.pt}"
shift || true

uv run jackbot-train \
  --resume "${CHECKPOINT}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  "$@"
