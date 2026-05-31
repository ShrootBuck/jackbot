#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

uv run jackbot-train \
  --wandb-mode "${WANDB_MODE:-online}" \
  --league \
  --eval-games "${JACKBOT_EVAL_GAMES:-256}" \
  --updates "${JACKBOT_GOD_UPDATES:-3000}" \
  "$@"
