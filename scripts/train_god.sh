#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

uv run jackbot-train \
  --wandb-mode online \
  --league \
  --eval-games 256 \
  --updates 3000 \
  "$@"
