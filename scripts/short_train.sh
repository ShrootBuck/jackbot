#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

UPDATES="${JACKBOT_UPDATES:-10}"
NUM_ENVS="${JACKBOT_NUM_ENVS:-64}"
ROLLOUT_LEN="${JACKBOT_ROLLOUT_LEN:-16}"

uv run jackbot-train \
  --updates "${UPDATES}" \
  --num-envs "${NUM_ENVS}" \
  --rollout-len "${ROLLOUT_LEN}" \
  --no-wandb \
  "$@"
