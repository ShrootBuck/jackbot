#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

uv run jackbot-train \
  --wandb-mode online \
  --device auto \
  --checkpoint-dir checkpoints \
  --latest-name jackbot_latest.pt \
  --best-name jackbot_best.pt \
  --updates 3000 \
  --num-envs 1024 \
  --rollout-len 64 \
  --ppo-epochs 3 \
  --minibatch-size 4096 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip 0.2 \
  --lr 0.0003 \
  --hidden-size 2048 \
  --entropy-coef 0.01 \
  --value-coef 0.5 \
  --belief-coef 0.05 \
  --max-grad-norm 1.0 \
  --shaping-start 1.0 \
  --shaping-decay-fraction 0.40 \
  --lr-anneal \
  --checkpoint-interval-updates 25 \
  --milestone-interval-updates 50 \
  --checkpoint-interval-seconds 900 \
  --eval-interval-updates 50 \
  --eval-games 256 \
  --eval-num-envs 32 \
  --eval-max-steps-per-game 2000 \
  --eval-opponent random \
  --eval-opponent heuristic \
  --league \
  --league-baselines heuristic,random \
  --league-auto-checkpoints \
  --league-max-checkpoints 8 \
  --league-self-play-weight 1.0 \
  --league-baseline-weight 1.0 \
  --league-checkpoint-weight 2.0 \
  --league-refresh-interval-updates 25 \
  --deterministic-league-opponents \
  "$@"
