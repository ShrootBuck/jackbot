#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/runs/wandb}"
mkdir -p "${WANDB_DIR}"

uv run jackbot-screen "$@"
