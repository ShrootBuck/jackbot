#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

has_resume=false
for arg in "$@"; do
  case "${arg}" in
    --resume|--resume=*)
      has_resume=true
      ;;
  esac
done

resume_args=()
if [[ "${has_resume}" == false ]]; then
  for candidate in \
    checkpoints/champions/promoted.pt \
    checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt \
    checkpoints/champions/pzrnunoa_update3000.pt; do
    if [[ -f "${candidate}" ]]; then
      resume_args=(--resume "${candidate}")
      break
    fi
  done
  if [[ ${#resume_args[@]} -eq 0 ]]; then
    echo "No god-run resume checkpoint found. Use the existing W&B download path or pass --resume PATH." >&2
    exit 1
  fi
fi

uv run jackbot-train \
  --profile genius \
  "${resume_args[@]}" \
  "$@"
