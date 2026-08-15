#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

JACKBOT_TABLE_CHECKPOINT="${JACKBOT_TABLE_CHECKPOINT:-checkpoints/champions/promoted.pt}"
if [[ ! -f "${JACKBOT_TABLE_CHECKPOINT}" ]]; then
  echo "Promoted table checkpoint not found: ${JACKBOT_TABLE_CHECKPOINT}" >&2
  exit 1
fi

exec uv run jackbot-advisor "${JACKBOT_TABLE_CHECKPOINT}" --preset god "$@"
