#!/usr/bin/env bash
set -euo pipefail

# Source the shared native/Python setup without dumping four boilerplate lines
# into every one-shot agent command.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh" >/dev/null

JACKBOT_AGENT_CHECKPOINT="${JACKBOT_AGENT_CHECKPOINT:-checkpoints/champions/promoted.pt}"
JACKBOT_AGENT_SESSION="${JACKBOT_AGENT_SESSION:-.jackbot-agent.json}"
JACKBOT_AGENT_PRESET="${JACKBOT_AGENT_PRESET:-god}"

if [[ ! -f "${JACKBOT_AGENT_CHECKPOINT}" ]]; then
  echo "Promoted table checkpoint not found: ${JACKBOT_AGENT_CHECKPOINT}" >&2
  exit 1
fi

exec uv run python -m jackbot.training.agent_play \
  --checkpoint "${JACKBOT_AGENT_CHECKPOINT}" \
  --session "${JACKBOT_AGENT_SESSION}" \
  --preset "${JACKBOT_AGENT_PRESET}" \
  "$@"
