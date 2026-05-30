#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cargo test
cargo clippy --all-targets -- -D warnings
uv run pytest
