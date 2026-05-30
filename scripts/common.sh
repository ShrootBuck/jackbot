#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Compile Rust for the CPU running this command. On Apple Silicon this lets LLVM
# use the actual local chip instead of a generic aarch64 baseline.
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

# Keep uv away from the system Python. PyTorch support is the constraint here.
export UV_PYTHON="${UV_PYTHON:-3.12}"

# Helps PyTorch survive occasional MPS operator gaps instead of hard-crashing.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

echo "repo: ${REPO_ROOT}"
echo "RUSTFLAGS=${RUSTFLAGS}"
echo "UV_PYTHON=${UV_PYTHON}"
