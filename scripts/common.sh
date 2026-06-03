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

# PyO3 needs the same Python that uv/maturin uses. Without this, `cargo test`
# can accidentally find Xcode's Python 3.9 framework on macOS and then fail to
# link against a library Apple does not ship.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export PYO3_PYTHON="${PYO3_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
fi

echo "repo: ${REPO_ROOT}"
echo "RUSTFLAGS=${RUSTFLAGS}"
echo "UV_PYTHON=${UV_PYTHON}"
if [[ -n "${PYO3_PYTHON:-}" ]]; then
  echo "PYO3_PYTHON=${PYO3_PYTHON}"
else
  echo "PYO3_PYTHON=(unset; run ./scripts/bootstrap.sh first if cargo links the wrong Python)"
fi
