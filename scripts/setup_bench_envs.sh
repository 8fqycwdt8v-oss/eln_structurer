#!/usr/bin/env bash
#
# Set up separate Python environments for the Phase-5 benchmark comparators.
#
# paragraph2actions and openchemie pin ancient torch versions (<1.5 for p2a)
# that conflict with the core eln_structurer venv (Python 3.11 + modern stack).
# We solve this by giving each comparator its own venv with its own Python
# version, then invoking them from the benchmark CLI via an isolated PATH.
#
# This script is documentation-as-code: it currently fails fast because the
# upstream dependency pins are incompatible with available wheels on modern
# Linux. Use it as a starting point and adapt as the upstream packages catch up.
#
# Usage:
#   bash scripts/setup_bench_envs.sh paragraph2actions
#   bash scripts/setup_bench_envs.sh openchemie
#   bash scripts/setup_bench_envs.sh all

set -euo pipefail

TARGET="${1:-all}"

setup_paragraph2actions() {
  echo "[setup] paragraph2actions in .venv-p2a/"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required; install from https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  # paragraph2actions requires Python 3.8 and torch<1.5.
  uv venv --python 3.8 .venv-p2a
  # shellcheck disable=SC1091
  source .venv-p2a/bin/activate
  uv pip install paragraph2actions
  echo "[setup] To run benchmarks with paragraph2actions:"
  echo "  source .venv-p2a/bin/activate"
  echo "  uv pip install -e ."
  echo "  uv run eln-structurer bench --adapter paragraph2actions"
  deactivate
}

setup_openchemie() {
  echo "[setup] openchemie in .venv-oce/"
  uv venv --python 3.10 .venv-oce
  # shellcheck disable=SC1091
  source .venv-oce/bin/activate
  # OpenChemIE is published on GitHub; install from source.
  uv pip install git+https://github.com/CrystalEye42/OpenChemIE.git
  echo "[setup] OpenChemIE installed. First-run model downloads ~1 GB."
  deactivate
}

case "$TARGET" in
  paragraph2actions) setup_paragraph2actions ;;
  openchemie)        setup_openchemie ;;
  all)
    setup_paragraph2actions || echo "[warn] paragraph2actions setup failed"
    setup_openchemie         || echo "[warn] openchemie setup failed"
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Usage: $0 {paragraph2actions|openchemie|all}" >&2
    exit 1
    ;;
esac
