#!/usr/bin/env bash
# Prerequisites for running benchmark_pi0_libero_rocm.py on a fresh machine (AMD ROCm).
#
# Usage (from the openpi repo root):
#   chmod +x setup_mi300x.sh
#   ./setup_mi300x.sh
#
# Environment (all optional):
#   PREPARE_PYTORCH_CHECKPOINT=1   Download gs://openpi-assets/checkpoints/pi0_libero and convert
#                                  to a PyTorch checkpoint (large download; run once).
#   LIBERO_CONFIG=pi0_libero       Config name for optional checkpoint prep (default: pi0_libero).
#   ROCM_PYTORCH_INDEX_URL=...     PyTorch wheel index (default: rocm7.2).
#   TORCH_VERSION=2.7.1            Must match pyproject.toml torch pin.
#   GIT_LFS_SKIP_SMUDGE=1          Passed to uv sync (default: 1) to avoid LFS pulls during install.
#   SKIP_SYSTEM_CHECKS=1           Do not warn if rocm-smi or /dev/kfd is missing.
#
# You must install the ROCm driver/stack on the host separately; see:
#   https://rocm.docs.amd.com/en/latest/deploy/linux/index.html

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"

ROCM_PYTORCH_INDEX_URL="${ROCM_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
PREPARE_PYTORCH_CHECKPOINT="${PREPARE_PYTORCH_CHECKPOINT:-0}"
LIBERO_CONFIG="${LIBERO_CONFIG:-pi0_libero}"
SKIP_SYSTEM_CHECKS="${SKIP_SYSTEM_CHECKS:-0}"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

log() {
  echo "[setup] $*"
}

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  die "Expected openpi at REPO_ROOT=$REPO_ROOT (pyproject.toml not found)."
fi

cd "$REPO_ROOT"

if [[ "$SKIP_SYSTEM_CHECKS" != "1" ]]; then
  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "[WARN] rocm-smi not found in PATH. Install ROCm and ensure GPUs are visible before benchmarking."
  elif ! rocm-smi >/dev/null 2>&1; then
    echo "[WARN] rocm-smi failed. Check ROCm installation and GPU drivers."
  fi
  if [[ ! -e /dev/kfd ]]; then
    echo "[WARN] /dev/kfd missing — AMD GPU access may not work (user may need render group membership)."
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  die "git is required."
fi

git lfs install

if ! command -v uv >/dev/null 2>&1; then
  log "uv not found; installing via https://astral.sh/uv/install.sh"
  curl -Lsf https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.local/bin"
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$UV_BIN:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; add ~/.local/bin to PATH and re-run."
fi

log "Syncing Python environment (GIT_LFS_SKIP_SMUDGE=$GIT_LFS_SKIP_SMUDGE) ..."
GIT_LFS_SKIP_SMUDGE="$GIT_LFS_SKIP_SMUDGE" uv sync

VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
[[ -x "$VENV_PY" ]] || die ".venv/bin/python missing after uv sync."

log "Installing PyTorch + torchvision (ROCm) into .venv — do not use 'uv run' here (lockfile pins CUDA torch)."
"$VENV_PIP" install "torch==${TORCH_VERSION}" torchvision \
  --index-url "$ROCM_PYTORCH_INDEX_URL" \
  --force-reinstall

log "Verifying ROCm torch ..."
"$VENV_PY" - <<'PY'
import sys
import torch
hip = getattr(torch.version, "hip", None)
if hip is None:
    print("[ERROR] torch is not a ROCm build (torch.version.hip is None).", file=sys.stderr)
    print("        Re-run setup or install torch from the ROCm index into .venv only.", file=sys.stderr)
    sys.exit(1)
print("  torch:", torch.__version__)
print("  HIP:  ", hip)
print("  GPUs: ", torch.cuda.device_count())
PY

if [[ "$PREPARE_PYTORCH_CHECKPOINT" == "1" ]]; then
  log "PREPARE_PYTORCH_CHECKPOINT=1: downloading JAX checkpoint and converting to PyTorch (this can take a long time) ..."
  CKPT_JAX="$(
    "$VENV_PY" -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/${LIBERO_CONFIG}')
print(p)
"
  )"
  CKPT_PT="${CKPT_JAX}_pytorch"
  if [[ -f "$CKPT_PT/model.safetensors" ]]; then
    log "PyTorch checkpoint already present: $CKPT_PT (skipping convert)."
  else
    "$VENV_PY" "$REPO_ROOT/examples/convert_jax_model_to_pytorch.py" \
      --checkpoint-dir "$CKPT_JAX" \
      --config-name "$LIBERO_CONFIG" \
      --output-path "$CKPT_PT"
  fi
  log "Done. Run inference benchmark with:"
  echo "  $VENV_PY $REPO_ROOT/benchmark_pi0_libero_rocm.py --config $LIBERO_CONFIG --checkpoint-dir $CKPT_PT --quick"
else
  log "Skipping checkpoint download/convert (set PREPARE_PYTORCH_CHECKPOINT=1 to automate)."
  log "For --backend pytorch you need model.safetensors (convert JAX checkpoint once) or pass --backend jax."
  echo "  Example (after first successful GCS download on demand):"
  echo "    $VENV_PY benchmark_pi0_libero_rocm.py --config pi0_libero --quick"
  echo "  Or with a local converted tree:"
  echo "    $VENV_PY benchmark_pi0_libero_rocm.py --config pi0_libero --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/${LIBERO_CONFIG}_pytorch --quick"
fi

log "Optional: simulation mode (--sim) needs LIBERO + MuJoCo and git submodules; see examples/libero/README.md"
log "  git submodule update --init --recursive"
log "Finish."
