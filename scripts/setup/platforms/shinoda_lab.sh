# Sourced by scripts/setup/setup.sh. SHINODA-LAB is the platform previously
# called lab. The virtualenv path is .venvs/se, shared with TSUBAME.
platform_prepare() {
  export AUDIO_DB_PLATFORM=shinoda_lab
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
  fi
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is not on PATH. Install Rust 1.76 or newer before the Rust FFT build." >&2
    export AUDIO_DB_CARGO=0
  else
    export AUDIO_DB_CARGO=1
  fi
  # The workstation CUDA toolkit under /usr/local/cuda is not used.
  # nvcc for Numba comes from the nvidia-cuda-nvcc wheel inside the role env.
  export AUDIO_DB_CUDA_FROM=wheel
  export AUDIO_DB_HDF5=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    local caps
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)
    if [[ -n "$caps" ]]; then
      export TORCH_CUDA_ARCH_LIST="$caps"
    else
      echo "nvidia-smi did not report a CUDA architecture. Fused kernels will not be built." >&2
      export AUDIO_DB_SKIP_NATIVE=1
    fi
  else
    echo "nvidia-smi is missing. Fused kernels will not be built on this machine." >&2
    export AUDIO_DB_SKIP_NATIVE=1
  fi
  if [[ "${AUDIO_DB_HDF5:-0}" != 1 ]]; then
    echo "Official libdfdata needs HDF5 headers. Set HDF5_DIR and rerun ROLES=deepfilternet." >&2
  fi
}
