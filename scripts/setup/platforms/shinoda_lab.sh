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
  # Numba ptxas comes from the nvidia-cuda-nvcc wheel (ptxas only). Extension
  # builds need a full toolkit; load a module when Environment Modules exist.
  export AUDIO_DB_CUDA_FROM=wheel
  if ! command -v module >/dev/null 2>&1 && [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
  if command -v module >/dev/null 2>&1; then
    if module load cuda/12.1 2>/dev/null || module load cuda/12.8 2>/dev/null; then
      export AUDIO_DB_CUDA_FROM=module
    else
      echo "cuda/12.1 and cuda/12.8 did not load. Fused CUDA extensions will not be built." >&2
    fi
  fi
  export AUDIO_DB_HDF5=0
  if [[ -f /usr/lib/x86_64-linux-gnu/hdf5/serial/include/hdf5.h ]]; then
    export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial
    export AUDIO_DB_HDF5=1
  fi
  # One shared .venvs/semambapp is used on every lab host. A cubin built
  # only for the machine running setup cannot load on the others:
  # Turing 7.5, A100 8.0, RTX 3090 8.6.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local caps
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)
    if [[ -z "$caps" ]]; then
      echo "nvidia-smi did not report a CUDA architecture. Fused kernels will not be built." >&2
      export AUDIO_DB_SKIP_NATIVE=1
    fi
  else
    echo "nvidia-smi is missing. Fused kernels will not be built on this machine." >&2
    export AUDIO_DB_SKIP_NATIVE=1
  fi
  if [[ "${AUDIO_DB_HDF5:-0}" != 1 ]]; then
    echo "Official libdfdata needs HDF5 headers. Install libhdf5-dev or set HDF5_DIR, then rerun ROLES=deepfilternet." >&2
  fi
  if [[ "${AUDIO_DB_CUDA_FROM}" == wheel ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Fused CUDA extensions need nvcc. Load cuda/12.1 (or cuda/12.8) via modules, then rerun ROLES=semambapp." >&2
  fi
}
