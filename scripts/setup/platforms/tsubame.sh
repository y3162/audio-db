# Sourced by scripts/setup/setup.sh. Do not execute directly.
platform_prepare() {
  export AUDIO_DB_PLATFORM=tsubame
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
  fi
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is not on PATH. Install Rust 1.76 or newer, then rerun." >&2
    return 1
  fi
  if ! command -v module >/dev/null 2>&1 && [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.3.2
    if module load hdf5-parallel/1.14.3/gcc11.4.1; then
      export AUDIO_DB_HDF5=1
      if [[ -n ${HDF5_PARALLEL_DIR:-} ]]; then
        export HDF5_DIR="$HDF5_PARALLEL_DIR"
      fi
    else
      echo "hdf5-parallel/1.14.3/gcc11.4.1 did not load. Official libdfdata will not be built." >&2
      export AUDIO_DB_HDF5=0
    fi
  else
    echo "Environment modules are unavailable. CUDA and HDF5 were not loaded." >&2
    return 1
  fi
  export TORCH_CUDA_ARCH_LIST=9.0
  export AUDIO_DB_CUDA_FROM=module
}
