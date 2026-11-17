# Point Numba at the CUDA 12 NVVM shipped in a virtualenv.
# An already-set CUDA_HOME is left unchanged. Official roles do not source this.
apply_cuda_wheel() {
  local python=$1
  if [[ -n "${CUDA_HOME:-}" ]]; then
    return 0
  fi
  if [[ ! -x "$python" ]]; then
    echo "cuda wheel: interpreter is missing: $python" >&2
    return 1
  fi
  local site home runtime libs
  site=$("$python" -c 'import site; print(site.getsitepackages()[0])') || return 1
  home="${site}/nvidia/cuda_nvcc"
  runtime="${site}/nvidia/cuda_runtime/lib"
  # nvidia-cuda-nvcc-cu12 ships libnvvm and ptxas. It does not ship the nvcc binary.
  if [[ ! -e "${home}/nvvm/lib64/libnvvm.so" ]]; then
    echo "cuda wheel: libnvvm is missing under ${home}" >&2
    return 1
  fi
  export CUDA_HOME="$home"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  libs="${CUDA_HOME}/nvvm/lib64:${CUDA_HOME}/lib64"
  if [[ -d "$runtime" ]]; then
    libs="${libs}:${runtime}"
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${libs}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${libs}"
  fi
}
