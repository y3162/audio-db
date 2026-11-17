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
  # Numba looks up libcudart.so.<version> in CUDA_HOME/lib64. The wheel ships
  # only libcudart.so.12, and uv pip sync removes this link without changing
  # the setup stamp. Without it, dlopen("libcudart.so") loads the linker's
  # CUDA 11.0 runtime, which Numba 0.61 rejects with an empty arch list.
  if [[ ! -e "${runtime}/libcudart.so.12" ]]; then
    echo "cuda wheel: libcudart.so.12 is missing under ${runtime}" >&2
    return 1
  fi
  mkdir -p "${home}/lib64"
  ln -sfn "${runtime}/libcudart.so.12" "${home}/lib64/libcudart.so.12"
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
