# Sourced by scripts/setup/setup.sh. Use the caller's toolchain.
platform_prepare() {
  if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME
    CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
  fi
}
