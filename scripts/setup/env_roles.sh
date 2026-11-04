# Sourced by scripts/setup/setup.sh after the platform script.
install_roles() {
  local roles=$1
  local status=0
  local role
  IFS=',' read -r -a role_list <<< "$roles"
  for role in "${role_list[@]}"; do
    install_role "$role" || status=1
  done
  build_df_rfft || status=1
  return "$status"
}

install_role() {
  local role=$1
  local venv=".venvs/${role}"
  local lock="requirements/${role}.lock.txt"
  local python
  python="$(pwd)/${venv}/bin/python"
  if [[ ! -f "$lock" ]]; then
    echo "missing $lock" >&2
    return 1
  fi
  if [[ ! -x "$python" ]]; then
    uv venv "$venv" --python 3.12 || return 1
  fi
  local digest
  digest=$(sha256sum "$lock" | awk '{print $1}')
  local stamp="${venv}/.audio-db-setup-stamp"
  if [[ "${FORCE_SETUP:-}" != 1 && -f "$stamp" && $(cat "$stamp") == "$digest" ]] && role_imports "$role"; then
    echo "unchanged venv ${role}"
    return 0
  fi
  uv pip sync --python "$python" --index-strategy unsafe-best-match "$lock" || return 1
  link_nvcc "$python"
  case "$role" in
    asr)
      if ! "$python" -c 'import whisper' >/dev/null 2>&1; then
        uv pip install --python "$python" --no-build-isolation --index-strategy unsafe-best-match openai-whisper==20240930 || return 1
      fi
      ;;
    semambapp)
      build_mamba "$python" || return 1
      ;;
    deepfilternet)
      build_libdf "$python" || return 1
      ;;
  esac
  if ! role_imports "$role"; then
    echo "${role} imports failed after install" >&2
    return 1
  fi
  printf '%s\n' "$digest" > "$stamp"
  echo "installed venv ${role}"
}

role_imports() {
  local role=$1
  local python=".venvs/${role}/bin/python"
  case "$role" in
    se|official)
      "$python" -c 'import torch, torchaudio, duckdb, yaml, numpy'
      ;;
    asr)
      "$python" -c 'import torch, whisper, duckdb'
      ;;
    semambapp)
      "$python" -c 'import torch, causal_conv1d_cuda, selective_scan_cuda'
      ;;
    deepfilternet)
      "$python" -c 'import torch, libdf, libdfdata'
      ;;
    *)
      echo "unknown role ${role}" >&2
      return 1
      ;;
  esac
}

link_nvcc() {
  local python=$1
  local site
  site=$("$python" -c 'import site; print(site.getsitepackages()[0])')
  local home="${site}/nvidia/cuda_nvcc"
  local runtime="${site}/nvidia/cuda_runtime/lib"
  if [[ -d "$home" && -e "${runtime}/libcudart.so.12" ]]; then
    mkdir -p "${home}/lib64"
    ln -sfn "${runtime}/libcudart.so.12" "${home}/lib64/libcudart.so.12"
  fi
}

build_mamba() {
  local python=$1
  if [[ "${AUDIO_DB_SKIP_NATIVE:-}" == 1 ]]; then
    echo "Fused CUDA extensions were not built. Re-run on a machine with a visible GPU." >&2
    return 1
  fi
  local causal="build/native-src/causal-conv1d"
  local mamba="build/native-src/mamba"
  if [[ ! -f "$causal/setup.py" || ! -f "$mamba/setup.py" ]]; then
    echo "Pinned causal-conv1d or mamba checkout is missing. Run: make assets" >&2
    return 1
  fi
  local cuda_home=${CUDA_HOME:-}
  if [[ "${AUDIO_DB_CUDA_FROM:-}" == wheel ]]; then
    local site
    site=$("$python" -c 'import site; print(site.getsitepackages()[0])')
    cuda_home="${site}/nvidia/cuda_nvcc"
  fi
  if [[ -z "$cuda_home" || ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc was not found. On TSUBAME run: module load cuda/12.3.2" >&2
    return 1
  fi
  export CUDA_HOME="$cuda_home"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export MAX_JOBS="${MAX_JOBS:-4}"
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export MAMBA_FORCE_BUILD=TRUE
  local site init
  site=$("$python" -c 'import site; print(site.getsitepackages()[0])') || return 1
  init="${site}/mamba_ssm/__init__.py"
  if [[ -f "$init" ]] && grep -q 'clean-rebuild:' "$init" \
      && "$python" -c 'import causal_conv1d_cuda, selective_scan_cuda' >/dev/null 2>&1; then
    return 0
  fi
  if ! compgen -G "${site}/causal_conv1d_cuda*.so" >/dev/null || ! compgen -G "${site}/selective_scan_cuda*.so" >/dev/null; then
    (
      cd "$causal"
      uv pip install --python "$python" --no-build-isolation --no-deps --index-strategy unsafe-best-match .
    ) || return 1
    (
      cd "$mamba"
      uv pip install --python "$python" --no-build-isolation --no-deps --index-strategy unsafe-best-match .
    ) || return 1
  fi
  "$python" scripts/apply_mamba_init_patch.py --init "$init" || return 1
}

libdf_imports() {
  local python=$1
  "$python" -c 'import libdf, libdfdata' >/dev/null 2>&1
}

install_libdf_wheels() {
  local python=$1
  local lib_wheel data_wheel
  lib_wheel=$(ls -1 build/libdf-cargo/wheels/DeepFilterLib-*.whl 2>/dev/null | head -1 || true)
  data_wheel=$(ls -1 build/libdf-cargo/wheels/DeepFilterDataLoader-*.whl 2>/dev/null | head -1 || true)
  if [[ -z "$lib_wheel" || -z "$data_wheel" ]]; then
    return 1
  fi
  uv pip install --python "$python" --no-deps "$lib_wheel" "$data_wheel" || return 1
  libdf_imports "$python"
}

# Cargo 1.99 rewrites the pinned lock (version 3) and refuses that rewrite
# when --locked is set. The official checkout stays at its commit. A shared
# clone under build/libdf-src may refresh its own lock, offline.
libdf_build_tree() {
  local src dest commit
  src="$(pwd)/src/se/official/deepfilternet"
  dest="$(pwd)/build/libdf-src"
  commit=$(git -C "$src" rev-parse HEAD)
  if [[ ! -d "$dest/.git" ]] || [[ "$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
    rm -rf "$dest"
    git clone --shared "$src" "$dest"
    git -C "$dest" checkout --detach "$commit"
  fi
  printf '%s\n' "$dest"
}

maturin_release() {
  local python=$1
  if "$python" -m maturin build --release --offline --locked --interpreter "$python"; then
    return 0
  fi
  "$python" -m maturin build --release --offline --interpreter "$python"
}

build_libdf() {
  local python=$1
  local repo="src/se/official/deepfilternet"
  if [[ ! -f "$repo/pyDF/Cargo.toml" ]]; then
    echo "Pinned DeepFilterNet checkout is missing. Run: make assets" >&2
    return 1
  fi
  if libdf_imports "$python"; then
    return 0
  fi
  if install_libdf_wheels "$python"; then
    echo "reinstalled libdf wheels"
    return 0
  fi
  export PATH="$(dirname "$python"):${PATH}"
  export CARGO_TARGET_DIR="${PWD}/build/libdf-cargo"
  local tree
  tree=$(libdf_build_tree)
  (
    cd "$tree/pyDF"
    maturin_release "$python"
  ) || return 1
  local wheel
  wheel=$(ls -1 "${CARGO_TARGET_DIR}/wheels"/DeepFilterLib-*.whl | head -1)
  uv pip install --python "$python" --no-deps "$wheel" || return 1
  if [[ "${AUDIO_DB_HDF5:-0}" != 1 && -z "${HDF5_DIR:-}" ]]; then
    echo "libdf imported path is installed. libdfdata was not built because HDF5_DIR is unset." >&2
    echo "On TSUBAME: module load hdf5-parallel/1.14.3/gcc11.4.1 && make setup-env PLATFORM=tsubame ROLES=deepfilternet" >&2
    return 1
  fi
  if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf is required to build libdfdata and was not found." >&2
    return 1
  fi
  (
    cd "$tree/pyDF-data"
    maturin_release "$python"
  ) || return 1
  wheel=$(ls -1 "${CARGO_TARGET_DIR}/wheels"/DeepFilterDataLoader-*.whl | head -1)
  uv pip install --python "$python" --no-deps "$wheel" || return 1
}

build_df_rfft() {
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is missing. libdf_rfft.so was not rebuilt." >&2
    return 1
  fi
  cargo build --release --locked \
    --manifest-path src/se/models/deepfilternet3/rfft_kernel/Cargo.toml \
    --target-dir build/df-rfft || return 1
  local built="build/df-rfft/release/libdf_rfft.so"
  local tracked="src/se/models/deepfilternet3/libdf_rfft.so"
  if [[ -f "$tracked" ]] && ! cmp -s "$built" "$tracked"; then
    echo "The new libdf_rfft.so differs from the tracked file. The tracked file was not replaced." >&2
    echo "Load the new build with DF_RFFT_LIBRARY=$PWD/$built" >&2
  fi
  return 0
}
