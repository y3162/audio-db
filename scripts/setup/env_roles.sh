# Sourced by scripts/setup/setup.sh after the platform script.
install_roles() {
  local roles=$1
  local status=0
  local role
  IFS=',' read -r -a role_list <<< "$roles"
  for role in "${role_list[@]}"; do
    install_role "$role" || status=1
  done
  if [[ ",$roles," == *,se,* ]]; then
    build_df_rfft || status=1
  fi
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
  if [[ "${FORCE_SETUP:-}" != 1 && -f "$stamp" && $(cat "$stamp") == "$digest" ]] \
      && [[ "$role" != se || "${AUDIO_DB_REBUILD_MAMBA:-}" != 1 ]] && role_imports "$role"; then
    echo "unchanged venv ${role}"
    return 0
  fi
  uv pip sync --python "$python" --index-strategy unsafe-best-match "$lock" || return 1
  link_nvcc "$python"
  case "$role" in
    se)
      if ! "$python" -c 'import whisper' >/dev/null 2>&1; then
        uv pip install --python "$python" --no-build-isolation --index-strategy unsafe-best-match openai-whisper==20240930 || return 1
      fi
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
    se)
      "$python" -c 'import torch, torchaudio, nemo, numba, whisper, duckdb, yaml, numpy, causal_conv1d_cuda, selective_scan_cuda; from nnAudio.features.cqt import CQT2010v2'
      ;;
    official)
      "$python" -c 'import torch, torchaudio, duckdb, yaml, numpy'
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
  # Keep the venv interpreter. readlink -f follows it to uv's base Python.
  python=$(cd "$(dirname "$python")" && pwd)/$(basename "$python")
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
  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "nvcc was not found. Put the CUDA Toolkit on PATH or set CUDA_HOME to its root, then rerun setup." >&2
    return 1
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export MAX_JOBS="${MAX_JOBS:-4}"
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export MAMBA_FORCE_BUILD=TRUE
  local site init
  site=$("$python" -c 'import site; print(site.getsitepackages()[0])') || return 1
  if [[ "${AUDIO_DB_REBUILD_MAMBA:-}" == 1 ]]; then
    uv pip uninstall --python "$python" causal-conv1d mamba-ssm >/dev/null 2>&1 || true
    rm -f "${site}"/causal_conv1d_cuda*.so "${site}"/selective_scan_cuda*.so
    # Objects from a different toolchain can be reused by an incremental build.
    rm -rf "${causal}/build" "${mamba}/build"
  fi
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

libdf_build_tree() {
  local src dest commit
  src="$(pwd)/src/official/deepfilternet"
  dest="$(pwd)/build/libdf-src"
  commit=$(git -C "$src" rev-parse HEAD) || return 1
  if [[ ! -d "$dest/.git" ]] || [[ "$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
    rm -rf "$dest" || return 1
    git clone --shared "$src" "$dest" || return 1
    git -C "$dest" checkout --detach "$commit" || return 1
  fi
  # The upstream workspace also includes a GUI demo and a LADSPA plugin.
  # Prune those members from the build copy so their dependencies are not resolved.
  sed '/^  "ladspa",$/d; /^  "demo",$/d' "$src/Cargo.toml" >"$dest/Cargo.toml" || return 1
  cp "$src/Cargo.lock" "$dest/Cargo.lock" || return 1
  printf '%s\n' "$dest"
}

build_libdf() (
  local python=$1
  local repo="src/official/deepfilternet"
  if [[ ! -f "$repo/pyDF/Cargo.toml" ]]; then
    echo "Pinned DeepFilterNet checkout is missing. Run: make assets" >&2
    return 1
  fi
  if libdf_imports "$python"; then
    return 0
  fi
  export PATH="$(dirname "$python"):${PATH}"
  local tree temporary
  tree=$(libdf_build_tree) || return 1
  (
    cd "$tree" || exit 1
    cargo tree -p DeepFilterLib -p DeepFilterDataLoader >/dev/null
  ) || return 1
  "$python" - "$repo/Cargo.lock" "$tree/Cargo.lock" <<'PY' || return 1
import sys
import tomllib


def packages(path):
    with open(path, "rb") as lock:
        return {
            (pkg["name"], pkg["version"], pkg.get("source"), pkg.get("checksum"))
            for pkg in tomllib.load(lock)["package"]
        }


added = packages(sys.argv[2]) - packages(sys.argv[1])
if added:
    names = ", ".join(sorted(f"{name} {version}" for name, version, _, _ in added))
    sys.exit(f"DeepFilterNet build would change pinned Rust dependencies: {names}")
PY
  temporary=$(mktemp -d) || return 1
  trap 'rm -rf "$temporary"' EXIT
  # Maturin copies Cargo's shared objects before packing the wheel. Keep
  # that copy on local storage by default when the checkout is on a share.
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$temporary/cargo}"
  (
    cd "$tree/pyDF" || exit 1
    "$python" -m maturin build --release --locked --compatibility linux \
      --interpreter "$python" --out "$temporary/lib"
  ) || return 1
  uv pip install --python "$python" --no-deps "$temporary/lib/"*.whl || return 1
  if [[ "${AUDIO_DB_HDF5:-0}" != 1 && -z "${HDF5_DIR:-}" ]] && ! pkg-config --exists hdf5; then
    echo "libdf was installed. libdfdata needs HDF5 headers. Make HDF5 visible to pkg-config or set HDF5_DIR, then rerun make setup-env ROLES=deepfilternet." >&2
    return 1
  fi
  (
    cd "$tree/pyDF-data" || exit 1
    "$python" -m maturin build --release --locked --compatibility linux \
      --interpreter "$python" --out "$temporary/data"
  ) || return 1
  uv pip install --python "$python" --no-deps "$temporary/data/"*.whl
)

build_df_rfft() {
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is missing. libdf_rfft.so was not rebuilt." >&2
    return 1
  fi
  local manifest="src/models/deepfilternet3/rfft_kernel/Cargo.toml"
  cargo build --release --locked --manifest-path "$manifest" --target-dir build/df-rfft || return 1
  local built="build/df-rfft/release/libdf_rfft.so"
  local tracked="src/models/deepfilternet3/libdf_rfft.so"
  if [[ -f "$tracked" ]] && ! cmp -s "$built" "$tracked"; then
    echo "The new libdf_rfft.so differs from the tracked file. The tracked file was not replaced." >&2
    echo "A process that does not set DF_RFFT_LIBRARY uses this build when the tracked file does not load." >&2
    echo "DF_RFFT_LIBRARY=$PWD/$built still overrides that choice." >&2
  fi
  return 0
}
