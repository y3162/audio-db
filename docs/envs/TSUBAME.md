# TSUBAME 4.0

現在の構築は次のとおり。ジョブの投入はこの手順に含まない。

```bash
cp .env.example .env
make setup PLATFORM=tsubame
make doctor
```

CUDA は `cuda/12.3.2`、HDF5 は `hdf5-parallel/1.14.3/gcc11.4.1` をセットアップが読み、融合カーネルの `TORCH_CUDA_ARCH_LIST` は `9.0` である。PyTorch の wheel は cu124。詳細は [セットアップ](SETUP.md)。

2026-11-04 に計算ノード `r4n11` で `make setup PLATFORM=tsubame` を再実行し、終了コード 0 だった。診断は `PASS=47 FAIL=0 SKIP=0 NOT_TESTED=0`。PyTorch の CUDA 加算と `causal_conv1d_fn` も通った。`deepfilternet` は、Cargo 1.99 が公式 `Cargo.lock` を `--locked` で更新できないため、既存 wheel を入れ直した。公式 checkout の `Cargo.lock` は変えていない。SHINODA-LAB の同じ入口は未実行である。

以下は 2026-11-02 から 2026-11-04 に記録した、旧パス `.venvs/tsubame` と `.venvs/clean-20261104` の構築例である。新しい役割別環境は上の `make setup` が作る。旧ディレクトリは削除しない。

GPU の計算と CUDA 拡張のビルドはスケジューラで確保した計算ノードで行う。`gpu_h` は CPU 4 コア、ホストメモリ 48 GB、H100 の半分の MIG を割り当てる。GPU メモリの容量はジョブ内で `nvidia-smi` と PyTorch から確認する。`gpu_1` は GPU 1 枚である。詳細は [公式ジョブ手引き](https://www.t4.cii.isct.ac.jp/docs/all/handbook.ja/jobs/) と [ソフトウェア手引き](https://www.t4.cii.isct.ac.jp/docs/all/handbook.ja/software/) を参照する。

長時間の学習は `qsub`、短時間の開発・デバッグは `qrsh` を使う。いずれも優先度は、ユーザーから別の指示がなければ `-p -5` とする。30 分程度以上のジョブは、投入前にそのジョブを含むユーザーの明示的な指示または追加承認が必要である。

以下の初期構築例では 5 モデルの小規模 GPU スモークを確認した。この環境構築例には融合カーネルを含めない。融合経路の制約は `.agents/ARCHITECTURE.md` と `.agents/OPERATIONS.md` にある。研究室環境をコピーせず、リポジトリのルートで実行する。

```bash
uv venv .venvs/tsubame --python 3.12
uv pip install --python .venvs/tsubame/bin/python torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venvs/tsubame/bin/python duckdb==2.0.0 numpy==2.5.4 scipy==1.18.1 librosa==1.0.0 soundfile==0.14.0 pyyaml==6.0.3 pystoi==0.4.1 pesq==0.0.4 pytest==9.2.0 einops==0.8.2 transformers==4.48.3 nnAudio==0.3.3
```

利用する TSUBAME グループを指定して GPU を確保する。ジョブ内で `nvidia-smi` の空きと MIG を確認してから計算する。デバイス番号を他ノードの物理番号に固定しない。

```bash
qrsh -g <TSUBAMEグループ> -p -5 -l gpu_h=1 -l h_rt=1:00:00
hostname
nvidia-smi
nvidia-smi -L
.venvs/tsubame/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))'
PYTHONPATH="$PWD" .venvs/tsubame/bin/python tests/numerics/gpu_smoke.py
```

SEMamba++ の `scan_backend: fused` は `mamba-ssm` と `causal-conv1d` が必要。上記の初期構築例には両方とも含まれない。fused 指定は `.venvs/semambapp` の `mamba-ssm` と `causal-conv1d` が必要である。研究室でビルドした拡張は流用しない。

初期のビルド調査では、確保した計算ノード上で次を試した。その後、CUDA 12.3.2 と PyTorch CUDA 12.4 の組合せで H100 の融合カーネルを実行したが、`mamba-ssm` の未使用 Mamba3 自動 import を外す必要があった。下記だけでは動作手順として完結しない。詳細は学習検証記録を参照する。

```bash
module load cuda/12.3.2
export PATH="$HOME/.local/bin:$PWD/.venvs/tsubame/bin:$PATH"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=9.0
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
uv pip install --python .venvs/tsubame/bin/python ninja wheel
uv pip install --python .venvs/tsubame/bin/python --no-deps --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.7.0
uv pip install --python .venvs/tsubame/bin/python --no-deps --no-build-isolation --no-binary mamba-ssm mamba-ssm==2.3.2.post1
```

旧環境 `.venvs/tsubame-audio` を使用した 2026-11-02 の `gpu_h=1` ジョブでは H100 MIG 3g.47gb、47,488 MiB、compute capability 9.0、driver 580.105.08 を確認し、PyTorch 2.6.0+cu124 の CUDA 演算が成功した。上記スモーク相当の小さな合成データでは 5 モデルの有限勾配、更新、checkpoint 保存・読込が成功した。SEMamba++ は reference scan。5 モデルとも保存した重みを推論 API に読み直し、短い合成波形を処理した。公式バッチ、融合カーネル、forward と backward 別のメモリピークは未検証である。構築時と step 全体の peak はジョブログに出すが、直前のモデルの reserved メモリを含み得る。

再作成した `.venvs/tsubame` でも、2026-11-02 に 5 分の `gpu_h=1` qrsh ジョブで GPU 構成の認識と単純な CUDA 演算を確認し、別の 5 分のジョブで上記 5 モデルの小規模な更新・checkpoint 保存と読込・推論が成功した。SEMamba++ は reference scan であり、融合経路は未検証である。

さらに同日、`.venvs/tsubame` を削除し、上記の初期構築コマンドのみで作り直した環境でも、5 モデルの同じスモークが成功した。ログは `.venvs/tsubame/logs/rebuild-gpu-smoke.log`。

既存環境を残した再取得・再ビルドは [クリーン再構築](CLEAN_REBUILD.md)。DeepFilterNet3 の解析 FFT は、リポジトリルートの `make df-rfft` で共有ライブラリを作る。Cargo の出力先は `DF_RFFT_TARGET`、読込先の上書きは `DF_RFFT_LIBRARY`。どちらもホスト固有のパスを既定にしない。Rust は利用者の `cargo` を使う。このリポジトリで通したコンパイラは rustc 1.99.1 で、別バージョンでの再コンパイルは未実施。手順は `make df-rfft`。`realfft` 3.3.0 は MIT、`rustfft` 6.4.1 は MIT OR Apache-2.0。libdf 自体は再配布しない。
