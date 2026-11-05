# 開発環境と検証

環境の構築手順は `docs/envs/SETUP.md`、`docs/envs/TSUBAME.md`、`docs/envs/SHINODA_LAB.md`、`docs/envs/ASR.md`。検証の記録は `docs/se/VALIDATION.md`、`docs/se/IMPLEMENTATION_STATUS.md`、`docs/asr/MODELS.md`。

## 仮想環境

プラットフォーム（`tsubame` / `shinoda_lab`）は `make setup` の引数だけである。仮想環境名には入れない。依存の正本は `requirements/<role>.lock.txt`。`make setup` は DB を初期化しない。

| 環境 | 用途 |
|---|---|
| `.venvs/se` | Python 3.12、torch 2.6.0+cu124、NumPy 2.5.4、DuckDB 2.0.0、`nnAudio==0.3.3`。SE、評価、`make db-init`、通常のテスト。SEMamba++ の CQT Discriminator が `nnAudio` を使う |
| `.venvs/asr` | Python 3.12、NeMo、Whisper、Transformers、NumPy 2.1.3。実 ASR を使う評価・スモーク。SE と ASR を同じプロセスでつなぐため、DuckDB・PESQ・STOI も入っている |
| `.venvs/official` | 公式実装・ESPnet レシピとの比較。NumPy 2.5 |
| `.venvs/semambapp` | SEMamba++ の fused CUDA。`mamba-ssm==2.3.2.post1` と `causal-conv1d==1.7.0` をソースビルドし、import 3 行だけ `patches/mamba_ssm-2.3.2.post1-init.patch` を適用する |
| `.venvs/deepfilternet` | 公式 `libdf` / `libdfdata`。NumPy 1.26。独立実装の実行時依存ではない。公式 checkout の `Cargo.lock` は更新しない。再ビルドは `build/libdf-src` だけで行う |

旧パス `.venvs/lab`、`.venvs/tsubame`、`.venvs/ref`、`.venvs/semamba`、`.venvs/cuda-nvcc`、`.venvs/clean-20261104` は役割別環境へ移行する前のものである。セットアップはこれらを削除しない。`.venvs/lab` は SHINODA-LAB の旧環境名で、今の対応は `.venvs/se` である。

`semamba` は `mamba-ssm` と `causal-conv1d` を sm_75（turing の Quadro RTX 5000）向けにソースビルドしてあり、cubin は sm_75 だけなので A100 ではロードできない。独立 SEMamba++ は既定では外部 scan を使わず、明示的に `scan_backend: fused` を指定した CUDA 実行時だけ `mamba_ssm` の `mamba_inner_fn` と Triton RMSNorm を読み込む。CPU では reference に戻る。

- NeMo は Python 3.13・NumPy 2.5.4 の lab と依存関係が合わないため、ASR の環境を分けている。NeMo などを lab に入れない。lab にパッケージを足すときは NumPy を上げない（`requirements/evaluation.txt` の手順）。
- Numba と NeMo は、`.venvs/*` の `nvidia/cuda_nvcc`（ptxas 12.2）を使う。システムの `/usr/local/cuda` は使わない。
- TSUBAME 4.0 は別の環境（`docs/envs/TSUBAME.md`）。既存環境を残した再構築手順は `docs/envs/CLEAN_REBUILD.md`。2026-11-04 の再構築先は `.venvs/clean-20261104` と、公式 `libdf` 用の `.venvs/clean-20261104-df`（NumPy 1.26）である。このノードには `.venvs/asr` が無く、Whisper Large v3 の小規模評価は `.venvs/clean-20261104` に `openai-whisper==20240930`、`tiktoken==0.7.0`、`more-itertools==10.5.0` を足して実行した。torch 2.6.0+cu124 と numpy 2.5.4 は変えていない。NeMo は入れていない。
- TSUBAME では長時間学習に `qsub`、短時間の開発・デバッグに `qrsh` を使う。資源と優先度の指定は `docs/envs/TSUBAME.md` を参照する。
- DeepFilterNet3 の spectral loss は通常、公式と同じ反射パディングを使う。厳密な決定的 CUDA では backward の制約のため別のパディングを使い、勾配の加算順が変わり得る。比較結果は `docs/training/VERIFICATION.md`。
- MP-SENet の学習 STFT は CPU と非決定的 CUDA では公式と同じ native centered reflection を使う。厳密な決定的 CUDA では reflection backward が使えないため明示的な反射に切り替え、値は一致しても勾配の加算順は変わり得る。公式 native 経路の検証と区別する。
- DeepFilterNet3 の解析 FFT・ERB の `log10`・unit norm は、`libdf_rfft.so` があるとき `realfft` 3.3.0（libdf 0.5.6 と同じ版）で計算する。無いときは `torch.fft.rfft` / `torch.log10` / `torch.sqrt` に戻り、libdf とのビット一致は成り立たない。`DF_RFFT_REQUIRED=1` のときは欠落を `FileNotFoundError` にする。ABI が違う共有ライブラリは `RuntimeError` にする。構築はリポジトリルートの `make df-rfft`（`Cargo.lock`、`realfft =3.3.0`）。`Cargo.toml` の `rust-version` は 1.70（`OnceLock`）。出力先は `DF_RFFT_TARGET`、読込先の上書きは `DF_RFFT_LIBRARY`。どちらもホストの固定パスを既定にしない。実行時に libdf は import しない。`realfft` 3.3.0 は MIT、`rustfft` 6.4.1 は MIT OR Apache-2.0 で、共有ライブラリへ静的に入る。libdf 自体は再配布しない。

## サーバーと GPU

- 研究室サーバーは ampere（RTX 3090 ×4）、tensor（A100 40GB ×7、80GB ×1）、turing（Quadro RTX 5000 ×4）、shannon（A100 80GB ×8）。役割別の `.venvs/se` を共通に使う。構築の入口は `make setup PLATFORM=shinoda_lab`。固定済み拡張の `setup.py` は CUDA 12.1 で `sm_75` / `sm_80` / `sm_87` / `sm_90` を明示するので、`TORCH_CUDA_ARCH_LIST` だけでは生成対象を絞れない。詳細は [SHINODA-LAB](../docs/envs/SHINODA_LAB.md)。再ビルドは `AUDIO_DB_REBUILD_MAMBA=1` を付けて `ROLES=semambapp` を実行するが、共有拡張と `build/native-src/*/build` を削除する。共有 `.so` は一番古い glibc のホストで作る。2026-11-04 時点では shannon が glibc 2.31、turing が 2.35 で、turing で作った拡張は shannon で `GLIBC_2.32`（`__libc_single_threaded`）不足によりロードできない。残した GCC 11 のオブジェクトを GCC 9 の再リンクに混ぜると、同じ未定義シンボルが残る。CUDA 12.1 の nvcc は `/net/ox/data/.apps/modules/binaries/cuda/12.1` を使う。`module load cuda/12.1` は別の 11.3 を指すことがある。
- GPU は他のユーザーも使う。使う前に `nvidia-smi` で空きを確認し、使用メモリが 1024 MiB 未満の GPU だけを指定する。評価 CLI は、使用メモリが 1024 MiB 以上の CUDA GPU を拒否する。`runtime.allow_busy_gpu: true` は、その GPU が自分のジョブだと分かっているときだけ使う。
- 学習の空き確認は、CUDA が認識した装置が MIG の場合、その MIG の空きメモリを使う。親 GPU の使用量は他の MIG を含み得るため判定に使わない。CUDA context 自身の使用分も含む。qsub での実機検証状況は `docs/training/JOB_CONTINUATION_AUDIT.md`。
- `CUDA_VISIBLE_DEVICES` で隠すと、見える GPU の番号は 0 から振り直される。`device` の指定はその番号。
- 設定やスモークに書かれた `cuda:1`（`configs/evaluation/e2e.yaml`、`tests/asr_smoke/run_smoke.py`）は固定の例。そのまま実行せず、空きを確認する。多くの試験は CPU で足りる。

## テスト

- CUDA checkpoint を GPU へ読み込んだ場合、CPU と CUDA の RNG 状態も GPU tensor になり得る。`restore_rng` は復元前に各状態を CPU に移す。CUDA 再開が動くことと、非決定的演算を含む学習がビット一致することは別に検証する。

```bash
.venvs/se/bin/python -m unittest discover -s tests/se -v          # SE
.venvs/se/bin/python -m unittest discover -s tests/asr -v         # ASR。重みをダウンロードしない
.venvs/se/bin/python -m unittest discover -s tests/evaluation -v  # 評価。一時 DB だけを書き、resource.duckdb は変更しない
.venvs/se/bin/python -m unittest discover -s tests/training -v    # 学習。CPU。resource.duckdb は開かない
.venvs/se/bin/python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --limit 2 --device cpu   # 合成音声 + mock
.venvs/asr/bin/python -m tests.asr_smoke.run_smoke parakeet whisper wav2vec2   # 実 ASR のスモーク（GPU）
```

学習の実行は `docs/training/MP_SENET.md`。決定的な CUDA 行列積には、プロセス起動前に `CUBLAS_WORKSPACE_CONFIG=:4096:8` を置く。複数 GPU は `CUDA_VISIBLE_DEVICES` と `torchrun` で、使用メモリが 1024 MiB 未満の GPU だけを見せる。

- チェックポイントや公式コードが無いテストは skip される。SKIP は PASS ではない。
- 実 SE と実 ASR をつなぐ接続確認は `configs/evaluation/e2e.yaml`。`.venvs/asr` で実行する。

## 公式コードと独立実装

- `src/se/official/` は公式実装の置き場。Git 管理外で、将来的に削除する予定（未実施）。
- 現在、公式一致の比較（`tests/se/reference/`、`tests/se/test_mp_senet.py`、`tests/se/test_tf_gridnet.py`）がこれを参照する。削除するときは、これらの扱いと `docs/se/VALIDATION.md` を同時に決める。参照を前提とする新しいコード・テスト・設定を増やさない。
- 独立 SE 実装は、公式コードに実行時依存しない。`tests/se/test_independence.py` は、`src/se/official/` を除いたコピーで MP-SENet と TF-GridNet が推論できることを確認する（ほかのモデルは範囲外）。`make db-init` も `official/` を読まない。
- ESPnet 全体をコピーしない。TF-GridNet は、推論に要る GridNet ブロックと正規化層だけを、出典 commit とライセンスを付けて引用している。ライセンスが不明なソース（SEMamba++）はコピーしない。
- 新しいモデルを足すときは、出典 commit とライセンスを `docs/se/MODELS.md` に記録し、公式出力との比較を行う。
- MossFormerGAN の公式 Generator と TF-Locoformer は `rotary-embedding-torch==0.8.3` を import する。`src.se` は TF-Locoformer を読み込むので、SE と ASR を同じプロセスで動かす `.venvs/asr` にも同じ版を入れる。2026-11-02 に `.venvs/lab` へ入れた。SEMamba++ の公式学習ステップは `.venvs/semamba` で実行する。その Triton RMSNorm は CPU tensor を拒否する。`.venvs/semamba` の DuckDB は 1.4.3 で、`voicebank.duckdb` は開けない。VoiceBank の準備は `.venvs/lab` が行い、worker だけ `FOUR_PROCESS_PYTHON` で semamba を指定する。
- チェックポイントのパスをコードに固定しない。呼び出し側が渡す。
- SEMamba++ の学習用 mel 損失は公式と同じ Slaney フィルタを `librosa.filters.mel` で CPU 上に作るため、学習環境に librosa が必要。推論 API はこの学習損失を import しない。公式 MP-SENet の import 用に空の `librosa` を `sys.modules` へ残さない。残すと、同じプロセスの SEMamba++ 公式 import が `librosa.filters` を読めなくなる。
- SEMamba++ の公式と独立の generator はパラメータ名が一致しても登録順が異なる。optimizer state を別実装へ位置のままロードしない。共通 checkpoint の比較は名前で対応付ける。パラメータに属さない state がある対応は失敗させる。検証と診断手順は [学習検証記録](../docs/training/VERIFICATION.md) を参照。
- SEMamba++ の fused mixer が `mamba_inner_fn` に渡す `xz` は、公式と同じ stride `(time, batch * time, 1)` の view である。ここを contiguous にコピーすると selective scan の呼び出し stride が変わり、backward が公式の再実行と別のばらつきになる。

## 検証の区別

| 呼び方 | 内容 | 言えること | 言えないこと |
|---|---|---|---|
| Mock | `name: mock`。SE は入力のコピー、ASR は試験用 | 配線・集計・再開・設定検証が通る | 実モデルが動くこと、性能 |
| 実モデルの推論スモーク | 実チェックポイントで短い入力が完了し、形状と有限値が正しい | その環境と入力でモデルが動く | 公式と一致すること、論文用のスコア |
| 公式一致検証 | 公式実装と公式チェックポイントの最終波形を、同じ入力・seed で比べ、最大絶対誤差を記録する | 記録した条件（モデル・長さ・サンプリングレート・device・seed）での一致 | 未計測の条件での一致 |

- 独立実装の CPU と CUDA の差は、公式との差ではない。公式の一致の証拠に使わない。
- TF-Locoformer の `WarmupReduceLROnPlateau` を公式一致と言うときは、`src/se/official/espnet` の `espnet2/schedulers/warmup_reducelronplateau.py` を import して比べる。`90eed8e` と checkout の当該ファイルが同一であることを確認する。移植クラスを二つ並べた比較は公式一致ではない。checkout が無い SKIP は PASS に数えない。学習ログの `si_sdr` は SI-SDR 損失、validation の `si_snr` は SI-SDR（dB）で、`valid.si_snr.best.pt` は後者を最大化する。
- ASR の「動作検証」は `docs/asr/MODELS.md` の表が正本。カタログに載っていることは検証済みを意味しない。
- 検証を報告するときは、何を（モデル、入力、device、seed、環境）確認したかを書く。
