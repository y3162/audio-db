# セットアップ

利用者が編集するのは `.env` だけである。プラットフォームは構築コマンドの引数で、仮想環境名には入らない。

```bash
cp .env.example .env
# AUDIO_DB_DATASET_ROOT と、必要なら UV_CACHE_DIR を書く
make setup PLATFORM=tsubame
make doctor
```

SHINODA-LAB では最後の構築コマンドだけを変える。

```bash
make setup PLATFORM=shinoda_lab
```

`source .env` は任意である。既に export した変数は `.env` より優先される。`make` は `.env` を、未設定の変数に限って読む。

## 役割

| パス | 用途 |
|---|---|
| `.venvs/se` | 5 モデルの独立実装、DB、評価、通常のテスト |
| `.venvs/asr` | NeMo、Whisper、Transformers。SE と ASR を同じプロセスでつなぐ |
| `.venvs/official` | 公式実装との比較。NumPy 2.5 |
| `.venvs/semambapp` | SEMamba++ の fused CUDA（`mamba-ssm` 2.3.2.post1、`causal-conv1d` 1.7.0） |
| `.venvs/deepfilternet` | 公式 `libdf`。PyO3 が NumPy 1.26 を要求するため分けている |

依存の正本は `requirements/<role>.in` と、そこから作った `requirements/<role>.lock.txt` である。通常の `make setup` は lock を更新しない。lock を作り直すときだけ `scripts/setup/compile_locks.sh` を実行する。

SE 学習ログを TensorBoard で確認できるよう、`.venvs/se` に `tensorboard==2.20.0` を含める。TSUBAME と SHINODA-LAB のどちらも `make setup PLATFORM=...` で構築される同じ lock を使う。

## 部分実行

```bash
make setup-env PLATFORM=tsubame ROLES=se
make assets MODELS=mp_senet
make assets-check
make db-link DRY_RUN=1
make db-link ONLY=LibriSpeech ALLOW_PARTIAL=1
make test-setup
```

`make setup` は `db-init`、`db-ingest`、`db-bootstrap` を呼ばない。

## 公式アセット

取得元・commit・SHA-256 は `requirements/official-assets.toml`。配置先は既存テストが読むパスである。

| モデル | ソース | 重み |
|---|---|---|
| MP-SENet | `src/se/official/mp-senet` @ `89932cfe` | `data/checkpoints/mp-senet/g_best_vb` と `g_best_dns`（リポジトリ内の `best_ckpt`） |
| DeepFilterNet3 | `src/se/official/deepfilternet` @ `d375b2d83` | `data/checkpoints/deepfilternet3/DeepFilterNet3/`（ピンした commit の公式 zip） |
| SEMamba++ | `src/se/official/semambapp` @ `cd369c89` | `data/checkpoints/semambapp/semambapp.pth` |
| MossFormerGAN | `src/se/official/clearervoice-studio` @ `6b3774dc` | `data/checkpoints/mossformergan/last_best_checkpoint.pt` |
| TF-GridNet | `src/se/official/espnet` @ `0d136c54`（新規 clone は `espnet2` のみ） | `data/checkpoints/tf-gridnet/urgent24/.../21epoch.pth` と `ms-snsd/valid.loss.best.pth` |

既にピンした commit のツリーがあれば再取得しない。別 commit や手元の変更は上書きしない。チェックサムが違う既存ファイルも置き換えない。

TF-GridNet の重みカードは、コードの Apache-2.0 とは別に利用条件を書いていない。SEMamba++ の重みカードは GPL-2.0 と表示されている。ソースツリーにライセンスファイルは無い。

## 失敗したとき

同じ `make setup PLATFORM=...` を再実行する。成功済みの仮想環境（lock のハッシュが一致し、import できるもの）と、検証済みのファイルはやり直さない。`FORCE_SETUP=1` は lock の同期からやり直すが、仮想環境ディレクトリは消さない。

公式 `libdfdata` は HDF5 が必要である。TSUBAME では `hdf5-parallel/1.14.3/gcc11.4.1` を読む。モジュールが無い場合、`libdf` 本体の成否と `libdfdata` の未構築は分けて表示する。

`libdf` のビルドは `src/se/official/deepfilternet` を書き換えない。Cargo 1.99 はピンした `Cargo.lock`（version 3）を `--locked` のまま読めず、ロックの更新を拒否する。再ビルドが必要なときは `build/libdf-src` の共有 clone だけでロックを更新し、`--offline` で maturin する。既に `build/libdf-cargo/wheels` にある wheel が import できるときは、それを入れ直して再コンパイルしない。
