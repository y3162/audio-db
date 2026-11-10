# コマンド・チートシート

環境は `.venvs/se` が既定である。プラットフォーム名は仮想環境のパスに入らない。

## Make

```bash
make setup PLATFORM=shinoda_lab
make setup PLATFORM=tsubame
make doctor
make db-link DRY_RUN=1
make db-init
make reproduction-doctor
make reproduction-init-db
make experiment-preflight CONFIG=tests/fixtures/configs/experiments/mp_senet_gan.yaml
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml
make test-cpu
```

`make db-ingest` は `corpora.audio_files` を作り直す。明示されたときだけ、DB を退避してから実行する。`make setup` は DB を初期化しない。

## CLI

```bash
.venvs/se/bin/python -m src.experiments.cli \
  --config configs/reproduction/experiments/variants/mossformergan_se_16k.yaml \
  --preflight
.venvs/se/bin/python -m src.experiments.data.cli doctor --database data/db/reproduction.duckdb
.venvs/se/bin/python -m src.evaluation.cli \
  --config tests/fixtures/configs/evaluation/smoke.yaml \
  --limit 2 --device cpu
.venvs/se/bin/python -m src.evaluation.cli \
  --config configs/reproduction/evaluation/example.yaml \
  --dry-run
```

## 実行環境

| 役割 | パス |
|---|---|
| SE、評価、実験、通常テスト | `.venvs/se` |
| 実 ASR | `.venvs/asr` |
| 公式実装との比較 | `.venvs/official` |
| SEMamba++ fused CUDA | `.venvs/semambapp` |
| 公式 `libdf` | `.venvs/deepfilternet` |

CPU テストは `CUDA_VISIBLE_DEVICES=""` で実行する。GPU を使う前に `nvidia-smi` を見る。

## 状態

```bash
nvidia-smi
make doctor
make reproduction-doctor
```

学習ディレクトリの `status.json` が `INTERRUPTED` のときは、同じ設定で `--resume` する。`complete.json` がある実行は再開しない。

## 詰まったとき

- preflight が GPU を拒否する: 空いている番号だけを `CUDA_VISIBLE_DEVICES` に入れる。自分のジョブでなければ `allow_busy_gpu` を付けない。
- 評価が Manifest を読めない: 評価の入力は名前付きセットだけである。登録は `make db-init`。
- `make db-init` が拒否する: `recipe_json` の無い `eval.cases` がある。DB は変更されていない。
- 再開が設定差で失敗する: `logging` 以外を変えていないか確認する。`stop_after_logical_steps` を外した設定への再開は、モデル構築の前に失敗する。
