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
make experiment CONFIG=... NPROC=2                 # DDP、2 rank
make experiment CONFIG=... RESUME=OUTPUT_DIRECTORY  # latest.pt から続ける
make experiment CONFIG=... STOP_AFTER=1200          # 1200 秒で安全に停止
make test-cpu
make test
```

`make db-ingest` は `corpora.audio_files` を作り直す。明示されたときだけ、DB を退避してから実行する。`make setup` は DB を初期化しない。`make test-cpu` は CPU テストである。`make test` は空いている GPU のテストも実行する。CPU の成功は GPU の成功ではない。

## CLI

```bash
.venvs/se/bin/python -m src.experiments.cli \
  --config configs/reproduction/experiments/variants/mossformergan_se_16k.yaml \
  --preflight
.venvs/se/bin/python -m src.experiments.launcher --config CONFIG --nproc 2 --resume OUTPUT_DIRECTORY
.venvs/se/bin/python scripts/experiments/show_status.py OUTPUT_DIRECTORY
scripts/experiments/job.sh CONFIG 2
.venvs/se/bin/python -m src.experiments.data.cli doctor --database data/db/reproduction.duckdb
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m tests.suite_guard \
  .venvs/se/bin/python -m unittest tests.experiments.test_checkpoint -v
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

学習ディレクトリの状態は `scripts/experiments/show_status.py` で見る。Launcher の終了コードは 0 が完了、3 が安全な停止、1 が失敗である。`INTERRUPTED`、`FAILED`、`liveness` が `stale` の `RUNNING` は、同じ設定で `--resume OUTPUT_DIRECTORY` する。`complete.json` がある実行は再開しない。詳しくは [学習・実験](experiments/RUN.md)。

## 詰まったとき

- preflight が GPU を拒否する: 空いている番号だけを `CUDA_VISIBLE_DEVICES` に入れる。自分のジョブでなければ `allow_busy_gpu` を付けない。
- 評価が Manifest を読めない: 評価の入力は名前付きセットだけである。登録は `make db-init`。
- `make db-init` が拒否する: `recipe_json` の無い `eval.cases` がある。DB は変更されていない。
- 再開が設定差で失敗する: `output`、`checkpoint`、`stop_after_logical_steps` 以外を変えていないか確認する。以前の YAML にあった Trainer の `checkpoint_interval` は、同じ値を Selector の `latest_every` に書く（[Checkpoint](experiments/CHECKPOINT.md) の「以前の形式との互換」）。
- 新規の起動が `output directory already holds a run` で止まる: そのディレクトリに `complete.json`、`latest.pt`、`archive/*.pt`、以前の形式の `step_*.pt` のどれかがある。続きは `--resume`、やり直しは別の `output.directory` を使う。
- 再開が「behind latest.pt」で止まる: Latest より前の Archive から続けようとしている。古い状態から始めるときは、新しい `output.directory` と `initialize` を書く。
- テストランナーが `data_guard` で止まる: user namespace が使えない。保護なしで走らせてよいときだけ `AUDIO_DB_TEST_UNGUARDED=1` を付ける（[テスト](TESTING.md)）。
- 評価が終了コード 1 で終わる: 必須の WER が欠けている。`run.json` の `status` が `incomplete` なら `--resume` で失敗した発話を計算し直す。`validation_failed` は入力か参照の不備である。
