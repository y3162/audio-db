# 環境とテスト

依存の正本は `requirements/<role>.lock.txt`。`make setup` は DB を初期化しない。

| 環境 | 用途 |
|---|---|
| `.venvs/se` | Python 3.12、torch 2.6.0+cu124。SE、評価、実験、通常のテスト |
| `.venvs/asr` | NeMo、Whisper、Transformers。実 ASR |
| `.venvs/official` | 公式実装との比較 |
| `.venvs/semambapp` | SEMamba++ の fused CUDA |
| `.venvs/deepfilternet` | 公式 `libdf`。独立実装の実行時依存ではない |

`src/se/official/` は Git 管理外。独立実装はここを import しない。テストの公式比較だけが読む。

## GPU

- 使う前に `nvidia-smi` を見る。実験 runtime は CUDA を要求したとき、空いていない GPU を `PrepareError` にする。
- `runtime.kwargs.allow_busy_gpu: true` はその装置が自分のジョブだと分かっているときだけ。
- `CUDA_VISIBLE_DEVICES` で隠すと、見える番号は 0 から振り直される。評価と学習の空き確認は、未設定と空文字をどちらも「見える番号を物理番号のまま見る」として扱う。UUID と MIG の文字列は、別の物理 GPU の番号に読み替えない。
- CPU テストは `CUDA_VISIBLE_DEVICES=""` で実行する。CPU の成功を GPU テストの PASS にしない。

```bash
make experiment-preflight CONFIG=tests/fixtures/configs/experiments/mp_senet_gan.yaml
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml [NPROC=2] [RESUME=dir]
make test-cpu
```

## 学習の起動

- 学習は Launcher（`python -m src.experiments.launcher`、`make experiment`、`scripts/experiments/job.sh`）から起動する。単一 GPU と DDP、TSUBAME のジョブ内、研究室サーバー、ローカルで同じである。preflight は Optimizer を更新しない。
- Launcher はジョブの投入、GPU の選択、再試行をしない。それらを足さない。失敗後は `--resume <出力ディレクトリ>` で続ける。
- 終了コードは 0 が `COMPLETED`、3 が `INTERRUPTED`、1 が `FAILED`。`status.json` の `RUNNING` は、`scripts/experiments/show_status.py` の `liveness` が `alive` のときだけ実行中とみなす。
- 手順は `docs/experiments/RUN.md`、仕様は `docs/experiments/LAUNCHER.md`。
- Launcher は、その起動の学習が `RUNNING` になってから、`nvidia-smi` のデバイス全体の使用メモリを実験ディレクトリの `gpu_memory.jsonl` に追記する。取得できないとき、GPU を隠したとき、間隔を 0 にしたときは学習を止めない。`torch.cuda` の allocated / reserved のピークは、これとは別の `AUDIO_DB_RESOURCE_PROFILE=1` である。

## テスト

- `make test-cpu` は CPU だけ。`make test` は、空いている GPU があるときだけ GPU レーンも実行する。CPU の成功を GPU の PASS にしない。
- テストは `data/` の下に書かない。書き込みは一時ディレクトリで行う。本番の DB やスナップショットを作る処理をテストに入れない。
- ランナーは `data/` とコーパスのリンク先を読み取り専用にしてからテストを起動する（`tests/suite_guard.py`）。保護できない環境では起動を拒否する。`AUDIO_DB_TEST_UNGUARDED=1` は、ユーザーが保護なしの実行を明示したときだけ使う。
- 単体の `unittest` に保護を付けるときは `python -m tests.suite_guard COMMAND` を前に置く。
- SKIP と NOT_RUN は PASS ではない。チェックポイントや公式ツリーが無いテストは skip、CUDA や user namespace が無いテストは NOT_RUN になる。
- 並列数、GPU の選び方、環境変数、結果の読み方は `docs/TESTING.md`。

## 検証の区別

| 呼び方 | 言えること |
|---|---|
| Mock | 配線、集計、再開、設定の拒否 |
| 実モデルのスモーク | その環境と入力で形状と有限値が正しい |
| 公式一致 | 記録したモデル、長さ、sample rate、device、seed での最大絶対誤差 |
| DDP と障害 | 小さな設定で、rank の例外、kill、OOM、停止のあとの状態と、再開の一致。CPU は Gloo、GPU は NCCL |

未計測の条件での一致や、論文スコアの再現とは書かない。TSUBAME の実ジョブでの動作を、研究室サーバーのテストから言わない。
