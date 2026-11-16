# 環境とテスト

依存の正本は `requirements/<role>.lock.txt`。`make setup` は DB を初期化しない。

## Codegraph

構造の把握、シンボル探索、呼び出し経路、編集前の影響範囲の手順。優先順位は `AGENTS.md` の「コードベースの調査（Codegraph）」。

| いつ使う | 例 |
|---|---|
| 仕組み・フロー | Launcher から Trainer まで、評価 CLI のパイプライン |
| シンボル・ファイルの所在 | `transcribe`、`enhance`、`build_model`、特定の Trainer |
| 編集前 | 変える関数の呼び出し元・先、同ファイル外の依存 |

**MCP と CLI**

- 第一候補は MCP の `codegraph_explore`。接続名は Cursor と Codex で異なる。使えるツールを見て呼び、名前空間は固定しない。
- MCP が無い、またはその場で使えないときは `codegraph explore -p <リポジトリルート> <query...>`。出力は MCP の `codegraph_explore` と同じである。
- どちらも適切に調査できないときは、下のフォールバックに移る。

**`projectPath` と `query`**

- MCP の `projectPath`、CLI の `-p` / `--path` には、リポジトリルートの絶対パスを渡す。Codegraph はそのパス以上で最も近い `.codegraph/` を使う。
- `query` は自然文、シンボル名、ファイル名、関連キーワード。フローでは経路に関わる名前を複数指定する（例: `launcher runtime checkpoint resume`）。
- 原則として少数回で足りる。足りないときだけ語を足すか絞って再探索する。

**結果の扱い**

- 返ったシンボル単位のソースは Read 相当として、同じ箇所をすぐ開き直さない。
- 正確性の検証、編集に必要な前後の文脈、ディスク上の最新ソースとの確認は行ってよい。食い違うときはソースを優先する。
- 呼び出し関係や影響範囲を、この結果だけで完全だと断定しない。ファイルをまたぐ解決は名前の照合であり、曖昧な呼び出しは複数候補になり得る。

**フォールバック**

次のときは Read / Grep / Glob を使う。

- MCP が使えず、CLI でも適切に調査できない
- `.codegraph/` が無い
- 結果が空、不完全、または不正確
- 動的ディスパッチ、Registry、Builder、YAML 設定など、静的解析だけでは追えない
- 編集や検証に、返却に無いソースが要る

**インデックス**

- 通常の変更は、MCP の file watcher による自動同期に任せる。反映は書き込みから少し遅れる。
- `codegraph init` とフル再構築の `codegraph index` は、ユーザーが明示したときだけ実行する。インデックスが無ければ作らない。
- 自動同期が動かないとき（watcher 無効、`projectPath` 経由で live watcher が無い、など）は、既存インデックスへの増分同期 `codegraph sync` を使ってよい。
- 古い可能性があるとき、および応答が未同期や auto-sync 無効を示しているときは、最新のソースを優先する。

| 環境 | 用途 |
|---|---|
| `.venvs/se` | Python 3.12、torch 2.6.0+cu124。SE、評価、実験、通常のテスト |
| `.venvs/asr` | NeMo、Whisper、Transformers。実 ASR |
| `.venvs/official` | 公式実装との比較 |
| `.venvs/semambapp` | SEMamba++ の fused CUDA |
| `.venvs/deepfilternet` | 公式 `libdf`。独立実装の実行時依存ではない |

ASR-guided SE の TDT 損失と SEMamba++ を同じプロセスで動かす環境は、この checkout には作っていない。`.venvs/asr` の numba は 0.61.0、`.venvs/se` の numba は 0.68.1 で、lock を足し合わせない。公式 venv は変更しない。実 Parakeet は `.venvs/asr`、SE の学習確認は `.venvs/se` で分ける。このホストの `/usr/local/cuda` は 11.0 で、Numba 0.61 はそのままでは TDT カーネルをコンパイルしない。`.venvs/asr` に入っている `nvidia/cuda_nvcc` と `nvidia/cuda_runtime` を `CUDA_HOME` と `LD_LIBRARY_PATH` で見せると、A100（8.0）向けにコンパイルできる。venv 自体は変えない。

`src/se/official/` は Git 管理外。独立実装はここを import しない。テストの公式比較だけが読む。

## GPU

- 使う前に `nvidia-smi` を見る。実験 runtime は、`CUDA_VISIBLE_DEVICES` で指定された CUDA デバイスを他プロセスの使用率や使用メモリでは拒否しない。CUDA が無い、デバイスが無い、rank が範囲外、world size がデバイス数を超える起動は `PrepareError` にする。
- `CUDA_VISIBLE_DEVICES` で隠すと、見える番号は 0 から振り直される。未設定と空文字は、見える番号を物理番号のまま見る。UUID と MIG の文字列は、別の物理 GPU の番号に読み替えない。
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
