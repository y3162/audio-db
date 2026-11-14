# テスト

実行方法、本番データの保護、並列実行の調整、結果の読み方を書く。エージェントが守る制約は [.agents/OPERATIONS.md](../.agents/OPERATIONS.md)。

## 実行

```bash
make test-cpu     # CPU だけ。CUDA は隠す
make test         # CPU に加え、空いている GPU があれば GPU レーンも実行する
```

どちらも `scripts/tests/run_suite.py` を呼ぶ。`tests/asr`、`tests/data`、`tests/evaluation`、`tests/experiments`、`tests/se`、`tests/setup`、`tests/numerics` のモジュールをプロセスプールで並列に実行する。
時間のかかるモジュールは先に起動し、他のワーカーが先に遊休になる時間を減らす。

ディレクトリ単体、モジュール単体は `unittest` をそのまま使う。本番データの保護を付けるときは `tests.suite_guard` を前に置く。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m tests.suite_guard \
  .venvs/se/bin/python -m unittest discover -s tests/experiments -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m tests.suite_guard \
  .venvs/se/bin/python -m unittest tests.experiments.test_checkpoint -v
```

`tests.suite_guard` を付けない `unittest` は保護なしで走る。スレッドは起動前の `OMP_NUM_THREADS` に従う。

## 本番データの保護

テストは `data/` の下に書かない。本番の DB、staging、スナップショット、コーパス、checkpoint、実験結果は読むだけである。書き込みは `tempfile` の一時ディレクトリで行うので、並列のテスト同士は書き込み先を共有しない。モジュール間の実行順の制約は無い。

ランナーは、この規則をカーネルで強制する。テストを 1 つも起動する前に、自分を新しい user namespace と mount namespace に入れ、`data/` と、その中のシンボリックリンクの先（コーパス）を読み取り専用で bind mount し直す。テストのプロセスはすべてこの中で動くので、本番データへの書き込みは `EROFS` で失敗する。uid と gid は変わらず、root 権限は使わない。ランナーの外のプロセスからは、何も変わって見えない。

```text
data_guard=read-only guarded_paths=15
```

- 保護できない環境（user namespace が無効など）では、ランナーはテストを起動せずに終了コード 2 で止まる。保護なしで走らせるときだけ `AUDIO_DB_TEST_UNGUARDED=1` を付ける。そのとき出力は `data_guard=off` になり、要約 JSON にも残る。
- 保護するのは `data/` と、`data/` から深さ 2 までのシンボリックリンクの先である。同じファイルへ別のマウントやパスから届く経路は対象外である。リンク先の親ディレクトリ（コーパス置き場の根）も対象外である。
- 本番の DB を作る処理は、テストではなく Make ターゲットにある（`make reproduction-register-p8`、`make reproduction-register-p9` など）。

`tests/evaluation/test_suite_guard.py` が、保護対象の列挙、保護下での書き込み拒否と読み取り、権限が増えていないこと、保護に失敗したランナーがテストを起動しないことを確かめる。

## GPU レーン

CPU のモジュールは `CUDA_VISIBLE_DEVICES=""` で走る。CPU の成功は GPU の成功ではない。

GPU レーンは、親プロセスの `CUDA_VISIBLE_DEVICES` の範囲で、使用メモリ 1024 MiB 未満かつ計算プロセスの無い GPU があるときだけ走る。変数が未設定なら全物理 GPU が範囲で、空文字なら GPU レーンは NOT_RUN になる。対象は、CUDA / NCCL の `tests/experiments/test_cuda_amp.py`、`tests.experiments.test_launcher.LauncherCudaTests`、`tests/numerics/test_model_tasks.py`、および GPU が要る実モデルの少数のテストである。2 枚必要なテストは、見える GPU が 1 枚なら NOT_RUN である。

ランナーは選んだ物理番号を `CUDA_VISIBLE_DEVICES` と `AUDIO_DB_CUDA_GPU0` / `AUDIO_DB_CUDA_GPU1` に入れる。CUDA の子プロセスは後者を自分の `CUDA_VISIBLE_DEVICES` に書き戻す。

## 調整

| 環境変数 / 引数 | 意味 | 既定 |
|---|---|---|
| `AUDIO_DB_TEST_WORKERS` / `--workers` | 同時に走らせるモジュール数 | affinity と cgroup の CPU 上限の小さい方から決める |
| `AUDIO_DB_TEST_THREADS` / `--threads` | 1 プロセスのスレッド数 | 同上。上限 8 |
| `AUDIO_DB_TEST_GPUS` / `--gpus` | 使う GPU の物理番号。親の可視範囲の外は選ばない。明示したときは空き確認を省く | 空いている GPU |
| `AUDIO_DB_TEST_GPU_SLOTS` | 同じ GPU 群で同時に走らせる GPU モジュール数 | 1 |
| `AUDIO_DB_TEST_TIMEOUT` / `--timeout` | 1 モジュールの上限秒 | 1800 |
| `AUDIO_DB_DFN3_WORKERS` | DeepFilterNet3 の分布確認のプロセス数 | 利用可能な CPU をワーカー数で割った数 |
| `AUDIO_DB_TEST_UNGUARDED=1` | 本番データの保護を外す | 保護する |
| `--summary PATH` | 要約 JSON の出力先 | 書かない |

明示したワーカー数とスレッド数はそのまま使う。`os.cpu_count()` は、affinity も cgroup も取れないときの代替である。GPU レーンも走るときは、CPU ワーカーと GPU スロットでスレッド幅を分ける。`deterministic: true` の実験は `torch.set_num_threads(1)` にする。`AUDIO_DB_TEST_GPUS` の整数と `GPU-` UUID は物理 GPU として扱う。MIG やそれ以外の表記は別の GPU に読み替えず、GPU レーンを NOT_RUN にする。

## 結果の読み方

| 表示 | 意味 |
|---|---|
| `ok` | 実行して通った |
| `skip` | チェックポイントや公式ツリーが無いなどで実行していない。PASS ではない |
| `not_run` | CUDA が要るのに見える GPU が無い、user namespace が無い、など。理由が `NOT_RUN` で始まる skip。PASS ではない |

| 種類 | 置き場 | 言えること |
|---|---|---|
| Mock | `tests/evaluation`、`tests/experiments` の多く | 配線、集計、再開、設定の拒否 |
| 実モデルのスモーク | `tests/se`、`tests/asr`、`tests/evaluation` の一部。チェックポイントが要る | その環境と入力で形状と有限値が正しい |
| 公式一致 | `tests/numerics`、`tests/numerics/reference/`（`.venvs/official` と `src/se/official/` が要る） | 記録したモデル、長さ、sample rate、device、seed での最大絶対誤差 |
| DDP と障害 | `tests/experiments/test_launcher.py`、`test_ddp.py`、`test_cuda_amp.py`、`test_phase6_interrupt.py` | 小さな設定で、rank の例外、kill、OOM、停止、猶予切れのあとの状態と、再開した結果が中断なしの実行と一致すること |

`test_launcher.py` の CPU のテストは Gloo の 2 rank、`LauncherCudaTests` は GPU 2 枚の NCCL である。どちらも 1 ノードの小さな設定であり、TSUBAME の実ジョブ、複数ノード、本番規模の学習は含まない。

これらを混ぜて「検証済み」と書かない。未計測の条件での一致や、論文スコアの再現とは書かない。

## モデル構造

仕様は [モデル](experiments/MODELS.md)。`tests/experiments/test_model_architecture.py` が、6 モデルの公式 YAML の構築、研究用の層数・幅・Discriminator 解像度の反映、未知キーと矛盾の拒否、`config_sha256` の変化、小さいネットの forward と backward を見る。公式初期化と 1 step の数値は各モデルの既存テストが既存の許容差で比べる。

## 損失グラフ

仕様は [損失](experiments/LOSS.md)。`tests/experiments/test_loss_objectives.py` が、6 モデルの YAML の係数と加算順、拒否、`config_sha256`、MP-SENet の項と勾配の `torch.equal`、DeepFilterNet3 の既存 oracle との一致、TF-GridNet の Trainer を通した研究用項の追加を見る。公式 Trainer の 1 step、蓄積、Gloo、再開は各 `tests/experiments/test_<model>.py` が、注入した Objective のまま既存の許容差で比べる。GPU を隠した成功は GPU の成功ではない。

## 公式実装との比較スクリプト

`tests/numerics/reference/` のスクリプトは `.venvs/official` で動かす。出力先は `AUDIO_DB_NUMERICS_OUT`、未設定なら `/tmp/audio-db-official-compare` である。`data/results/` には書かない。MossFormerGAN の比較は、スレッド数を 1 にしないと MKL-DNN の非決定性で差が出る（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）。記録済みの誤差は [公式出力との数値比較](se/VALIDATION.md)。
