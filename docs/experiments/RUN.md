# 学習・実験

学習の入口は `src/experiments/` だけである。`enhance()` は推論のままである。この文書は起動、再開、出力の見方の手順を書く。

| 知りたいこと | 文書 |
|---|---|
| Latest、Archive、Selector、再開と初期化の条件 | [Checkpoint](CHECKPOINT.md) |
| プロセスの監視、終了状態、停止、`RUNNING` の生死 | [Launcher](LAUNCHER.md) |
| 検証指標による停止と設定例 | [停止ポリシー](STOPPING.md) |
| 式、Optimizer の順序、変えてはいけない条件 | [.agents/ARCHITECTURE.md](../../.agents/ARCHITECTURE.md) |

## 設定の置き場

| 場所 | 用途 |
|---|---|
| `configs/reproduction/experiments/official/` | 公式忠実条件。`experiment_claim: official_faithful`。検証に公式 test を使う設定は独立な最終評価ではない |
| `tests/fixtures/configs/experiments/` | CPU で 1 logical step などを確認する宣言。本格学習の設定ではない |
| `configs/data/` | Dataset、Protocol、Build、公式リスト。ID はファイルの置き場所ではなく内容から決まる |
| `configs/db/init.yaml` | 評価用 DB の名前付きセット。学習の入力には使わない |

計画上の更新回数と、実行後の `metrics.jsonl` にある logical step・optimizer 更新は別である。未実行の本学習の実績は推定で埋めない。実行ディレクトリの集計は `src.experiments.budget.observed_budget` が読む。Early Stopping で止まっても、書かれた metrics と status がその実績である。

YAML の `class` には公開 import パスを書く。公開境界と checkpoint のクラス識別は [Checkpoint](CHECKPOINT.md#公開パスとクラス識別) を参照する。

`official/` に無い設定を、パスを変えただけでは公式再現と呼ばない。`dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked である。

VoiceBank の公式設定は `data/db/snapshots-voicebank-float-v1/` の snapshot を使う。旧 PCM16 変換の snapshot は新しい仕様では読み込めない。実素材から新たに準備する際は、既存の DB・staging・snapshot を上書きせず、次のように空いている別のパスを指定する。

```bash
.venvs/se/bin/python -m src.experiments.data.cli prepare --step all \
  --database data/db/reproduction-voicebank-float-v1.duckdb \
  --staging data/staging/reproduction-voicebank-float-v1 \
  --snapshots data/db/snapshots-voicebank-float-v1 \
  --resource-db data/db/resource.duckdb
```

旧 checkpoint の再開互換性は保証しない。

## 実行

Preflight はグラフを検証し、Optimizer は更新しない。`make experiment` は Launcher 経由で学習を始める。

```bash
make experiment-preflight CONFIG=tests/fixtures/configs/experiments/mp_senet_gan.yaml
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml NPROC=2
```

同じことは直接でも実行できる。

```bash
.venvs/se/bin/python -m src.experiments.cli --config CONFIG --preflight
.venvs/se/bin/python -m src.experiments.launcher --config CONFIG
CUDA_VISIBLE_DEVICES=0,1 .venvs/se/bin/python -m src.experiments.launcher --config CONFIG --nproc 2
```

DDP は `--nproc` を world size にする。学習する module の `prepare` に `distribute: ddp` が要る。`runtime.kwargs.world_size` を書いた設定は、その値と `--nproc` が違うと拒否する。CUDA では rank ごとに GPU を 1 枚使い、2 つの rank が同じ GPU を使う起動は拒否する。`CUDA_VISIBLE_DEVICES` で隠すと、見える番号は 0 から振り直される。実験 runtime は、指定された CUDA デバイスを他プロセスの使用率や使用メモリでは拒否しない。CUDA が使えない、見えるデバイスが無い、`LOCAL_RANK` が範囲外、world size がデバイス数を超える起動は拒否する。実際の CUDA OOM はエラーとして記録する。Launcher はデバイス全体の使用メモリを `gpu_memory.jsonl` に追記する。これは起動の可否には使わない。

ChunkSampler の `global_batch` は `micro_batch × world_size × group_size` と一致させ、`group_size` は Trainer の勾配蓄積回数と一致させる。チャンクの計画は実効 batch 単位で作り、各 rank の計画を micro-batch に分けて蓄積する。world size が変わるとデータの順序も変わるため、実効 batch が同じでも別の実験である。

`python -m src.experiments.cli` を直接呼んでも学習できる。その場合は、プロセスが kill されたときの判定と後始末をする者がいない。学習は Launcher から起動する。

ジョブスクリプトからは `scripts/experiments/job.sh` を使う。TSUBAME での投入は [TSUBAME 4.0](../envs/TSUBAME.md)。

```bash
scripts/experiments/job.sh CONFIG 2
AUDIO_DB_RESUME=OUTPUT_DIRECTORY AUDIO_DB_STOP_AFTER=1200 scripts/experiments/job.sh CONFIG 2
```

## 終了の見方と再開

Launcher の終了コードは、0 が完了、3 が安全な停止、1 が失敗である。

```bash
.venvs/se/bin/python scripts/experiments/show_status.py OUTPUT_DIRECTORY
```

| 状態 | 次にすること |
|---|---|
| `COMPLETED` | 終わり。`complete.json` がある。再開はできない |
| `INTERRUPTED` | `--resume OUTPUT_DIRECTORY` で続ける |
| `FAILED` | `status.json` の `reason` と `failures`、ジョブのログで原因を確かめてから、`--resume OUTPUT_DIRECTORY` で続ける |
| `RUNNING` で `liveness` が `stale` | 記録されたプロセスは残っていない。`--resume OUTPUT_DIRECTORY` で続ける |
| `RUNNING` で `liveness` が `alive` | 実行中。触らない |
| `RUNNING` で `liveness` が `unknown` | この環境では生存を確定できない。起動したホストで pid を確認する |

```bash
.venvs/se/bin/python -m src.experiments.launcher --config CONFIG --resume OUTPUT_DIRECTORY
make experiment CONFIG=... RESUME=OUTPUT_DIRECTORY
```

ディレクトリを渡すと `latest.pt` から続く。`--resume` を付けない起動は、`complete.json`、`latest.pt`、`archive/*.pt`、以前の形式の `step_*.pt` のどれかがあるディレクトリを拒否する。続きは `--resume`、やり直しは別の `output.directory` で行う。

再開は設定の違いを拒否する。`stop_after_logical_steps` で止めた実行は、その上限だけを外した設定で再開できる。Selector のクラスと設定も一致させる。拒否の条件は [Checkpoint](CHECKPOINT.md)。

## 保存の方針を変える

YAML の `checkpoint.selector` に Selector を書く。省くと `StandardSelector` で、Latest は 1000 logical step ごとと、境界、安全な停止、完了で置き換える。毎 step は `latest_every: 1` である。`PeriodicArchiveSelector` と `TopKSelector` は、`latest_every` を省くと 1 のままである。

```yaml
checkpoint:
  selector:
    class: src.experiments.selectors.PeriodicArchiveSelector
    kwargs:
      latest_every: 1000
      archive_every: 10000
      keep_last: 5
```

Best や Top-K を再開できる形で残すときは `TopKSelector`、独自の規則は `Selector` を継承したクラスを書く。

## 既存の checkpoint から新しい実験を始める

元の実験を続けるのではなく、重みなどを読んで別の実験を始めるときは、新しい `output.directory` と `initialize` を書く。

```yaml
initialize:
  checkpoint: data/results/experiments/base-run/latest.pt
  states: [generator]
```

選べる state と拒否される組み合わせは [Checkpoint](CHECKPOINT.md)。元の実験のファイルは変更しない。

## ディレクトリ

| 場所 | 責務 |
|---|---|
| `src/experiments/api.py`、`cli.py`、`config.py`、`errors.py` | 公開入口。YAML の読み込みと実行 |
| `src/experiments/launcher.py` | プロセスの起動、監視、終了状態の記録 |
| `src/experiments/selectors.py` | Selector の公開 import パス |
| `src/experiments/core/` | コンポーネントの生成、依存の解決、状態の登録 |
| `src/experiments/execution/` | CPU/GPU/DDP、checkpoint、Selector、再開、中断、来歴、実行状態、プロファイリング |
| `src/experiments/training/` | 共通の Trainer と損失 |
| `src/experiments/official/<model>/` | そのモデルの公式 Model、Trainer、参照ステップ |
| `src/experiments/calc/` | 共有する STFT、スペクトル損失、scheduler、RNG |
| `src/experiments/data/` | スナップショットとオンデマンド描画。学習ループは持たない |

直下の `runtime.py` や `trainers.py` などは公開 import パスである。実装は上のディレクトリにあり、公開パスは同じオブジェクトを返す。YAML の `class` は公開パスのまま書く。

```yaml
class: src.experiments.runtime.ProcessRuntime
```

```yaml
class: src.experiments.trainers.StandardTrainer
```

研究用の手法を足すときは、公式再現の Model、Trainer、参照ステップを変更せず、独立した領域に置く。

## コンポーネント

公式手法の Model、Trainer、参照ステップは `src/experiments/official/<model>/` にある。YAML は既存の `class`、`inject`、`ref`、`parameters_of` でつなぐ。

```yaml
class: src.experiments.official.mp_senet.trainer.MPSENetGANTrainer
```

```yaml
class: src.experiments.official.mp_senet.models.MPSENetGANGenerator
```

DB の準備は学習 CLI とは別である。

```bash
.venvs/se/bin/python -m src.experiments.data.cli doctor --database data/db/reproduction.duckdb
```

## 出力の見方

- `metrics.jsonl`: logical step ごとの損失
- `status.json`、`launch.json`、`failure.rank<N>.json`: 実行状態と失敗の記録
- `latest.pt`、`archive/*.pt`: `audio-db-experiment-v1` の checkpoint
- `gpu_memory.jsonl`、`gpu_memory.json`: Launcher が書いた、デバイス全体の GPU メモリの時系列と最大値。`CUDA_VISIBLE_DEVICES` が空、または間隔が 0 のときは無い。見方は [Launcher](LAUNCHER.md)
- `data_provenance.json`: Dataset、snapshot、使用集合の内容ハッシュ。rank 0 が一度だけ書く
- `best_generator.pt`、`valid.loss.best.pt` など: 公式手順の推論用の重み。再開には使えない

全ファイルの一覧は [Checkpoint](CHECKPOINT.md) の「出力ディレクトリ」。

`selection_uses_test` が true の checkpoint は、評価素材が離れていても independent にしない。来歴が無い checkpoint の評価は `UNKNOWN` である。
