# 学習・実験

学習の入口は `src/experiments/` だけである。`enhance()` は推論のままである。checkpoint 形式は `audio-db-experiment-v1`。式、Optimizer の順序、再開を拒む条件は [.agents/ARCHITECTURE.md](../../.agents/ARCHITECTURE.md)。

## 設定の置き場

| 場所 | 用途 |
|---|---|
| `configs/reproduction/experiments/official/` | 公式忠実条件。`experiment_claim: official_faithful`。検証に公式 test を使う設定は独立な最終評価ではない |
| `configs/reproduction/experiments/variants/` | 公式との差分を明示した比較条件。`leakage_safe`、研究用の固定検証、smoke 素材、micro-batch 幅の違いを含む |
| `tests/fixtures/configs/experiments/` | CPU で 1 logical step などを確認する宣言。本格学習の設定ではない |
| `configs/data/` | Dataset、Protocol、Build、公式リスト。ID はファイルの置き場所ではなく内容から決まる |
| `configs/db/init.yaml` | 評価用 DB の名前付きセット。学習の入力には使わない |

`official/` に無い設定を、パスを変えただけでは公式再現と呼ばない。`dfn3.online.official@1` と `semambapp.gsr.official@1` は blocked である。

## 実行

Preflight はグラフを検証し、Optimizer は更新しない。`make experiment` は学習を始める。

```bash
make experiment-preflight CONFIG=tests/fixtures/configs/experiments/mp_senet_gan.yaml
make experiment CONFIG=configs/reproduction/experiments/official/mp_senet_gan.yaml
```

同じことは CLI でも実行できる。

```bash
.venvs/se/bin/python -m src.experiments.cli \
  --config configs/reproduction/experiments/official/mp_senet_gan.yaml \
  --preflight
.venvs/se/bin/python -m src.experiments.cli \
  --config configs/reproduction/experiments/variants/mp_senet_gan.yaml
```

CUDA を要求する設定は、使用メモリが空いていない GPU を拒否する。`runtime.kwargs.allow_busy_gpu: true` は、その GPU が自分のジョブだと分かっているときだけ使う。`CUDA_VISIBLE_DEVICES` で隠すと、見える番号は 0 から振り直される。

## Checkpoint と再開

出力先は設定の `output.directory` である。完成した checkpoint、`status.json`、`metrics.jsonl` を見る。`status.json` は `RUNNING` / `CHECKPOINTED` / `INTERRUPTED` / `COMPLETED` / `FAILED` を区別する。公式の終了だけが `complete.json` を書く。

```bash
.venvs/se/bin/python -m src.experiments.cli \
  --config configs/reproduction/experiments/official/mp_senet_gan.yaml \
  --resume data/results/experiments/phase5-production/mp-senet-gan-official/step_00000001.pt
```

`--resume` は `logging` 以外の設定差を拒否する。`config_sha256` は `output` と `stop_after_logical_steps` を除いた設定の SHA-256 で、YAML のパスはハッシュに入らない。意味の違うクラスパスは別の設定になる。公式コンポーネントへ移した Model と Trainer、および `WarmupReduceLROnPlateau` の移動前パスだけは、同じ設定として再開できる。`audio-db-training-v1` は読まない。

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
- `status.json`: 実行状態
- `data_provenance.json`: Dataset、snapshot、使用集合の内容ハッシュ。rank 0 が一度だけ書く
- `step_*.pt`: `audio-db-experiment-v1`

`selection_uses_test` が true の checkpoint は、評価素材が離れていても independent にしない。来歴が無い checkpoint の評価は `UNKNOWN` である。
