# Data System の移行

Phase 3-A は文書だけである。`src/`、`tests/`、学習 YAML、`src/se/official/`、DB、snapshot、素材は変えない。現行の公式再現は `python -m src.training.se.cli` のまま実行する。新経路の結果を、現行経路の再現結果と呼ばない。公式実装との数値一致は、この計画のどの Phase でも未測定のまま主張しない。

構成は [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md)、契約は [DATA_PUBLIC_API.md](DATA_PUBLIC_API.md)、試験は [DATA_ACCEPTANCE_CRITERIA.md](DATA_ACCEPTANCE_CRITERIA.md)。

## 再利用、改修、再実装、廃止

| 現行 | 判断 | 理由 |
|---|---|---|
| `src/reproduction/db/`、`prepare/`、`export_snapshot` | 再利用。移さない | 登録と学習を分ける案 B。学習は呼ばない |
| `load_snapshot`、`AssetStore`、`find_snapshot_dir` | 再利用 | DataSource が開く実装である |
| `rank_order`、`resolve_order_policy`、`order_params` | 再利用 | 平坦 policy の本体 |
| `mix_seed` と各 protocol の `sample_seed` | 再利用 | realization の seed 組 |
| `planner._crop` | 改修は公開名の委譲だけ。式は変えない | 新 Protocol が私有関数へ依存しないため。中身は `_crop` の呼び出し |
| `render_sample`、`transform.py`、`resample_pcm16` | 再利用 | 固定ペアの数値 |
| `plan_chunk_epoch`、`iterate_chunks` | 再利用。写さない | cache と shard が一つの iterator であるため |
| `render_realization`（MS-SNSD と DNS） | 再利用 | 混合式は build モジュールに残す |
| `plan_online_epoch`、`render_mixture` | 再利用 | DFN3 の並びと混合。並びは Sampler、混合は Protocol |
| `plan_semamba_epoch` | 試験の比較対象にだけ使う | 本番の並びは `rank_order("per_rank_independent_v1")`。試験が添字列の一致を見る |
| `render_example` | 再利用 | SEMamba の劣化順 |
| `assemble` の末尾 0 | 再利用するか、同じ pad を Batcher に 10 行以内で書く | 波形の値は pad の前に決まっている。試験は pad 後の最大絶対誤差 0 |
| `ReproductionTrainingData`、`OnlineDFN3TrainingData`、`OnlineSemambaTrainingData`、`EpochStream` | 現行 CLI では残す。新 DataModule はインスタンス化しない | 融合した供給面が新しい公開 API ではない |
| `OnDemandChunkData`、`DnsOnDemandChunkData` | 同上 | 再開状態が現行 Trainer にある。新経路は `ChunkSampler` と `state_dict` |
| `Trainer._chunk_state`、`Trainer._run_chunk_epochs` | 現行 CLI では残す。新経路へコピーしない | micro-batch 添字は Experiment の `update_index` に統一する |
| `LegacyTrainingData`、`RecipeDataset` | 残す。新研究の既定経路にしない | 評価用 DB を開く経路である |
| `src/training/data/sampler.py` の `epoch_indices` | 残す。新 Sampler からは呼ばない | `global_shuffle_stride_v1` が必要なら `rank_order` を使う |
| `ids.ALGORITHMS` と snapshot の rule | 再利用。学習側で第二の規則表を作らない | 未知 algorithm は `DataError` |
| `data_provenance.json` の形式 | 再利用。書くのはモデル接続後の Trainer | 評価は JSON だけを読む |
| `TensorPairData` と `Batch` の 4 フィールド | 改修は optional フィールドの追加と、モジュールからパッケージへの移動 | 既存 YAML のクラスパスを維持する |
| `audio-db-training-v1` | 廃止しない。新ローダは開かない | Phase 2 の契約 |
| 共通の混合エンジン | 作らない | 実装済みの方針のまま |

`src/reproduction/` の数値関数を、新パッケージへコピーして二つの正本にしない。呼び出す。公開名の委譲を足すときだけ、委譲先のファイルを編集する。

## 実装順

各段は、前段の受け入れが記録されてから始める。巨大な骨格を先に作らない。3-B が最初の実装である。

| 段 | 実装 | 完了の見え方 |
|---|---|---|
| 3-A | このディレクトリの設計書。コードは無い | 本計画の文書が揃っている。この段はそれで完了 |
| 3-B | `data.py` をパッケージへ移す。`SnapshotSource`、`OrderSampler`、`FixedPairProtocol`、`MicroBatcher`、`SnapshotDataModule`。固定ペアの world size 1。`StandardTrainer` の device 移動と `group_size` 検査。`close` は `Experiment.close()`。`DataError` | [DATA_ACCEPTANCE_CRITERIA.md](DATA_ACCEPTANCE_CRITERIA.md) の 3-B。GPU は使わない。現行 CLI のファイルは差分が無い。記録は [PHASE_3B_FIXED_PAIR.md](PHASE_3B_FIXED_PAIR.md) |
| 3-C | `ChunkSampler` と `ChunkProtocol`。MS-SNSD と DNS は描画関数の選択だけが違う。world size 1 の短い `num_samples_per_epoch` | chunk の受け入れ。公式規模の 360,000 件は走らせない。記録は [PHASE_3C_CHUNK.md](PHASE_3C_CHUNK.md) |
| 3-D | `Dfn3IndexSampler`、`Dfn3Protocol`、`SemambaProtocol`。world size 1。`group_size` の規則は公開 API のとおり | オンラインの受け入れ。公式素材の blocked dataset は開かない |
| 3-E | world size 2。検証分担。prefetch スレッド。worker 数を変えても realization が同じこと | DDP の受け入れ。NCCL の試験は Phase 2-C の GPU 条件に従い、使用メモリが 1024 MiB 未満の GPU だけを使う。CPU の Gloo で足りる項目は GPU を使わない |
| 3-F | 平坦 Sampler の `set_sampling_weights`。抽選アルゴリズムの名前と試験ベクトルを、実装のときにこの文書へ追記する | 重みの受け入れ。chunk と DFN3 には掛けない |
| 4 | 現行四モデルの Trainer と、固定ペア DataModule を使った MP-SENet の 1 step。来歴 JSON | 既存の Phase 4 回帰。DataModule の未接続を、その記録の前提から外す |

3-B の時点では SE の `build_model` を呼ばない。合成スナップショットまたは既存の試験 fixture で、波形が `render_sample` と一致することまでである。実在の大規模 snapshot を必須にしない。fixture が無い環境では、その試験を skip と記録する。skip は成功ではない。

DeepFilterNet3 と SEMamba++ の新 Trainer は、四モデルの回帰記録が終わるまで作らない、という現行の移行計画を維持する。3-D はデータ供給の試験であり、モデルの学習開始ではない。

## 並走

```text
python -m src.training.se.cli --config configs/training/reproduction/...
python -m src.experiments.cli --config configs/experiments/...
```

公式再現 YAML は移さない。同じ `output.directory` を両方から使うことは、現行どおり lock で拒否される。

新コードが `src.reproduction` の純関数を import した PR で、`.agents/PROJECT.md` の依存方向を更新する。3-A では更新しない。`reproduction` から `experiments` への import は足さない。

## 回帰

各実装段の最後に、次を実行する。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/experiments -v
.venvs/se/bin/python -m unittest discover -s tests/reproduction -v
```

`tests/reproduction` の失敗数が、その段の直前に記録した失敗数を上回らないこと。記録が無い状態で「回帰なし」と書かない。skip は成功に数えない。`tests/training` は、`src/training/` を変えた段だけ実行する。3-B が Trainer の device 移動だけを変える場合は、`tests/training` ではなく `tests/experiments` が対象である。`src/training/` は変えない。

公式処理との比較と、旧 audio-db との比較は別の記録にする。

| 比較 | 何を同じとみなすか | いつ |
|---|---|---|
| 新 DataModule と既存の純関数 | CPU、同じ引数、波形の最大絶対誤差 0 | 3-B から 3-E の受け入れ |
| 新 Trainer と現行 Task | 既存 Phase 4 の 1 update、CPU、`deterministic: true` | Phase 4 |
| 公式リポジトリの波形や学習曲線 | 既存の検証文書が既に分けている範囲だけ。未測定を成功にしない | この移行計画の完了条件には入れない |

## ロールバック

3-B 以降のロールバックは、`src/experiments/data/` パッケージと、その試験と、`StandardTrainer` に足した device 移動を戻すことである。`src/reproduction/` に足した公開委譲は、呼び出しが無くなるなら残してもよいが、式の変更は戻す。DB と snapshot は、この移行のために削除も再作成もしない。

## 旧 Phase 3 行の扱い

[MIGRATION_PLAN.md](MIGRATION_PLAN.md) の Phase 表にあった「DataModule アダプタで既存関数を包み、MP-SENet を 1 step 進める」は、3-B から 3-E の供給実装と、Phase 4 の 1 step に分ける。アダプタクラスは作らない。1 step の時期を後ろへ移す理由は、供給の契約が無い状態でモデルを接続すると、試験が `TensorPairData` の形に戻ってしまうためである。
