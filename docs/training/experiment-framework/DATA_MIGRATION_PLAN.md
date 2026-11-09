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
| 3-D | `Dfn3IndexSampler`、`Dfn3Protocol`、`SemambaProtocol`。world size 1。`group_size` の規則は公開 API のとおり | 実装済み。記録は [PHASE_3D_ONLINE.md](PHASE_3D_ONLINE.md)。公式素材の blocked dataset は開いていない |
| 3-E | world size 2。検証分担。prefetch スレッド。worker 数を変えても realization が同じこと | 実装済み。記録は [PHASE_3E_DISTRIBUTED_DATA.md](PHASE_3E_DISTRIBUTED_DATA.md)。CPU / Gloo で測った。NCCL は使っていない |
| 3-F | 平坦 Sampler の `set_sampling_weights`。抽選は下の `weighted_categorical_v1`。試験ベクトルもそこに固定した | 実装済み。記録は [PHASE_3F_WEIGHTED_SAMPLING.md](PHASE_3F_WEIGHTED_SAMPLING.md)。chunk と DFN3 には掛けない |
| 4 | 現行四モデルの Trainer と、固定ペア DataModule を使った MP-SENet の 1 step。来歴 JSON | 既存の Phase 4 回帰。DataModule の未接続を、その記録の前提から外す |

3-B の時点では SE の `build_model` を呼ばない。合成スナップショットまたは既存の試験 fixture で、波形が `render_sample` と一致することまでである。実在の大規模 snapshot を必須にしない。fixture が無い環境では、その試験を skip と記録する。skip は成功ではない。

四モデルの回帰記録は Phase 4-F で終わり、DeepFilterNet3 の Trainer は Phase 4-G で接続した。3-D の時点ではデータ供給の試験であり、モデルの学習開始ではなかった。SEMamba++ の Trainer はまだ作らない。

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

## Phase 3-F の抽選（実装前に固定）

研究用 policy の名前は `weighted_categorical_v1` である。公式の `rank_order` ではない。重みを渡していない epoch は、これまでどおり snapshot の policy と `rank_order` を使い、`weights_sha256` は `"none"` のままである。

### 数学

学習サンプルを、固定ペアは `train.cases` の並び、SEMamba++ は発話 index `0 .. count-1` の並びに置く。この並びの位置 `i` の Sample ID へ、呼び出し元が渡した重み `w_i` を付ける。

実効重みは `ids.digest` と同じ 6 桁である。`format(float(w), ".6f")` の文字列を数値に戻し、その文字列の整数部・小数部からマイクロ重み `round_6(w) * 1_000_000` を正確な整数として作る。抽選と hash は両方この値を使う。6 桁で 0 になった重みは抽出しない。マイクロ重みがすべて 0 の表は拒否する。

正のマイクロ重みは、それらの最大公約数で割ってから使う。比率が同じ整数列は、同じ抽選列になる。hash に入れるのは割る前の実効重み表である。

正の重み `v_i` について

\[
P(i)=\frac{v_i}{\sum_j v_j}
\]

復元抽出する。抽選回数は、重みが 0 のサンプルを含めた学習サンプル数 `n` である。重みを変えても、この回数は変えない。正の重みが 1 件でも、`n` 回まで復元で埋める。

### 乱数

生成器は `splitmix64-v1` である。

```text
z = (x + 0x9E3779B97F4A7C15) mod 2^64
z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9 mod 2^64
z = (z xor (z >> 27)) * 0x94D049BB133111EB mod 2^64
return z xor (z >> 31)
```

ストリームの初期値は `mix_seed("weighted_categorical_v1", training_seed, epoch)` である。epoch ごとに別の列になる。rank は種に入れない。全 rank が同じ全域列を計算する。

`k` 番目の 64 bit は `splitmix64((stream + k) mod 2^64)` で、`k` は 1 から増やす。総和 `T` に対して、必要な bit 幅を 64 の倍数に切り上げ、その整数が `floor(2^width / T) * T` 未満のときだけ採用する。棄却抽出なので剰余の偏りは使わない。採用した値 `mod T` を、正の重みをサンプル順に積んだ累積和へ写す。

### 分担

全域列の長さは `n` である。rank 分担の名前は `stride_drop_tail_v1` である。`rank_mod_fixed_v1` でも `per_rank_independent_v1` でもない。

`usable = (n // world_size) * world_size` とし、末尾 `n % world_size` 件を捨てる。rank `r` は残った列の `r, r+world_size, r+2*world_size, ...` を受け取る。world size 1 は全域列をそのまま使う。

そのあとの micro-batch と group size の端数は、既存の `MicroBatcher` が捨てる。重み付きモードは DFN3 の短い最終 batch を残す規則を使わない。`n < world_size` で rank の列が空になるときは `DataError` である。

同じ重み、seed、epoch、world size なら、同じ Sample ID 列になる。重みがすべて 1 でも復元抽出なので、非復元の `rank_order` と同じ列にはしない。

### Epoch 境界

`SnapshotEpochView.close()` を終了契約にする。GC では閉じない。`close` は二度呼んでも成功する。閉じた view の `batch` は `DataError` である。

`set_sampling_weights` は、開いている view が 1 つでもあれば `DataError` で、その view の計画は変えない。計画を作った epoch は凍結する。重みの変更は、まだ計画を作っていない epoch から効く。凍結した epoch の計画キャッシュを、新しい重みで作り直さない。先読みキャッシュは重み変更時に捨て、新しい計画の batch に古い配列を返さない。検証の列と波形は重みを読まない。

`StandardTrainer` は、`train_epoch` が返した view を、その step の正常終了、中断、例外の `finally` で `close` する。`close` が無い DataModule は何もしない。

`state_dict` の `weights` は、`epoch` フィールドの計画を作った表である。まだ次 epoch に効いていない変更は、その `epoch` の `weights` にしない。未設定は `null`。設定時は Sample ID から 6 桁の実効重みへの写像である。`weights_sha256` はその写像の `ids.digest` で、未設定は `"none"` である。重み付き epoch の `order_policy` は `weighted_categorical_v1`、`order_params` はアルゴリズム名、`splitmix64-v1`、復元抽出、`training_count`、`stride_drop_tail_v1`、scale `1000000`、training seed である。`identity()` の `order_policy` は snapshot の policy のままで、重みでは変えない。

再開は、実行中の DataModule に既についている重み表と checkpoint の `weights` が一致するときだけ計画を再計算する。checkpoint の表で上書きしない。`initialize_from` は data 状態を戻さない。

world size が 2 以上で process group があるとき、epoch 計画を作る全 rank が `weights_sha256` を集め、1 つでも違えば `DataError` である。rank 0 から他 rank へ重みを配る処理は作らない。片 rank が計画に入る前に失敗したときは、process group の timeout で待つ側も戻る。

### 試験ベクトル

Sample ID は `s0, s1, s2, s3` の順である。seed は 7。実効重みがすべて 1 の epoch 0、world size 1 は `s1, s0, s3, s2` である。同じ列の world size 2 は rank 0 が `s1, s3`、rank 1 が `s0, s2` である。これは非復元 permutation と同じである必要はない。

重み `s0=1, s1=3, s2=0, s3=1` の epoch 0、world size 1 は `s1, s1, s3, s1` である。同じ表の epoch 1 は `s1, s1, s1, s3` である。world size 2 の epoch 0 は rank 0 が `s1, s3`、rank 1 が `s1, s1` である。`s2` は出ない。同じ表で 32 回引くと `s1, s1, s3, s1, s3, s3, s0, s3, s1, s1, s0, s1, s1, s1, s3, s1, s1, s1, s1, s1, s1, s1, s3, s3, s0, s3, s3, s1, s1, s3, s0, s1` で、やはり `s2` は無い。2 倍した `2, 6, 0, 2` は epoch 0 の 4 件と一致する。hash は `{"s0": 1.0, "s1": 3.0, "s2": 0.0, "s3": 1.0}` の digest `c5d7db2a48e3978848a888a328c3a497a5eb6320c8674e9e34b52af7d24bfd7c` である。すべて 1 の digest は `47aa23d5bc83a82b29858e45a2fa748c105766822191c9c7ec64807da5e55c42` である。

件数 7、重みすべて 1、seed 7、epoch 0、world size 2 の全域列は index `5, 6, 6, 5, 6, 1, 1` である。末尾 1 件を捨て、rank 0 は `5, 6, 6`、rank 1 は `6, 5, 1` である。

分布の受け入れは、結果を見る前に次で固定する。サンプル 2 件、重み `1:3`、seed 11、epoch 0、復元 4000 回。index 0 の回数は 918 以上 1082 以下である。これは `n*0.25 ± 3*sqrt(n*0.25*0.75)` を整数へ切り上げた区間である。この区間は決定的な ID 列の試験とは別である。

## ロールバック

3-B 以降のロールバックは、`src/experiments/data/` パッケージと、その試験と、`StandardTrainer` に足した device 移動を戻すことである。`src/reproduction/` に足した公開委譲は、呼び出しが無くなるなら残してもよいが、式の変更は戻す。DB と snapshot は、この移行のために削除も再作成もしない。

## 旧 Phase 3 行の扱い

[MIGRATION_PLAN.md](MIGRATION_PLAN.md) の Phase 表にあった「DataModule アダプタで既存関数を包み、MP-SENet を 1 step 進める」は、3-B から 3-E の供給実装と、Phase 4 の 1 step に分ける。アダプタクラスは作らない。1 step の時期を後ろへ移す理由は、供給の契約が無い状態でモデルを接続すると、試験が `TensorPairData` の形に戻ってしまうためである。
