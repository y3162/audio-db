# Data System の受け入れ

Phase 3-A では試験を追加しない。この表は 3-B 以降の実装が、何を測れば完了と書いてよいかを固定する。未実行の行を成功と書かない。skip は成功ではない。公式リポジトリとの一致は、この表の期待に含まれない。固定ペアの Phase 3-B で測った行と、測っていない行は [PHASE_3B_FIXED_PAIR.md](PHASE_3B_FIXED_PAIR.md) に分けて記録する。

方法の「合成 snapshot」は、`tests/reproduction` が既に使っている小さな fixture を読む。新しい大規模データの生成はしない。`resource.duckdb` と `reproduction.duckdb` は開かない。

## Component 構築

- 方法: 固定ペアの YAML 例（[DATA_PUBLIC_API.md](DATA_PUBLIC_API.md)）を、合成 snapshot に向けて `build_experiment` する。`TensorPairData` の既存 YAML も同じプロセスで構築する。
- 期待: DataModule の `setup` が終わり、Source は開いている。`role: data` は DataModule だけで、Source と Sampler と Protocol と Batcher は StateRegistry の保存対象に入っていない。`from src.experiments.data import TensorPairData, Batch` が成功し、既存の線形実験テストが失敗しない。DuckDB を import したあとに接続していないことを、このプロセスの試験が `duckdb` の接続関数を差し替えて呼び出し回数 0 で確認する。
- Phase: 3-B

## Sample と Batch

- 方法: 固定ペアの `train_epoch(0, start_update=0).batch(0)` を 2 回呼ぶ。
- 期待: 2 回の `inputs` の最大絶対誤差が 0。`batch` の戻りは長さ 1。`sample_id` は `sample_ids` のカンマ連結。`lengths` の各要素は、パッド前の本数と一致し、`inputs.shape[-1]` 以下である。`SampleRef` に相当する配列オブジェクトは `Batch` のフィールドに無い。
- Phase: 3-B

## 固定ペア

- 方法: 同じ seed、epoch、world size 1、同じ case 列で、`plan_fixed_pair` が返す crop と case id の列を、新 DataModule の micro-batch 順と比較する。続けて、同じ `SampleRef` に対する `render_sample` の波形と `inputs` / `targets` を比較する。正規化は MP-SENet と MossFormerGAN を別々に行う。
- 期待: case id、crop の start と length、波形の最大絶対誤差がすべて 0。新コードが `ReproductionTrainingData` を構築していないことを、そのクラスのコンストラクタを差し替えて呼び出し回数 0 で確認する。
- Phase: 3-B
- 非目標: 公式 16 kHz 変換器との一致。`verify_on_read` が偽のときのファイル差し替え検出。

## chunk

- 方法: 合成の realization が数件の snapshot で、`num_samples_per_epoch` をその件数以下にし、world size 1、`shuffle=True` と `False` を別々に `plan_chunk_epoch` の `(case_id, start, stop)` と比較する。描画は同じ span の現行 `_pair` と比較する。MS-SNSD 用の描画関数と DNS 用の描画関数を、algorithm 名で 1 件ずつ選ぶ。
- 期待: span が一致し、波形の最大絶対誤差が 0。`group_size` が 1 でない YAML は `DataError`。`plan_chunk_epoch` を world size 1 に落とさず、引数の world size がプロセスの world size と一致する。
- Phase: 3-C
- 非目標: 公式の未 seed 生成とのビット一致。360,000 件の全計画。

## オンライン混合と劣化

- 方法: 合成の素材リストで、DFN3 は `plan_online_epoch` の添字と `render_mixture` の波形を、新 Sampler / Protocol と比較する。SEMamba は `rank_order("per_rank_independent_v1")` の添字と `render_example` の波形を比較する。world size 1。valid の波形を epoch 0 と epoch 1 で比較する。
- 期待: 添字と波形の最大絶対誤差が 0。valid の波形は epoch が違っても最大絶対誤差 0。DFN3 の valid の並びは epoch が違うと変わってよい。変わったかどうかは試験が添字列を記録し、波形一致と別に書く。`group_size` が 2 の DFN3 は `DataError`。SEMamba の `group_size` 2 は、micro-batch 列の長さが 2 の倍数になり、余った 1 件を捨てる。
- Phase: 3-D
- 非目標: Rust `StdRng`、公式のプロセス全体の乱数消費、論文の 24,000 サンプル設定。

## 数値的な処理一致

- 方法: 上の固定ペア、chunk、オンラインの各比較を、CPU で、`deterministic` の設定とは独立に記録する。データ供給の比較に GPU は使わない。
- 期待: 各記録に、呼んだ関数名、seed、epoch、world size、最大絶対誤差が書いてある。誤差 0 でない行は、その Phase を完了にしない。
- Phase: 3-B、3-C、3-D
- 非目標: その後のモデル forward のビット一致。Phase 2-C の AMP 試験の代替にしない。

## データ順序

- 方法: world size 1 で、epoch 0 と epoch 1 の `sample_ids` 列を保存する。MP-SENet の `fixed_single_shuffle_v1` と、MossFormer の `rank_mod_fixed_v1` と、SEMamba の `per_rank_independent_v1` を別々に行う。
- 期待: MP-SENet の world size 1 は epoch 0 と 1 の列が一致する（shuffle seed 1234 が epoch で変わらない）。MossFormer と SEMamba は、現行 `rank_order` の列と一致する。一致しない列を、公式の DataLoader に近いという理由で採用しない。
- Phase: 3-B（固定ペアの 2 policy）、3-D（SEMamba）

## Dataset Identity

- 方法: 同じ snapshot で `identity()` を 2 回呼ぶ。`dataset_id` または `snapshot_hash` を書き換えた `state_dict` を `load_state_dict` に渡す。name だけが違う二つ目のディレクトリは作らず、identity の比較キーに name が入っていないことを辞書の一致条件で確認する。
- 期待: 2 回の `dataset_id` と `snapshot_hash` は一致する。改竄した state は `StateError`。`require_profile: leakage_safe` の DataModule に `official_faithful` の snapshot を渡すと `DataError`。キーを省略した公式用の構築は、profile が `official_faithful` でも `setup` が成功する。`selection_uses_test` が真の snapshot は、その真を `identity()` に残し、偽へ書き換えない。
- Phase: 3-B

## Resume

- 方法: micro-batch 添字 1 を Trainer が消費した状態を保存する。保存前に添字 2 を `batch` で生成して捨てる。別プロセスで `resume` し、次の `batch` の `sample_id` を見る。`plan_hash` を 1 文字変えたファイルも開く。
- 期待: 次の `sample_id` は、中断せず添字 2 を読んだときの `sample_id` と一致する。捨てた先読みは `update_index` に入っていない。`update_index` は 2。改竄した `plan_hash` は `StateError` または `CheckpointError`。prefetch を有効にした実行と無効にした実行で、同じ添字の波形の最大絶対誤差は 0。
- Phase: 3-B
- 前提: checkpoint 形式は `audio-db-experiment-v1` のまま。必須キーは増やす必要が無い。増やす場合は [STATE_AND_CHECKPOINT.md](STATE_AND_CHECKPOINT.md) の version を上げ、この表の期待を先に更新する。

## DDP

- 方法: world size 2、CPU Gloo、固定ペアと chunk と DFN3 と SEMamba のそれぞれで、両 rank の `plan_hash` を比べる。rank 0 の micro-batch 0 と rank 1 の micro-batch 0 の `sample_id` 集合を比べる。保存前に rank 1 の data `plan_hash` だけを変えた試験ダブルで `save` する。
- 期待: 両 rank の `plan_hash` は一致する。固定ペアの `torch_distributed_sampler_v1` と `rank_mod_fixed_v1` では、同じ micro-batch 添字の sample id が rank 間で交差しない。SEMamba の `per_rank_independent_v1` は交差してよい。交差の有無を試験が記録し、交差を失敗にしない。改竄した保存は完成ファイルを置換しない。world size 1 の checkpoint を world size 2 で開くと `CheckpointError`。
- Phase: 3-E
- 非目標: NCCL 上のビット一致。通信が必要な検査は Phase 2-C の試験が既にカバーしている Runtime 側であり、この行は data の列が rank 間で矛盾しないことである。

## Dynamic Sampling

- 方法: 平坦 Sampler で、全 sample の重み 1 の列と、1 件だけ重み 0 の列を、同じ seed と epoch で比較する。`train_epoch` が返ったあとに `set_sampling_weights` を呼ぶ。重みつきの `state_dict` を、重みが `null` の DataModule へ載せる。
- 期待: 重み 0 の sample id は、その epoch の列に出ない。開いている view への変更は `DataError` で、既に返った view の `batch(0)` は変更前と最大絶対誤差 0。重みの有無が違う `load_state_dict` は `StateError`。chunk の DataModule にこのメソッドを呼ぶと `DataError`。
- Phase: 3-F
- 前提: 抽選の名前と試験ベクトルが [DATA_MIGRATION_PLAN.md](DATA_MIGRATION_PLAN.md) に追記されている。追記前に実装しない。

## 既存テストの回帰

- 方法: 各実装段の最後に、移行計画のコマンドを実行する。
- 期待: `tests/experiments` の既存失敗数が増えない。`tests/reproduction` の失敗数が、段の直前の記録を上回らない。`src/training/` と `src/reproduction/` の数値関数に差分が無い段では、`tests/training` をデータ移行の回帰とは呼ばない。
- Phase: 3-B、3-C、3-D、3-E、3-F

## モデル接続（データ供給の完了条件ではない）

- 方法: Phase 4 で、`src.se.build_model("mp_senet", ...)` と固定ペア DataModule を使い、短い波形の L1 を CPU で 1 logical step 進める。
- 期待: 損失が有限。Generator の少なくとも 1 パラメータが初期値と異なる。`enhance()` の呼び出し回数は 0。公式の複合損失はこの行に含めない。
- Phase: 4
