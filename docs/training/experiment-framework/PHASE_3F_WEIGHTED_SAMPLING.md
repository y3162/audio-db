# Phase 3-F Dynamic Weighted Sampling

平坦な `OrderSampler` に、外部から渡した重みで復元抽出する `weighted_categorical_v1` を足した。新しい DataModule は作らない。Loss や WER から重みを計算する処理、Expert 割当、動的 MoE は入れていない。公式の `rank_order` は、重みを渡していない epoch では変えていない。

測定は 2026-11-09、`.venvs/se`、`CUDA_VISIBLE_DEVICES=""` である。GPU は使っていない。代表の学習モデルは `tests.experiments.support.GainWave` である。本格 SE の学習再現ではない。

アルゴリズム名、種、試験ベクトルは実装前に [DATA_MIGRATION_PLAN.md](DATA_MIGRATION_PLAN.md) の「Phase 3-F の抽選」へ固定した。

## 1. 抽選アルゴリズムと数学的仕様

研究用 policy の名前は `weighted_categorical_v1` である。生成器は `splitmix64-v1`。rank 分担は `stride_drop_tail_v1` である。`rank_mod_fixed_v1` でも `per_rank_independent_v1` でもない。

実効重みは `format(float(w), ".6f")` で、`ids.digest` と同じ 6 桁である。抽選の整数は、その文字列から `整数部 * 1_000_000 + 小数部` として作る。hash に入れるのもこの実効重みで、抽選だけが別の丸めを持たない。6 桁で 0 になった値は抽出しない。正の整数は最大公約数で割ってから累積和を作るので、6 桁の格子上で比率が同じ表は同じ列になる。hash は割る前の表である。

確率は正の実効重みに比例する。復元抽出である。抽選回数は、重み 0 を含めた学習サンプル数 `n` である。重みを変えてもこの回数は変えない。正の重みが 1 件でも `n` 回まで埋める。合計が 0 の表は `DataError` である。

全域列を全 rank が同じ種から計算する。`usable = (n // world_size) * world_size` とし、末尾を捨て、rank `r` は `r, r+world_size, ...` を受け取る。その後の micro-batch と group size の端数は既存の `MicroBatcher` が捨てる。`n < world_size` で rank の列が空なら `DataError` である。

## 2. Sample Identity と重み表

固定ペアの Sample ID は `train.cases` の `case_id` である。SEMamba++ の Sample ID は `"{protocol_id}:train:{index}"` で、素材 ID ではない。表は学習サンプルを過不足なく含む。欠落を 1 で埋めない。bool、NaN、Inf、負数、非数は拒否する。

`state_dict["weights"]` は、その `epoch` の計画を作った実効重みである。未設定は `null`。まだ計画していない次 epoch への変更は、現在の `epoch` の `weights` に入れない。

## 3. Seed と Epoch 規則

ストリームは `mix_seed("weighted_categorical_v1", training_seed, epoch)` である。epoch ごとに別の列になる。rank も worker も種に入れない。重みがすべて 1 でも非復元の `rank_order` と同じ列にはしない。seed 7、4 件、すべて 1、epoch 0 の列は `s1, s0, s3, s2` で、`fixed_single_shuffle_v1` の permutation とは違った。

## 4. DDP の rank 分担

world size 2 の正式測定は `torch.distributed.run --standalone --nproc_per_node 2`、CPU / Gloo である。全 rank が同じ全域列を作り、`stride_drop_tail_v1` で分ける。process group があるとき、epoch 計画の作成で `weights_sha256` を集める。1 つでも違えば `DataError("ranks disagree on sampling weights")` で、完成 checkpoint は出さない。rank 0 から重みを配る処理は無い。

件数 7、重みすべて 1、seed 7、epoch 0 の全域列は `5, 6, 6, 5, 6, 1, 1` である。末尾 1 件を捨て、rank 0 は `5, 6, 6`、rank 1 は `6, 5, 1` である。

片 rank の重み表が違う試験は 8 秒の process group timeout と 25 秒のプロセス上限で戻り、無期限には待たなかった。rank 0 だけが全重み 0 で `set_sampling_weights` に失敗する試験も、同じ上限の中で非 0 終了した。

## 5. Epoch View のライフサイクル

`SnapshotEpochView.close()` が終了契約である。二度呼んでも成功する。閉じた view の `batch` は `DataError` である。GC では閉じない。

開いている view がある `set_sampling_weights` は `DataError` で、その view の sample id と波形は変わらない。計画を作った epoch は凍結し、同じ epoch を開き直しても列は変わらない。変更は、まだ計画を作っていない epoch から効く。先読みキャッシュは重み変更時に捨てる。検証の列と波形は重みを読まない。

`StandardTrainer` は、各 step の `train_epoch` が返した view を `finally` で閉じる。正常終了、中断、`BoomLoss` の例外で閉じることを CPU で見た。`close` が無い `TensorPairData` の view はそのままである。

## 6. Plan Hash

重みの無い epoch の payload は Phase 3-E までと同じで、`weights_sha256` は `"none"`、`order_policy` は snapshot の policy である。重み付き epoch は `order_policy` を `weighted_categorical_v1` にし、`order_params` にアルゴリズム、`splitmix64-v1`、復元抽出、`training_count`、`stride_drop_tail_v1`、scale `1000000`、training seed を入れる。`weights_sha256` は実効重み表の `ids.digest` である。

`identity()` の `order_policy` は snapshot の policy のままである。重みの有無は `state_dict["weights"]` と plan hash で区別する。worker 数は相変わらず plan hash に入らない。

`{"s0": 1.0, "s1": 3.0, "s2": 0.0, "s3": 1.0}` の digest は `c5d7db2a48e3978848a888a328c3a497a5eb6320c8674e9e34b52af7d24bfd7c` である。すべて 1 の digest は `47aa23d5bc83a82b29858e45a2fa748c105766822191c9c7ec64807da5e55c42` である。

## 7. Checkpoint と Resume

形式は `audio-db-experiment-v1` のままである。必須キーは足していない。`weights` は既存キーで、未設定は `null`、設定時は Sample ID から実効重みへの写像である。

再開は、実行中の DataModule に既についている表と checkpoint の `weights` が一致するときだけ計画を再計算する。checkpoint の表で `self._weights` を置き換えない。一致後に plan hash を再計算し、micro-batch cursor を戻す。重みなしから重みあり、重みありから重みなし、違う表は `StateError` である。`initialize_from` は data 状態を戻さない。

`epoch` と `weights` は同じ計画に属する。epoch 0 を重みなしで凍結したあと次 epoch 用の表を置いても、`epoch == 0` の `state_dict` の `weights` は `null` のままで、plan hash も変わらない。epoch を 1 に進めてから保存すると、その計画の表が入る。

## 8. 数値試験ベクトル

seed 7。ID は `s0, s1, s2, s3`。

| 条件 | 列 |
|---|---|
| 重みすべて 1、epoch 0、world size 1 | `s1, s0, s3, s2` |
| 同じ、world size 2 | rank 0 `s1, s3`、rank 1 `s0, s2` |
| `1, 3, 0, 1`、epoch 0、world size 1 | `s1, s1, s3, s1` |
| 同じ、epoch 1 | `s1, s1, s1, s3` |
| 同じ、world size 2 | rank 0 `s1, s3`、rank 1 `s1, s1` |
| 2 倍の `2, 6, 0, 2`、epoch 0 | `s1, s1, s3, s1` |
| `1e-7, 1, 1e-6, 0`、epoch 0、4 回 | index `1, 1, 1, 1` |

`1, 3, 0, 1` の 32 回は `s1, s1, s3, s1, s3, s3, s0, s3, s1, s1, s0, s1, s1, s1, s3, s1, s1, s1, s1, s1, s1, s1, s3, s3, s0, s3, s3, s1, s1, s3, s0, s1` で、`s2` は無い。

固定ペアの合成 snapshot では、同じ関数の位置列と `sample_id` 列が一致した。重み 0 の case は列に出ない。同じ case と epoch の realization は、重みの有無で入力と教師の最大絶対誤差 0 だった。SEMamba++ の重みキーは `:train:{index}` で、重み 0 の発話 index は学習列に出ず、検証の sample id 列は変わらなかった。

## 9. 統計的な抽選分布

決定的な ID 列とは別に、実装前に区間を固定した。サンプル 2 件、重み `1:3`、seed 11、epoch 0、復元 4000 回。index 0 の回数は 918 以上 1082 以下である。これは `4000 * 0.25 ± 3 * sqrt(4000 * 0.25 * 0.75)` を整数へ寄せた区間である。

測定は index 0 が 1046 回、index 1 が 2954 回だった。区間の中である。この回数に合わせて区間は動かしていない。

## 10. 既存の公式 Sampler との差

重みを渡さない MP-SENet、MossFormerGAN、SEMamba++ は、これまでどおり `rank_order` の列と `"none"` の `weights_sha256` を使う。すべて 1 の重み付き列は、それとは別の復元抽出である。

chunk の `ChunkSampler` と DFN3 の `Dfn3IndexSampler` で `set_sampling_weights` を呼ぶと `DataError` である。検証と test の計画関数は重みを読まない。公式 protocol の描画式は変えていない。

## 11. 回帰テスト

変更前、2026-11-09:

- `tests/experiments`: 81 件、失敗 0、skip 1。skip は `test_cuda_amp` の `NOT_RUN: CUDA is not visible`。
- `tests/reproduction`: 135 件、error 1、skip 2。error は `tests/reproduction/test_p8_dfn3.py` の `GLIBC_2.33` で、今回の変更前からある。

変更後、同じ日、同じコマンド:

- `tests/experiments`: 92 件、484.135 秒、失敗 0、skip 1。増分 11 件は `tests/experiments/test_weighted_sampling.py` である。skip は変更前と同じ `test_cuda_amp` の `NOT_RUN: CUDA is not visible` である。
- `tests/reproduction`: 135 件、467.606 秒、error 1、skip 2。error は変更前と同じ `tests/reproduction/test_p8_dfn3.py` の `GLIBC_2.33` である。直していない。

新しい skip と、今回の変更による failure は無い。`src/training/` は変えていない。GPU は使っていない。

## 12. Phase 4 への引き継ぎ

動的重み付きサンプリングの機構はここまでである。難しい発話の自動推定、カリキュラムの段、Expert ごとの DataModule、公式 SE の学習再現は Phase 4 以降である。検証損失の rank 集計も入れていない。重み表は呼び出し側が全 rank に同じものを渡す。DataModule は rank 0 の表を黙って配らない。
