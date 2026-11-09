# Phase 3-C Chunk Data Pipeline

TF-GridNet 向けの `pregenerated.ms_snsd_tfgridnet` と、TF-Locoformer 向けの `pregenerated.dns_tflocoformer` を、Phase 3-B の `SnapshotDataModule` から供給する。SE モデル、公式損失、公式 Trainer は移していない。公式規模の 360,000 件は走らせていない。world size 2 の Data System 正式受け入れは Phase 3-E のままである。

測定は 2026-11-09、`.venvs/se`、`CUDA_VISIBLE_DEVICES=""` である。GPU 試験は行っていない。

## 1. 実装したクラスと公開 API

追加した公開名は `ChunkSampler` と `ChunkProtocol` である。YAML のクラスは `src.experiments.data.ChunkSampler` と `src.experiments.data.ChunkProtocol` である。DataModule は新しいクラスを作らず、`SnapshotDataModule` の `setup`、`train_epoch`、`validation_epoch`、`state_dict`、`load_state_dict`、`identity`、`lineage`、`close` をそのまま使う。

`EpochView.batch(i)` は長さ 1 のリストで、要素は 1 micro-batch の `Batch` である。`inputs` は noisy、`targets` は clean、`lengths` は切り出したサンプル数、`teacher_kind` は case の `dry_clean` である。`sample_id` は case id をカンマで結ぶ。`realization_id` は `{case_id}:{start}:{stop}` である。

`TensorPairData`、`Batch`、`EpochView` の import は残る。固定ペアの `OrderSampler` と `FixedPairProtocol` も残る。Factory はモデル名で分岐しない。Source、Sampler、Protocol、Batcher は `role: other` で、StateRegistry には入らない。

学習は DuckDB を開かない。`OnDemandChunkData`、`DnsOnDemandChunkData`、`ReproductionTrainingData` は新 DataModule の中で作らない。構築試験は、これらのコンストラクタを差し替えて呼び出し回数 0 を確認した。

既存の数値を関数に出した。式は移していない。

- `chunk_case_length` は realization 文書の `output.length` と case の `frames` を比べる。音声ファイルは読まない。旧 `_length` はこれを呼ぶ。
- `ms_snsd_rendering` と `render_ms_snsd_pair`。旧 `OnDemandChunkData._pair` は後者を呼ぶ。
- `dns_rendering` と `render_dns_pair`。旧 `DnsOnDemandChunkData._pair` は後者を呼ぶ。

## 2. Phase 3-A 設計との差異

Chunk 専用の DataModule は作らない。Phase 3-A の公開 API も、固定ペアと chunk で同じ DataModule である。

`plan_chunk_epoch` の `rank_batches` は、すでに rank 内の micro-batch である。Sampler の公開メソッドは平坦な `SampleRef` 列のままにした。DataModule はそれを `MicroBatcher.groups` に渡す。`per_rank_batch` が `micro_batch` と等しく、`group_size` が 1 のとき、再区切りは元の batch 境界と一致する。公開メソッドの引数は変えていない。

`global_batch` は YAML 例に無い。省略したときは `micro_batch * world_size` を計画関数へ渡す。kwargs に書いた値がその積と違うときは `DataError`。`group_size != 1` も `DataError` である。

`plan_hash` のキーは Phase 3-B のままである。chunk 長、整数 shift、shift の文字列、cache、shuffle、`num_samples_per_epoch` は、hash の `order_params` に入る。`order_params` 関数が固定ペアへ返す辞書は変えていない。全 rank の残した `SampleRef` も hash に入る。worker 数、prefetch、device、消費済み cursor は入らない。

`identity()` の `window` は、Sampler が `chunk_length` を持つときその整数である。chunk の `normalize` は null である。`order_policy` は `espnet_chunk_v1` である。

`close` の所有者は Phase 3-B のまま `Experiment.close()` である。checkpoint 形式は `audio-db-experiment-v1` のままである。`weights` は null である。

## 3. Chunk 計画の接続方式

`ChunkSampler` はケース列と `chunk_case_length` の長さだけを `plan_chunk_epoch` に渡す。波形は計画時に作らない。ファイル全体のデコードもしない。

渡す epoch は Trainer の epoch に 1 を足した値である。world size はプロセスの world size で、shard 済みの列を world size 1 として渡し直さない。shuffle と rank 分割は計画関数の中で 1 回だけ行われる。Sampler も Batcher も、そのあと shuffle も shard もしない。

使う列は `rank_batches` である。これは `iterator_stop = min_rank_batches` のあとで、cache の端数 chunk は計画関数がすでに捨てている。Sampler はこの batch を平坦化し、`SampleRef.chunk` に `ChunkSpan(start, stop)` を入れる。`index` は、その rank の平坦列の中の位置である。

`group_size` が 1 で `micro_batch` が `per_rank_batch` と等しいので、`MicroBatcher` は落とした尾を追加で捨てない。試験は、再区切りの case id、start、stop、micro-batch 数が `rank_batches` と一致することを見た。

検証は学習計画を使わない。`validation_epoch` は `valid.cases` を、`shuffle=False`、`num_samples=None`、同じ world size で計画する。rank の再分割はしない。

## 4. MS-SNSD の描画方式

algorithm が `pregenerated.ms_snsd_tfgridnet` のとき、`ChunkProtocol` は `render_ms_snsd_pair` を呼ぶ。中身は現行の `_pair` と同じである。

1. snapshot の realization 文書を読む。
2. `test.cases` のときだけ `clean_test` と `noise_test` を使い、train と valid は train 側の素材を使う。
3. `build/ms_snsd.py` の `render_realization` で clean と noisy を作る。
4. `_pcm16_roundtrip` のあと float32 にする。
5. 計画長と有限値を確認する。
6. そのあと `[start:stop]` を copy する。

RMS 正規化はしない。`rule_json.normalize` が null でない snapshot は `DataError` である。切り出しを PCM_16 より前にはしない。

## 5. DNS 2020 の描画方式

algorithm が `pregenerated.dns_tflocoformer` のとき、`ChunkProtocol` は `render_dns_pair` を呼ぶ。中身は現行の `_pair` と同じである。

1. `dns_rendering` が rule の renderer と retry policy の組を確認する。`dns_pcm16_v1` は `length_plan_no_sample_retry`、`dns_pcm16_accept_v2` は `deterministic_case_stream_v1` だけを受ける。
2. 文書の retry policy が snapshot の policy と違うときは拒否する。
3. `build/dns2020.py` の `render_realization` を呼ぶ。16 kHz でない `source_rate` がある受理計画は、この関数が全ファイルをリサンプルしてから保存済み span を切り、そのあと SNR 混合と PCM_16 を行う。
4. float32 にして計画長と有限値を確認する。
5. そのあと `ascontiguousarray` で `[start:stop]` を取る。

MS-SNSD と DNS を一つの混合器にはしていない。リサンプルは DNS の既存関数の中だけである。

## 6. 数値比較結果

fixture は一時ディレクトリの合成 snapshot である。train は 6 件で、長さは 32、80、96、128、160、112 サンプルである。32 は chunk 長以下なので計画関数が捨てる。valid は 96、80、48。test は 40 で、valid とは case id が重ならない。chunk 長は 32、shift ratio は 0.5、cache は 2、`num_samples_per_epoch` は 3、seed は 7 である。

呼んだ計画関数は `plan_chunk_epoch` である。world size 1、micro-batch 2、global batch 2、epoch 0 と 1、`shuffle=True` と `shuffle=False` を別々に比べた。case id、start、stop、micro-batch 境界、micro-batch 数は一致した。`espnet_epoch` は Trainer epoch に 1 を足した値だった。`shuffle=False` の epoch 0 では `discarded_tail_chunks` が 1 で、その端数は Sampler の列に無い。長さ 32 の case は列に出ない。

world size 2 は偽の runtime で、micro-batch 1、global batch 2、`shuffle=False`、epoch 0 である。計画関数へ渡した `world_size` は 2 で、1 には落としていない。rank 1 の batch 境界は `rank_batches` と一致した。`discarded_batches[1]` は 0 より大きく、残した batch 数は両 rank で同じだった。rank 0 と rank 1 の `plan_hash_for_state({"epoch": 0})` は一致した。ProcessRuntime と Gloo は使っていない。これは Phase 3-E の正式受け入れではない。

MS-SNSD の波形は、同じ span について `OnDemandChunkData.collate_chunks` と比べた。epoch 0、`shuffle=True` の全 micro-batch と、長さ 96 の case の手で置いた span `16:48` である。noisy と clean の最大絶対誤差は 0 だった。PCM_16 は `render_ms_snsd_pair` の中で、切り出しより前に行われる。

DNS は別 snapshot である。rule は `dns_pcm16_accept_v2` と `deterministic_case_stream_v1` である。train の先頭 case の雑音は 8 kHz、90 サンプルで、文書の `source_rate` は 8000 である。`DnsOnDemandChunkData.collate_chunks` と、epoch 0、`shuffle=False` の全 micro-batch、およびその case の span `4:36` を比べた。noisy と clean の最大絶対誤差は 0 だった。リサンプル、PCM_16、chunk の切り出しの順は `render_realization` のあとで span を取る現行順のままである。

検証は `shuffle=False`、`num_samples=None`、valid の case 列である。epoch 0 と 1 の span は一致した。学習 epoch 0 の span とは case id が重ならず、列も違う。batch 数は学習計画の `updates` と検証計画の `updates` を別々に確認した。

## 7. Checkpoint / Resume

CPU、FP32、world size 1 である。中断側は `stop_after_logical_steps: 2`、prefetch on、最大 4 logical step である。`step_00000002.pt` があり、`complete.json` は無かった。同じディレクトリへ resume し、別プロセスで続けた。連続実行は別ディレクトリで prefetch off である。metrics の index 2 の sample id は一致した。

プロセス内で戻すと、`update_index` は 2、epoch は 0 で、`plan_hash` は prefetch off の新しい DataModule の epoch 0 と一致した。`batch(2)` の sample id、realization id、noisy、clean は、その新しい DataModule と最大絶対誤差 0 だった。先に描いた未消費の chunk は `update_index` に入っていない。`weights` は null である。

保存ファイルの `plan_hash` を `"f" * 64` にすると、復元は `CheckpointError` になった。

## 8. 異常系

測ったものは次である。

- `group_size` 2 の `SnapshotDataModule.setup` は `DataError`。同じ指定の YAML は `build_experiment` が `BuildError` で包む。
- `global_batch: 4` と micro-batch 2、world size 1 の組は `DataError`。
- 範囲外の micro-batch 添字は `StateError`。
- 同じ添字を 2 回読むと、入力と教師の最大絶対誤差は 0 で、tensor の device は CPU である。

素材ファイルの欠損を、この段の試験では新たに測っていない。`ReproductionError` は描画と計画の入口で `DataError` に包む。

## 9. 既存テストの回帰

変更前:

| 対象 | 結果 |
|---|---|
| `tests/experiments` | Ran 54 tests in 125.964s、OK (skipped=1)、EXIT 0。ログ `/tmp/phase3c-experiments-before.txt` |
| `tests/reproduction` | Ran 135 tests in 460.756s、FAILED (errors=1, skipped=2)、EXIT 1。ログ `/tmp/phase3c-reproduction-before.txt` |

事前の reproduction の 1 error は `test_p8_dfn3.TrainerSmokeTests.test_one_deepfilternet_step_is_finite` で、`libdf_rfft.so` が GLIBC_2.33 を要求して読めない。Phase 3-B の記録と同じである。skip も assertion も変えていない。

変更後の `tests/experiments` は、Ran 63 tests in 140.559s、OK (skipped=1)、EXIT 0。ログ `/tmp/phase3c-experiments-after.txt`。増分 9 件は `tests/experiments/test_chunk_data.py` で、失敗は 0、新規 skip は 0、NOT_RUN は 0 である。既存の skip 1 は CUDA 非表示の `tests/experiments/test_cuda_amp.py` で、変更前からある。

変更後の `tests/reproduction` は、Ran 135 tests in 464.025s、FAILED (errors=1, skipped=2)、EXIT 1。ログ `/tmp/phase3c-reproduction-after.txt`。error は事前と同じ `test_one_deepfilternet_step_is_finite` で、`libdf_rfft.so` が読めない。errors と skipped は事前から増えていない。失敗の新規追加は 0、skip の新規追加は 0、NOT_RUN は 0 である。

GPU 試験は必須ではない。実行していない。

## 10. Phase 3-D への残課題

- DFN3 のオンライン混合と、SEMamba++ のオンライン劣化。
- 動的な重み付き Sampler。chunk には掛けない。
- prefetch の worker プール。今の prefetch は次の 1 micro-batch を CPU で先に描くだけである。
- world size 2 の Data System 正式受け入れ。今回の偽 runtime は、計画関数の引数と hash の一致までである。
- SE モデルの forward、公式損失、GAN Trainer、本格的な GPU 学習、FSDP、sharded checkpoint。
- 公式の未 seed 生成とのビット一致、および 360,000 件の全計画。

## 11. 判定

**GO。**

`ChunkSampler` と `ChunkProtocol` を公開 API から構築し、`plan_chunk_epoch` と chunk 順、span、micro-batch 境界が一致した。MS-SNSD と DNS 2020 の波形は、既存の描画と最大絶対誤差 0 だった。検証計画は学習計画と分かれている。CPU の別プロセス再開で、次の chunk の sample id と波形が連続実行と一致した。`tests/experiments` は 63 件 OK、skip 1 で、失敗は 0 である。`tests/reproduction` の error は事前と同じ 1 件のままである。

公式 SE の学習再現完了ではない。world size 2 の Data System 正式受け入れ、オンライン供給、公式規模の chunk 計画は測っていない。
