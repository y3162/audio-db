# Phase 3-E Distributed Data System

6 構成の `SnapshotDataModule` を、CPU / Gloo の実 2 プロセスで供給し、同じ world size の別プロセス群から再開する。新しい DataModule は作らない。公式 SE、公式損失、GAN Trainer、動的 Sampler、FSDP、sharded checkpoint、world size を変える再開は入れていない。`src/training/` は変えていない。`src/reproduction/` の数値式は変えていない。

測定は 2026-11-09、`.venvs/se`、`CUDA_VISIBLE_DEVICES=""` である。GPU は使っていない。学習モデルは長さに依存しないスカラー利得 `tests.experiments.support.GainWave` である。本格 SE の学習再現ではない。

## 1. 変更ファイル一覧

変更:

- `src/experiments/data/fixed_pair.py`。`load_workers` と有界な prefetch スレッド。`close` と `load_state_dict` が未終了スレッドを成功にしない。
- `src/experiments/data/online.py`。`Dfn3Protocol` と `SemambaProtocol` の clip キャッシュを `threading.Lock` で守る。数値は変えていない。
- `requirements/se.in` と `requirements/se.lock.txt`。`resampy==0.4.3`。
- `tests/experiments/support.py`。`GainWave` と、rank 不一致・構築失敗・描画失敗の試験ダブル。
- `tests/experiments/test_fixed_pair_data.py`。合成 snapshot の件数を引数にした。既定は従来の 8 / 2 / 2 のままである。

追加:

- `tests/experiments/distributed_data_worker.py`。`torchrun` の 2 プロセスが読む供給と波形の作業。
- `tests/experiments/test_distributed_data.py`。
- この記録。

`src/training/` と `src/reproduction/` の数値関数には差分が無い。DB、snapshot、素材は再構築していない。

## 2. 実際の world size 2 の実行方法

起動は `python -m torch.distributed.run --standalone --nproc_per_node 2 --master_port <空き>` である。環境は `CUDA_VISIBLE_DEVICES=""`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` で、親プロセスの `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT` は外す。CUDA が見えないので `ProcessRuntime` は Gloo を選ぶ。`init_process_group` の timeout は runtime の `timeout_seconds` で、既定 120 秒、失敗試験は 8 秒である。

供給の受け入れは `tests.experiments.distributed_data_worker` をこの起動で読む。学習の受け入れは同じ起動で `python -m src.experiments.cli --config ...` を読む。どちらの試験も、偽の world size を 1 プロセスに渡していない。

## 3. 各 Sampler の rank 分担

計画関数の式は変えていない。2026-11-09 に seed 1、epoch 0、world size 2 で測った列は次である。

| policy | 件数 | rank 0 | rank 1 | 重なり | 欠番 |
|---|---|---|---|---|---|
| `torch_distributed_sampler_v1` | 8 | `[4, 7, 2, 1]` | `[0, 3, 5, 6]` | 無し | 無し |
| `torch_distributed_sampler_v1` | 7 | `[4, 5, 2]` | `[0, 3, 6]` | 無し | `[1]` |
| `rank_mod_fixed_v1` | 8 | `[0, 2, 6, 4]` | `[1, 3, 7, 5]` | 無し | 無し |
| `rank_mod_fixed_v1` | 7 | `[4, 0, 2, 5]` | `[1, 3, 4, 0]` | `[0, 4]` | `[6]` |
| `per_rank_independent_v1` | 7 | `[2, 6, 4]` | `[3, 5, 0]` | 無し | `[1]` |

`torch_distributed_sampler_v1` は `drop_last` である。件数 7 の欠番は並べ替えた列から落ちた添字で、整数の末尾 6 とは限らない。この seed では 1 が落ち、各 rank は 3 件で互いに素である。全件を覆うようにはしていない。

`rank_mod_fixed_v1` は各 rank `ceil(size / world_size)` 件である。`distributed_sampler_coverage(7, 2)` は `num_samples_per_rank=4`、`dropped_tail=1`、`padded_duplicates=2`、`unique_assigned=6` だった。先頭の繰り返しがある。全件一意の被覆にはしていない。

`per_rank_independent_v1` は rank ごとの `randperm` である。重なりと欠番を許す。この seed は重なりが空で欠番が `[1]` だった。別の seed で重なることを失敗にはしない。`plan_semamba_epoch` は端数の micro-batch と端数の蓄積群を捨てる。件数 7、micro-batch 1、`accumulation` 1 では各 rank 3 更新で捨ては 0 だった。`accumulation` 2 では各 rank 1 群（micro-batch 2 本）になり、rank 0 は添字 4、rank 1 は添字 0 を捨てた。

chunk は `plan_chunk_epoch` をプロセスの world size のまま呼ぶ。world size 1 の計画をあとから割っていない。`iterator_stop` は `min_rank_batches` である。合成の MS-SNSD と DNS で、両 rank の plan hash と更新回数が一致した。

DFN3 は `plan_online_epoch` である。world size が 1 より大きいとき `_shard` は `(-len) % world_size` 件を先頭から繰り返してから `padded[rank::world_size]` にする。件数 5、micro-batch 2、world size 2 では `padded_samples=1`、各 rank の batch 長は `[2, 1]`、`partial_last=[1, 1]`、更新回数は 2 で両 rank が同じだった。world size 1 の短い尾を残す式は変えていない。

`StandardTrainer` は局所 batch の平均損失を `accumulation` で割り、DDP が rank 勾配を等しく平均する。最終 micro-batch の件数が rank で違うと、短い rank の 1 件の実効重みは `(1 / 局所件数) * (1 / accumulation) * (1 / world_size)` になり、長い rank の 1 件より大きい。DFN3 の world size 2 は pad で件数が揃うので、この式の件数差は出ない。Sampler の式はこの重みを揃えるために変えていない。

## 4. 各 Protocol の Validation 分担

学習の `train_epoch` とは別の列である。実 2 プロセスで次を確認した。

| protocol | rank 0 | rank 1 |
|---|---|---|
| `fixed_pair.mp_senet` | `validation_refs(..., rank=None)` の全件。空ではない | 空の iterator |
| `fixed_pair.mossformergan_se_16k` | `rank_mod_fixed_v1` のこの rank。空ではない | 同じ方針の rank 1。空ではない |
| `pregenerated.*` | `plan_chunk_epoch(shuffle=False, num_samples=None)` の span | 同じ関数の rank 1 |
| `online.dfn3` | world size 1 の全件。空ではない | 空の iterator |
| `online.semambapp_gsr` | 素材順。空ではない | 空の iterator |

空の iterator は成功である。検証損失を rank 間で集める処理は Trainer に無い。公式の検証損失とモデル選択は実装していない。集計は後続 Trainer の責務である。

MP-SENet の検証波形は epoch 0 と 1 の先頭 batch で最大絶対誤差 0 だった。固定ペアの `realization_id` は epoch を含むので、波形が同じでも realization 文字列は epoch で変わる。chunk の realization は `case:start:stop` で、epoch を含まない。DFN3 と SEMamba は、並べ替えではなく同じ sample index の波形を epoch 0 と 1 で比べ、最大絶対誤差 0 だった。整数の `sampling_factor` だけを測った。

## 5. Prefetch と Worker 設計

`SnapshotDataModule` の引数は `prefetch`、`load_workers`、`worker_join_timeout`（既定 30 秒）である。

- `prefetch=False` は `load_workers` を 0 にする。0 以外を渡すと `ValueError`。
- `prefetch=True` で `load_workers` を省略すると 1 になる。Phase 3-D までの `prefetch=True` はそのまま動く。
- `prefetch=True` かつ `load_workers < 1` は `ValueError`。

スレッドは daemon ではない。名前は `audio-db-prefetch-{i}` である。キューは `(epoch, index, group)` と停止用の `None` である。collation はロックの外で行う。先読みは `index+1` から `index+load_workers` までである。キャッシュの上限は `max(1, load_workers+1)` で、入れたキー以外の最も小さいキーから捨てる。

先読みは `update_index` を書かない。未消費の batch は `state_dict` に入らない。`load_state_dict` は比較の前に `_stop_prefetch(propagate_error=False)` を呼び、古い先読みの例外を状態へ載せない。`close` は `propagate_error=True` で止める。`join` が timeout 以内に終わらないスレッドは `self._workers` に残し、`DataError("prefetch workers did not exit: ...")` にする。残ったスレッドを成功として消さない。二度目の `close` は残っていれば再試行する。worker の例外は次の `materialize` または `close` で伝播する。

描画は呼び出しごとに `np.random.Generator(PCG64(sample_seed(...)))` を作る。Protocol の乱数を prefetch スレッドで共有しない。rank と worker 番号は DFN3 と SEMamba の波形の種に入れない。clip キャッシュはロック付きの二重検査である。`load_workers` と `prefetch` は `plan_hash` にも `identity()` にも入らない。実験の `config_sha256` には component の kwargs として入る。worker 数だけが違う再開は `CheckpointError("checkpoint config identity does not match this experiment")` で拒否する。この検査は緩めていない。波形が一致することと、設定が違う checkpoint を再開することは別である。

## 6. Worker 数による波形比較

実 2 プロセスの各 rank で、`load_workers` 0、1、2 の batch 0 を比べた。6 構成と、件数 7 の MP-SENet、`group_size` 2 の SEMamba を含む。plan hash、sample id、realization id は一致し、input と teacher の最大絶対誤差は 0 だった。`update_index` は 0 のままだった。world size 1 の固定ペアでも、worker 0、1、2 の batch 0 と 1 が同じだった。全 batch を消費したあとのキャッシュ長は `load_workers+1` 以下だった。

描画が例外を出す protocol では、その例外が `batch` から出た。止まらないスレッドを `worker_join_timeout=0.2` で `close` すると、メッセージは `did not exit` だった。

## 7. 分散 Checkpoint / Resume

形式は `audio-db-experiment-v1` のままである。6 構成とも CPU / Gloo、world size 2、logical step 4 の連続実行と、2 step で中断して別の 2 プロセス群から 4 step まで再開した実行を比べた。中断ファイル `step_00000002.pt` のバイト列は再開後も同じだった。最終 checkpoint の data `epoch`、`update_index`、`plan_hash`、`world_size`（2）、metrics の sample id 列は一致した。

代表は MP-SENet、AdamW `lr=0.01`、StepLR `step_size=1` `gamma=0.5`、`L1Loss`、`accumulation=1`、`deterministic: true` である。parameter、optimizer、scheduler、trainer 状態の最大絶対誤差は 0、`rng_by_rank` は一致、loss 列は一致した。他の 5 構成は同じ GainWave と AdamW で、scheduler は付けていない。data 列と sample id 列は 6 構成とも一致した。

中断 checkpoint を別の 2 プロセスで復元し、cursor を進めずに次の batch を描いた。両 rank の保存 epoch、保存 index、plan hash は一致し、sample id、realization id、波形 digest は空ではなかった。

`load_workers=1` の checkpoint を `load_workers=2` で開くと、config identity の不一致で拒否され、元ファイルのバイト列は変わらなかった。

## 8. rank 間不一致と異常系

既存の保存前比較と process group の timeout を使った。任意のプロセス故障からの自動復旧は入れていない。

- rank 1 だけが `update_index` を 1 進めた保存は `ranks disagree at state.data.update_index` で終わり、`step_00000001.pt` は無く、`status.json` は `FAILED` だった。
- rank 1 の `state_dict` だけが `plan_hash` を `"0" * 64` にした保存は `plan_hash` を含む不一致で終わり、完成ファイルは無かった。
- 中断ファイルの data `plan_hash` を `"0" * 64` に変えた別ファイルからの再開は `plan_hash` を含む拒否で、元の `step_00000001.pt` のバイト列は変わらなかった。完了済みディレクトリは `complete.json` で先に拒否されるので、この試験は `stop_after_logical_steps: 1` の中断ファイルを使った。
- rank 0 の `setup` が `DataError("injected build failure on rank 0")` を出すと、プロセスは 25 秒以内に終了し、出力にその文言があった。Gloo の timeout は 8 秒である。
- rank 1 の最初の `materialize` が `DataError("injected render failure on rank 1")` を出すと、rank 0 は backward や checkpoint の collective で待ち続けず、25 秒以内に終了し、出力にその文言があった。

失敗を skip にはしていない。

## 9. SEMamba++ の依存関係確認

`SEMAMBA_GSR_V1` の `bandwidth_type` は `kaiser_best`、`kaiser_fast`、`scipy`、`polyphase` である。`lowpass_type` は `butter`、`cheby1`、`cheby2`、`bessel`、`ellip` である。`bandwidth_limitation` は `librosa.resample` を呼ぶ。Kaiser の帯域制限は `resampy` を要求する。`scipy` と `polyphase` は `resampy` が無くても動く。劣化確率や数値処理は変えていない。

`.venvs/se` に `resampy` が無かったのは、`requirements/se.in` が pin しておらず、`requirements/asr.lock.txt` の `resampy==0.4.3` を SE 環境が使わないためである。P9 の検証は `.venvs/asr` のその版を `sys.path` に足していた。本番コードはそのパスを足さない。今回は環境の欠落として `requirements/se.in` に `resampy==0.4.3` を足し、`scripts/setup/compile_locks.sh` と同じ `uv pip compile` で `requirements/se.lock.txt` を作り直して `.venvs/se` に入れた。`numpy` は `2.5.4`、`torch` は `2.6.0+cu124` のままである。lock に無かった tensorboard の推移依存と、`markdown` の `3.11.2` から `3.11.3` への pin もこの再コンパイルで入った。数値スタックの変更ではない。

`BandwidthPathTests` は 1,600 サンプルで 4 種の resampler と 5 種の lowpass を `kaiser_best` と組み合わせ、有限かつ同じ形状だった。Kaiser を含む劣化経路は `.venvs/se` で動く。DDP の data 結果とは別の確認である。

## 10. 回帰テスト

Phase 3-D 終了時の基準は `tests/experiments` 71 件、失敗 0、既存 skip 1（`test_cuda_amp`。CUDA を隠している）、`tests/reproduction` 135 件、既存 error 1（`test_p8_dfn3.py` の `GLIBC_2.33`）、skip 2 である。

実装の途中で走らせた `tests/experiments` は、書きかけの `SnapshotDataModule` を import し、`_stop_prefetch` のキーワード引数と `_raise_prefetch_error` の欠落で 2 失敗した。これは変更前の基準ではない。`tests/reproduction` の同時実行は 135 件、error 1、skip 2 で、基準と同じだった。

変更後のコマンドは次である。

```text
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/experiments -v
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/reproduction -v
```

変更後の `tests/experiments` は 81 件、失敗 0、skip 1 である。skip は `test_cuda_amp` の `NOT_RUN: CUDA is not visible` で、Phase 3-D の既存 skip と同じである。増えた 10 件は Phase 3-E の prefetch、帯域制限、rank 規則、実 2 プロセス供給、6 構成の再開、失敗注入である。

変更後の `tests/reproduction` は 135 件、error 1、skip 2 である。error は `test_p8_dfn3.TrainerSmokeTests.test_one_deepfilternet_step_is_finite` の `GLIBC_2.33`（`libdf_rfft.so`）で、Phase 3-D から残っている。新しい error、failure、skip は無い。

## 11. Phase 3-F への残課題

- 平坦 Sampler の `set_sampling_weights`。chunk と DFN3 には掛けない。
- 検証損失の rank 集計、公式の検証損失、モデル選択。
- 公式 SE モデル、公式損失、GAN Trainer、本格 SE の学習。
- FSDP、sharded checkpoint、world size を変える再開。
- NCCL 上の data 供給。この段は CPU / Gloo で足りる。

## 12. 判定

GO。

6 構成の CPU / Gloo world size 2 供給、rank 規則、plan hash、代表構成を含む再開、worker 数を変えた波形、rank 不一致が完成ファイルを残さないこと、既存回帰は満たした。本格 SE の学習再現ではない。Phase 3-F の動的 Sampler と、検証損失の rank 集計は残している。
