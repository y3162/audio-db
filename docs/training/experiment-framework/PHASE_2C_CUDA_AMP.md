# Phase 2-C CUDA / NCCL / AMP

CUDA / FP32 の world size 1、CUDA / FP32 / NCCL の world size 2、CUDA / AMP FP16 の world size 1、CUDA / AMP FP16 / NCCL の world size 2 を、Phase 2-B までと同じ公開 API で実装した。FSDP、sharded checkpoint、AMP BF16、GAN の学習、公式 SE、公式 Dataset はこの結果に含まれない。

測定は 2026-11-08、`.venvs/se` の PyTorch `2.6.0+cu124`、CUDA 12.4、NCCL `(2, 21, 5)`、ドライバ `535.183.01`、NVIDIA A100-PCIE-40GB である。試験は `AUDIO_DB_CUDA_GPU0=1` と `AUDIO_DB_CUDA_GPU1=2` で物理 GPU を選び、起動前に `CUBLAS_WORKSPACE_CONFIG=:4096:8` を置いた。モデルは `torch.nn.Linear` だけである。使用メモリが 1024 MiB を超える GPU は `NOT_RUN` とし、成功に数えない。この実行では skip は無かった。

## CUDA Runtime

| Device | Precision | world size | Backend |
|---|---|---|---|
| CPU | FP32 | 1 | none |
| CPU | FP32 | 2 以上 | Gloo |
| CUDA | FP32 | 1 | none |
| CUDA | FP32 | 2 以上 | NCCL |
| CUDA | AMP FP16 | 1 | none |
| CUDA | AMP FP16 | 2 以上 | NCCL |

`amp_bf16`、CPU 上の `amp_fp16`、CUDA が使えないときの `device: cuda` は `PrepareError` で、FP32 や CPU や単一プロセスへ落とさない。`LOCAL_RANK` が見える CUDA 枚数以上、`WORLD_SIZE` が見える枚数より大きい、既存 process group の backend が要求と違う、二つの rank の device UUID が同じ、のいずれも `PrepareError` である。

CUDA は見える装置の `cuda:LOCAL_RANK` を選ぶ。`CUBLAS_WORKSPACE_CONFIG` は `set_device` より前に `setdefault` する。cuBLAS は context 作成時にこの値を読む。Builder は `set_determinism(config.deterministic, cuda=(device.type == "cuda"))` を呼ぶ。`torch.manual_seed` は見えている全 CUDA device を seed するので、試験は `CUDA_VISIBLE_DEVICES` をその rank の GPU に限る。

world size 1 は process group を作らず、`distribute: ddp` でも包まない。CUDA の world size 2 以上は NCCL で、`init_process_group(..., device_id=self.device)` と DDP の `device_ids=[local_rank]` を使う。1 rank が 1 GPU である。初期化に失敗したときは `shutdown` して process group を破棄する。`sync_batchnorm` が BatchNorm を `SyncBatchNorm` にするのは、CUDA かつ world size が 2 以上のときだけである。

CPU と同じく、この単一 GPU の DDP は Parameter オブジェクトを置き換えない。Optimizer は prepare の後に同じ Parameter を受け取る。resume は unwrapped の `state_module` へその場で `load_state_dict` する。FSDP はこの前提を使えないので拒否する。

## AMP と GradScaler

プロセスにつき `torch.amp.GradScaler("cuda")` を一つ持つ。Trainer が呼ぶのは `autocast`、`backward`、`clip`、`step`、`end_step` だけである。`GradScaler.step` は呼ばない。

1 logical step の順は、全 micro-batch の backward、Optimizer ごとに 1 回の unscale、clip、overflow の rank 間合意、step または skip、`end_step` で `scaler.update()` を 1 回、である。非有限の loss は backward の前に `RuntimeFailure` であり、overflow skip ではない。有限 loss の backward がスケールで溢れるのは、その後の `step` が判断する。

`clip` は AMP のとき `optimizer` を必須にし、その Optimizer を unscale してから `clip_grad_norm_` する。返す値は unscale 後、clip 前の norm である。同じ step でもう一度 unscale すると `RuntimeFailure`。FP32 は `optimizer` を無視する。`StandardTrainer` は常に Optimizer を渡す。

`applied` が False のとき、Optimizer の更新回数は増やさず、step ごとの Scheduler も進めない。logical step は進む。`set_grad_scale` は step の間にだけスケールを置き換える試験用である。遅延初期化前は `_init_scale` を書き、初期化後は `update(new_scale=)` を使う。後者は growth tracker をリセットしない。

プロセス内で測った clip は、`init_scale` 32、損失 `(weight * 2).sum()`、scaled grad 64、unscale 後の grad 2、L2 norm は `sqrt(8)`、SGD の学習率 0.1 で重み 1 が 0.8 になった。`optimizer.step` の戻り値は `None` だが `applied` は True だった。`growth_interval` 2、`init_scale` 2 の成功 2 回のあと、scale は 4、tracker は 0 だった。新しい `Linear` では scale 65536 は更新し、131072 は有限 loss のまま skip して重みを変えず、scale は 65536 に戻った。次の 65536 は更新した。NaN loss は `not finite` で止まった。

## Overflow の rank 間協調

PyTorch 2.6.0 は overflow を `GradScaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]` に置く。公開 API でこの旗を rank 間 OR する入口は無く、`state_dict` は `update` の後でなければ現在の overflow を表さない。`step` はその private tensor を in-place の `all_reduce` MAX し、全 rank へ 0 か 1 を書き戻す。`scaler.update()` は同じ旗を見る。この private 辞書は `torch==2.6.0+cu124` で確認した。AMP の overflow に `agree(policy="identical")` は使わない。不一致を例外にすると、片 rank だけの overflow を全 rank skip にできない。FP32 の `step` は従来どおり `agree(True, identical)` のあと更新する。

## `StepResult.applied`

`optimizer.step` を包み、その包みが呼ばれたときだけ `applied` を True にする。`GradScaler.step()` の戻り値が `None` であることを skip の根拠にしない。SGD と Adam は更新しても `None` を返す。

## Checkpoint と Resume

形式は `audio-db-experiment-v1`、schema 1 のままである。必須キーは足していない。`device` が無い古い runtime state は CPU と読む。runtime state は `device`、`precision`、`world_size`、`grad_scaler` を持つ。FP32 の `grad_scaler` は `None` で、値があると復元しない。AMP は辞書が欠ける、キーが足りない、scale が 0 以下、だと `CheckpointError` で、空の Scaler を新しく作らない。device 種別、precision、world size の不一致は `StateError` である。

CUDA の device index、name、UUID は、複製される runtime state ではなく各 `rng_by_rank` 要素の `cuda_device_identity` に入れる。rank ごとに UUID が違うためである。復元は identity が無い、index か UUID が現在の device と違う、`rank` 欄が添字と違う、のいずれも拒否する。保存するのは `capture_rank_rng` の現在 device だけで、`get_rng_state_all` は使わない。`initialize_from` は従来どおり module の重みだけを載せる。

保存前の rank 比較は runtime を含む。GradScaler の scale と growth tracker も比べる。NCCL の object collective に GPU storage を載せないため、比較前にテンソルを CPU へ写す。それでも Optimizer state 全体を全 rank へ pickle する。大規模 SE の checkpoint 方式としては不十分であり、sharded checkpoint は作っていない。

## C1〜C4

連続 4 logical step と、2 step で止めて別プロセスから 4 step まで再開した実行を比べた。状態と `rng_by_rank` の最大絶対誤差は、下の 4 条件とも 0 だった。`metrics.jsonl` も一致した。

| 条件 | 内容 | 結果 |
|---|---|---|
| C1 | CUDA / FP32 / world size 1。`grad_clip` 0.5。checkpoint 前に CUDA RNG を消費 | 最大絶対誤差 0。runtime の `device` は `cuda`、`grad_scaler` は `None`。UUID を書き換えたファイルは `CUDA device` で拒否。step 2 と step 4 の重みは異なる |
| C2 | CUDA / FP32 / NCCL / world size 2。DDP | 最大絶対誤差 0。sample id は rank 間で交わらない。二つの UUID は異なる。rank 0 の RNG を rank 1 へコピーしたファイルは拒否。rank 1 の重みを保存前に変えると `ranks disagree` で、直前の `step_00000001.pt` の SHA-256 は残る。見える GPU が 1 枚の world size 2 は拒否し、process group timeout 10 秒で終わる |
| C3 | CUDA / AMP FP16 / world size 1。autocast。`init_scale` 65536、`grad_clip` 1、学習率 0.01 | logical step 1 の直前に scale を 131072 へ上げ、有限 loss のまま skip させた。step 2 の checkpoint は更新回数 1、scale 65536、`autocast_on` 1、`applied` は `[True, False]`。Scaler を消したファイルは `grad scaler` で拒否。logical step 2 の直前に scale を 1024 へ戻したので、再開後の `applied` は `[True, False, True, True]`、更新回数は 3。最大絶対誤差 0 |
| C4 | CUDA / AMP FP16 / NCCL / world size 2。蓄積 2、micro-batch 1、例 16、`init_scale` 65536 | 再開の最大絶対誤差 0。両実行の GradScaler state は一致。sample id は交わらない。別試験で rank 1 だけが logical step 1 の勾配を無限大にし、`init_scale` 1024、`grad_clip` 1。`applied` は `[True, False, True]`、更新回数 2、scale 512、Scheduler の `last_epoch` は 2 |

C3 で 1024 へ戻すのは、この小さな `Linear` では既定の 65536 のまま後続 batch の FP16 勾配が溢れることがあるためである。契約の確認は、意図した overflow が全 rank で skip になり、その後の有限 scale が再開も含めて更新されることである。

PyTorch 2.6 の `LRScheduler` は構築時に `step()` を呼ぶ。`StepLR` の `last_epoch` は構築直後が 0 である。C4 の成功 2 回で 2 になり、skip した step は Scheduler を進めていない。

## 非決定性

この `Linear` と決定的アルゴリズム、同じ物理 GPU、`CUBLAS_WORKSPACE_CONFIG=:4096:8` では最大絶対誤差 0 だった。決定的 backward が無い CUDA kernel を含むモデルへ、この一致を広げない。別 UUID の GPU への RNG 再開は拒否する。CUDA RNG を戻せることと、非決定的 kernel のビット一致は別である。

## 大規模モデルでの制約

- overflow 合意は PyTorch 2.6.0 の private 辞書に依存する。版を上げるときはこの辞書の有無を先に確認する。
- rank 比較は Optimizer state 全体を `all_gather_object` する。本格 SE の保存方式にはしない。
- プロセスにつき Scaler は一つである。複数 Optimizer の実学習と GAN の更新順は未実装である。
- world size を変える resume、BF16、FSDP、sharded checkpoint は拒否するか、実装していない。
- process group の初期化中に rank が死んだ場合は、設定した timeout で終わる。それより強いハング回避は検証していない。

## 回帰

GPU 試験は `tests.experiments.test_cuda_amp` の 11 件が 121.195 秒で OK だった。CPU 回帰は `CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/experiments -v` が 43 件、108.145 秒、OK（skipped=1。CUDA を隠したため `test_cuda_amp` の module が `NOT_RUN`）。代表の `tests.training.test_rng_streams`、`test_batch`、`test_accumulation`、`test_protocol` は 16 件、27.131 秒、OK だった。
