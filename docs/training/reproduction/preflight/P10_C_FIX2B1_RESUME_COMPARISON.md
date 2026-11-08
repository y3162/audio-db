# P10-C Fix.2-B.1 再開の数値比較

比較の前に許容と試行回数を固定し、`/tmp/p10c-fix2b1/plan.json` に書いた。そのあとで学習プロセスを起動した。

| 対象 | atol |
|---|---|
| パラメータ | `1e-3` |
| Adam moment | `1e-3` |
| BatchNorm 浮動 buffer | `1e-4` |
| 蓄積勾配 | `1e-4` |
| rank 間のパラメータと moment | `1e-6` |
| 保存直前と読み込み直後 | `0` |

離散状態は完全一致である。同じ checkpoint からの再開は、この atol 以内であるか、その同じ checkpoint から再開した 2 回の差の 2 倍以内である。2 倍は Fix.2-B の継続ノイズの規則であり、許容そのものは広げていない。独立した 2 回の連続実行の差は、再開誤差の分母にしない。

試行は、モードごとに連続 2 回、同じ step 1 checkpoint からの再開 2 回、更新上限 3 である。seed 777、world size 2、`mossformergan.vb28.ondemand.official@1`、global 4、micro 1、蓄積 2。

## 実行環境

| 項目 | 値 |
|---|---|
| 日付 | 2026-11-08 |
| HEAD | `ff5a5abfe0d2fbe1920c0304cd285ba853cbc8ce`（比較時の作業木は修正済み） |
| PyTorch / CUDA / cuDNN | `2.6.0+cu124` / 12.4 / 90100 |
| 物理 GPU | 0 と 6。`CUDA_VISIBLE_DEVICES=0,6` |
| 起動 | `CUDA_LAUNCH_BLOCKING=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.distributed.run --nproc_per_node 2` |
| 出力 | `/tmp/p10c-fix2b1/`。既存の `data/checkpoints` と `data/results` は書いていない |

`CUBLAS_WORKSPACE_CONFIG` は CLI がプロセス起動時に置くのと同じである。`deterministic: false` でも cuDNN の決定論フラグは立てない。TF32 は許可したままである。

## deterministic false

4 本とも終了コード 0、`INTERRUPTED`、`global_step` 3、`official_step` 6、`accu_count` 0 だった。連続 a は 30 秒、連続 b は 33 秒、再開 1 は 33 秒、再開 2 は 24 秒である。

同じ step 1 から再開した step 3 と、連続 a の step 3 を比べた。再開 1 と再開 2 の差を継続ノイズとした。

| 項目 | 連続 a 対 再開 1 | 再開 1 対 再開 2 | 独立した連続 2 回 | atol |
|---|---:|---:|---:|---:|
| Generator | 5.50e-4 | 5.35e-4 | 1.18e-3 | 1e-3 |
| Discriminator | 9.55e-4 | 9.55e-4 | 1.35e-3 | 1e-3 |
| Adam Generator | 1.38e-5 | 2.15e-5 | 6.95e-5 | 1e-3 |
| Adam Discriminator | 2.06e-4 | 2.27e-4 | 3.75e-4 | 1e-3 |
| BN 浮動 buffer | 2.16e-5 | 3.00e-5 | 5.93e-4 | 1e-4 |
| 蓄積勾配 | 0 | 0 | 0 | 1e-4 |

step 3 は窓が閉じているので勾配 tensor は空である。step 2 の BN 浮動差は 0、Generator は 2.54e-4、Discriminator は 4.60e-5 だった。

`rng_by_rank` は連続 a、再開 1、再開 2 で一致した。従来の `rng` 一覧は、再開後の未使用 device を戻さないので一致しない。学習 device の一致は `rng_by_rank` で見ている。

再開直後の最初の Generator forward の SHA-256 は、連続実行の 2 回目の更新の最初の forward と、rank 0 も rank 1 も一致した。

更新後の rank 間差は、パラメータも Adam の浮動 moment も 0 だった。

独立した連続 2 回の Generator `1.18e-3` と Discriminator `1.35e-3` は atol を超える。これは step 0 からの cuDNN の積であり、同じ checkpoint からの再開差より大きい。再開差は atol 以内である。

## deterministic true

`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、TF32 無効、SDPA は math のみ、を CUDA 初期化の前に置いた。`warn_only` は使っていない。

最初の backward で停止した。

```text
adaptive_max_pool2d_backward_cuda does not have a deterministic implementation
```

箇所は `loss.backward()` の中である。Discriminator の `CNN` が `F.adaptive_max_pool2d` を 5 回呼ぶ。出力サイズは公式の `[101, 7]`、`[50, 7]`、`[25, 5]`、`[12, 5]`、`[6, 3]` であり、全体を 1 に潰す pool ではない。MP-SENet の `amax` 置換は値を変えない global max だけに使っており、この層へは広げていない。

連続 true の step 3 は無い。true の再開比較も無い。本格 YAML は `deterministic: false` のままである。

## rank 別 RNG

修正前の 1 更新では、両 rank の現在 device は同じハッシュまで消費され、checkpoint の slot 1 は seed 直後のままだった。修正後の読み込み直後は、rank 0 の現在 device 0 と rank 1 の現在 device 1 が、それぞれ `rng_by_rank` の CUDA 状態と一致した。Python、NumPy、Torch CPU も一致した。

## BN buffer

修正前の保存直前、8 個の SyncBatchNorm は rank 間で一致した。読み込み直後も一致した。修正後の同じ checkpoint 再開では、step 2 の浮動 buffer 差は 0、step 3 は `2.16e-5` で atol `1e-4` 以内だった。Fix.2-B の `cnn.bn6.running_var` `1.45e-2` は、rank 1 の Dropout 列が戻ったあとの差であり、buffer の保存欠落ではなかった。

## Optimizer と蓄積勾配

読み込み直後の Adam moment は最大絶対差 0 だった。`accu_count=1` の再開では、epoch ログの再開直後が蓄積 1、勾配ありのままだった。epoch 境界の次 epoch では、蓄積 1、勾配ありから、蓄積 0、勾配なしへ戻った。

独立実行どうしの未完了勾配差は Generator `3.88e-4`、再開との差は `3.95e-4` で、連続差の 2 倍以内だった。絶対 atol `1e-4` は連続実行どうしでも超える。同じ checkpoint からの step 3 では窓が閉じており差は 0 である。

## CUDA 安定性

既知の R5 を、修正後のプロセスで合成 3 回、実 batch 3 回、続けて 824 件の Validation まで走らせた。6 回とも PASS、rank 間のパラメータ差は 0、illegal memory access は 0 だった。各 rank の Validation は 412 件で、計画の case id と一致した。終了コードは 0、経過は 198 秒である。状態は `NOT_REPRODUCED` のままである。

`SIGTERM` は step 1 の `CHECKPOINTED` になり、別プロセスが step 3 まで進んだ。case id は連続実行と一致した。`SIGKILL` のあと step 1 は残り、別プロセスが再開した。`validation_pending` の再開は `global_step` 2 のまま Validation を 1 回だけ確定した。確定行は 1 行、損失は 0.3866295932792127 だった。保存前に殺した初期化では step ファイルも一時ファイルも残らなかった。

片 rank の意図した例外 4 本は、9.3 秒から 11.7 秒で相手も終了した。
