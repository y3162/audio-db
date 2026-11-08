# P10-C Fix.2-E GPU 実行

日付: 2026-11-08。ホスト `tensor`。CPU 全件は、時間比較のためには再実行していない。

## 過去ログ

個別テストの秒数は、`/tmp/fix2d-pytest-final5.log` にも `/tmp/fix2e-final.log` にも無い。どちらも pytest の合計だけである。

| ログ | 範囲 | 結果 | 秒 |
|---|---|---|---:|
| `/tmp/fix2d-pytest-final5.log` | `tests/training` と `tests/reproduction` | 411 passed、11 failed、14 skipped | 2494.63 |
| `/tmp/fix2e-final.log` | 同じ。`CUDA_VISIBLE_DEVICES` は空 | 419 passed、14 skipped | 2472.09 |
| `/tmp/fix2e-dfn.log` | kernel、DeepFilter 3 件、TensorBoard の実モデルと 2 rank、preview | 10 passed、16 subtests | 1247.88 |

この 1247.88 秒はグループの合計である。中の 1 件が何秒だったかはログに無い。ソースにあった「MossFormerGAN が 669 秒」は、このログの行ではないので実測としては使わない。

## 分類

GPU が見えるとき、その GPU で forward / backward / optimizer を実行する。

- `tests/training/test_model_tasks.py` の実 step。例外は下の CPU 固定
- TensorBoard の `test_models`、`test_other_real_models_two_ranks_validation`、`test_real_gridnet_two_ranks_validation`
- `test_small_models_forward`
- `test_p8_dfn3.py` の DeepFilterNet3 1 step
- `test_p9_semamba.py` の SEMamba++ 1 step（48,000 sample、縮小 architecture）

2 rank は、見える GPU が 2 枚以上のとき NCCL である。1 枚以下は従来の CPU gloo である。`AUDIO_DB_TB_DEVICE` と `AUDIO_DB_TB_DDP_DEVICE` があればそちらを使う。

CPU のままにした。

- 公式 CPU 数値照合。TF-Locoformer、MP-SENet、SEMamba の mel / MRD、libdf、RFFT kernel、ESPnet chunk
- mock と mock の 2 rank
- gloo を確認する MossFormer の SyncBatchNorm と PESQ skip
- DB、snapshot、来歴
- SEMamba++ の micro-batch と 1 batch の勾配比較。同じ seed の CUDA で generator 勾配の相対差が約 2 だった。許容誤差を広げると、蓄積が一致していることの確認にならない

既に GPU 必須なものは変えていない。`test_moss_probe_accumulation.py` と `test_df_reference_spectral.py` は CUDA が無いと SKIP のままである。

GPU 上の今回の試験は `use_deterministic_algorithms` を有効にしない。`reflection_pad1d` と `adaptive_max_pool2d` の CUDA backward に決定的実装が無い。確認するのは step の完走、再読込、有限値、rank の一致である。`deterministic: true` の本番学習とは別である。

## 変更後の実測

GPU は物理 0 と 1。`CUDA_VISIBLE_DEVICES=0,1`。ログ `/tmp/fix2e-gpu2.log`。pytest 154.33 秒、壁時計 157 秒。16 passed、1 failed、16 subtests passed。失敗は SEMamba の蓄積比較だけで、その後 CPU に固定した。

| テスト | call | 装置 |
|---|---:|---|
| TensorBoard 2 rank の実モデル 6 件 | 80.77 秒 | GPU / NCCL |
| TF-GridNet の小さい 2 rank | 9.30 秒 | GPU / NCCL |
| TensorBoard の実モデル 5 件 | 8.64 秒 | GPU |
| P9 の SEMamba 1 step | 5.85 秒。setup 23.53 秒はデータ展開 | GPU |
| mock の 2 rank | 5.19 秒 | CPU のまま |
| GridNet の小さい step と再読込 | 2.68 秒 | GPU |
| MossFormerGAN 公式幅 1 step | 1.92 秒 | GPU |
| 5 モデルの forward preview | 1.45 秒 | GPU |
| SEMamba++ の有限 step | 1.38 秒 | GPU |
| GridNet 公式幅 1 step | 0.82 秒 | GPU |
| DeepFilterNet3 公式幅 1 step | 0.54 秒 | GPU |
| P8 の DeepFilterNet3 1 step | 0.36 秒 | GPU |
| DeepFilter の cosine 1 step | 0.26 秒 | GPU |
| DeepFilter の縮小 1 step | 0.16 秒 | GPU |
| GridNet urgent24 1 step | 0.05 秒 | GPU |

SEMamba の蓄積比較は、その失敗のあと CPU で再実行した。GPU が見えるプロセスで 6.94 秒、`CUDA_VISIBLE_DEVICES` が空で 7.16 秒。どちらも 1 passed。ログは `/tmp/fix2e-acc-gpuvisible.log` と `/tmp/fix2e-acc-cpu.log`。

CPU フォールバックの代表は、GridNet の小さい step である。`CUDA_VISIBLE_DEVICES` を空にして 1.41 秒、1 passed。ログ `/tmp/fix2e-cpu-fallback.log`。同じテストの GPU call は 2.68 秒で、この小さいモデルでは GPU の方が短いとは言えない。条件も違うので、速度比にはしない。

1247.88 秒の CPU グループと、154.33 秒の GPU 実行は、含まれるテストが同じではない。前者には RFFT kernel があり、後者には P8、P9、GridNet の 2 rank、mock の 2 rank がある。厳密な前後比較ではない。
