# P10-C Fix.2-B 実装

MossFormerGAN の 2 GPU 学習で、Fix.2-A の更新規則を維持したまま、CUDA 上の一連の更新、Validation、Checkpoint 再開を短時間で確認した。121 epoch の本格学習は始めていない。第三者検証は始めていない。

開始時の `HEAD` は `f31829ccb2ed62da2cb2fc45a5afd30d609bc905`、ブランチは `feature/reproduction`。公式チェックアウトは `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` のまま、公式ソースは変更していない。PyTorch は `2.6.0+cu124`、CUDA は 12.4、cuDNN は 90100。実行 GPU は物理 6 と 7（`CUDA_VISIBLE_DEVICES=6,7`）。`CUDA_LAUNCH_BLOCKING=1`。

## 対象 Issue

- P10P-D-005。更新後の同一プロセス Validation で出ていた CUDA illegal memory access。Fix.1 の再実行 1 回では再現していなかった。
- 2 GPU の正式な中断。`SIGTERM` のハンドラは rank 0 だけに付いていた。`segment_steps` と `deadline` は world size 1 のまま拒否される。
- Checkpoint からの別プロセス再開、未完了の蓄積窓、`validation_pending`。

## 再現した問題

CUDA illegal memory access は、事前に決めた 6 回（合成 3、実 VoiceBank batch 3）と、その後の 824 件 Validation、共通 Trainer の短時間実行では再現しなかった。`NOT_REPRODUCED` であり、`RESOLVED` ではない。

再現して直したものは、2 GPU の `SIGTERM` である。ハンドラが rank 0 だけだと、torchrun が全 rank に送る `SIGTERM` で rank 1 は既定動作の終了になり、rank 0 だけが更新境界まで進む。停止フラグも rank 間で共有されないので、届くタイミングがずれると次の collective で分かれる。

## 修正内容

`validate_on_data_epoch_end` を持つタスク（MossFormerGAN）は、world size が 2 以上のとき全 rank が `SIGTERM` を受ける。データ epoch の更新境界で、停止フラグを `MAX` の all-reduce にかけ、全 rank が同じ境界で checkpoint して止まる。`deadline` と `segment_steps` は world size 1 のままである。他モデルのループにはこの collective を足していない。

Fix.2-A の更新順、増分平均、片 rank PESQ 失敗の全 rank スキップ、`official_step` と `global_step` の分離、`broadcast_buffers=True`、`find_unused_parameters=True`、CUDA 上の Discriminator SyncBatchNorm は変えていない。

保証する中断点は次のとおりである。

- Trainer が Optimizer 更新を受理した境界。ここでの `SIGTERM` は、その境界の checkpoint のあとで全 rank が止まる。
- データ epoch 末。Validation の前に `validation_pending` 付き checkpoint を書き、Validation と Plateau のあとで pending の無い checkpoint を書く。
- 蓄積窓が埋まる前の勾配は、その境界の checkpoint の `accumulation` に入っているときだけ復元する。命令の途中からは戻さない。
- `SIGKILL` は書き終わった checkpoint を残す。保存中の一時ファイルは完成 checkpoint にならない。

## 変更ファイル

- `src/training/core/trainer.py`
- `tests/training/test_mossformer_stop_agreement.py`
- `tests/training/p10c_fix2b_gpu.py`
- `.agents/DECISIONS.md`
- `docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX_ISSUES.md`

MP-SENet、TF-GridNet、TF-Locoformer、Dataset、DB、新しい学習 YAML、公式ソースは変更していない。検証用の YAML は `/tmp/p10c-fix2b/` にだけ書き、リポジトリの本格 YAML は変えていない。

## 自己テスト

CPU の `tests.training.test_mossformer_stop_agreement` は 2 件 PASS だった。rank 0 だけの停止フラグが world 2 の gloo で両 rank に伝わる。

GPU の手順と数値は `P10_C_FIX2B_GPU_STABILITY.md` と `P10_C_FIX2B_RESUME_VERIFICATION.md` に分けた。概要は次のとおりである。

- R5 系列 6/6 が PASS。illegal memory access は 0。
- 実 Dataset の 2 GPU 更新で、両 rank のパラメータ差は 0。
- `SIGTERM` で step 1 の `CHECKPOINTED` になり、別プロセスが同じ設定のまま step 3 まで進んだ。case id 列は中断なし実行と一致した。
- `validation_pending` の再開は Optimizer 更新をやり直さず、Validation を 1 回だけ確定した。
- 片 rank 例外の相手は 9–11 秒で終了した。無期限の待ちにはならなかった。

## 未解決事項

- CUDA illegal memory access の原因は分かっていない。今回の系列で再現しなかったことを修正済みにはしない。
- 再開と中断なし実行の離散状態は一致した。`deterministic: false` のままでは、Discriminator の BatchNorm `running_var` が事前の 2 倍規則を超えた。詳細は再開の文書。
- 公式 2 GPU の 2 回目の Discriminator backward が rank 間で割れないことは、Fix.2-A の `INTENTIONAL_DIFFERENCE` のままである。
- `segment_steps` と `deadline` は 2 GPU では使えない。
- 本番の process group は timeout を渡さない。NCCL 既定は 10 分である。
- グループ内の micro-batch がすべて失敗したとき、`pesq_skip` の損失欄は 0.0 になる。Validation の失敗 batch は 0 として足していない。学習の分母とは別である。
