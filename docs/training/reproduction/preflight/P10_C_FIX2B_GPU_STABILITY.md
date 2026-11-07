# P10-C Fix.2-B CUDA 安定性

Fix.2-A の更新規則のまま、公式幅の MossFormerGAN を 2 GPU で短時間動かした。本格の 121 epoch は始めていない。過去の PASS は今回の結果に数えていない。

## GPU 実行条件

| 項目 | 値 |
|---|---|
| 日付 | 2026-11-08 |
| HEAD | `f31829ccb2ed62da2cb2fc45a5afd30d609bc905` |
| PyTorch | `2.6.0+cu124` |
| CUDA / cuDNN | 12.4 / 90100 |
| 物理 GPU | 6 と 7。`CUDA_VISIBLE_DEVICES=6,7` |
| 起動 | `CUDA_LAUNCH_BLOCKING=1`、`torch.distributed.run --nproc_per_node 2` |
| モデル | 公式幅 `se_16k`。初期化は G1 と D4 |
| バッチ | world 2、global 4、micro 1、蓄積 2、crop 32,000 |
| データ | `mossformergan.vb28.ondemand.official@1`、seed 777、オンデマンド描画 |
| plan hash | `13244439cac19d7548884593287bbeb3a013747fa7ba50aef7a6a54169ed107a` |
| SyncBatchNorm | Discriminator 8。Generator 0 |
| 出力 | `/tmp/p10c-fix2b/`。既存の `data/checkpoints` と `data/results` は書いていない |

rank 間のパラメータ許容は、実行前に絶対差 `1e-6` とした。観測した最大差は 0 だった。

## R5 の再現条件

同一プロセスグループで、次を 1 回とした。繰り返す前にモデルは作り直す。

1. 公式幅を 2 GPU DDP で初期化する。
2. batch を取る。合成は `(1, 32000)` の乱数。実データは Dataset の update 0。
3. 成功 micro-batch を 2 つ処理し、Optimizer を 1 回更新する。
4. rank 0 の PESQ だけを失敗させる。rank 1 は計算し、全 rank が backward しない。
5. 続く成功 2 つで蓄積窓を埋め、Optimizer を更新する。
6. 同じプロセスで Validation forward を 1 batch 行う。
7. Discriminator の SyncBatchNorm forward を行う。入力は `(1, 1, 201, 19)`。
8. rank 0 が checkpoint を書く。
9. 次の学習 micro-batch へ進む。

同期境界は、各段階の前と後で `torch.cuda.synchronize()` し、続けて barrier した。最後に成功した名前は rank 別の jsonl に残した。

反復は実行前に、合成 3 回、実データ 3 回と決めた。失敗した回を集計から外していない。6 回とも最後の `next-update` まで戻った。

| 回 | 入力 | 終了 | 処理 | 成功 | 失敗 | Optimizer | `official_step` | rank 差 |
|---|---|---|---|---|---|---|---|---|
| 1–3 | 合成 | PASS | 6 | 5 | 1 | 2 | 6 | 0 |
| 4–6 | 実 batch | PASS | 6 | 5 | 1 | 2 | 6 | 0 |

rank 0 の失敗理由は `pesq`。rank 1 は 6 回とも `peer`。illegal memory access、NCCL watchdog、プロセス死は 0。状態は `NOT_REPRODUCED`。原因は不明なので `RESOLVED` にしない。

合成の成功段は、雑音に対する実 PESQ が毎回欠測して更新が空になるのを避けるため、目標を 0.5 に固定した。失敗段は rank 0 だけ `None` を返す。実 batch の成功段は `pesq` パッケージの wideband を使った。

## 824 件 Validation

R5 の 6 回のあと、同じプロセスで公式幅の新しいモデルを作り、`validation_refs` の 412 件を両 rank が順に `validation_report` した。

| rank | 計画 | 完了 | case id が計画と一致 |
|---|---|---|---|
| 0 | 412 | 412 | はい |
| 1 | 412 | 412 | はい |

これは world 2 の公式 rank 分割である。1 batch の R5 Validation や、後段の `validation_max_batches: 2` は、この 824 件の証拠ではない。

R5 の 1 batch Validation は、実 batch の回で `pesq_gate=included` だった。同期境界 `validation-after` のあとで落ちていない。

## PESQ と非有限損失

実波形 1 本を使い、PESQ の成否だけを固定して状態遷移を見た。各条件の前に蓄積窓を捨てている。重みが変わらなかった条件は、失敗だけで Optimizer を進めていない。

| 条件 | 処理 | 成功 | 失敗 | Optimizer | `official_step` 増分 | 窓 | 重み |
|---|---|---|---|---|---|---|---|
| 全 rank 成功 | 2 | 2 | 0 | 1 | 2 | 0 | 変化 |
| rank 0 だけ失敗 | 1 | 0 | 1 | 0 | 1 | 0 | 不変 |
| rank 1 だけ失敗 | 1 | 0 | 1 | 0 | 1 | 0 | 不変 |
| 両 rank 失敗 | 2 | 0 | 2 | 0 | 2 | 0 | 不変 |
| 成功、失敗、成功 | 3 | 2 | 1 | 1 | 3 | 0 | 変化 |
| 失敗、成功、成功 | 3 | 2 | 1 | 1 | 3 | 0 | 変化 |
| Generator 損失が非有限 | 1 | 0 | 1 | 0 | 1 | 0 | 不変 |
| Discriminator 損失が非有限 | 1 | 0 | 1 | 0 | 1 | 0 | 不変 |

rank 差はすべて 0。片 rank の失敗で相手が collective に残った記録は無い。

## 共通 Trainer の実データ更新

本格 YAML と同じモデル、seed、Dataset、world size、global batch 4、micro 1、蓄積 2。違いは検証用の上限と checkpoint 間隔だけで、中断と再開でその YAML は変えていない。`verification_max_updates: 3`、`checkpoint_interval: 1`。121 epoch の `350053` 更新には届かせていない。

中断なしの rank 0 は 3 更新とも rank 差 0。学習率は Generator `5e-4`、Discriminator `1e-3`。clip 後の勾配 norm は 12 回の backward で最大 10.0 で、閾値 10 に当たった段がある。`official_step` は 6、失敗は 0、`accu_count` は 0。

case id は両 rank の計画に従う。rank 0 の 3 更新は次のとおりである。

1. `1664c3fe473bbb2b5914685c9d47c10fffd044ee41117d1426271761bb3ad109`、`4e0814eb6a79ca84b6e339c1771cebb3a119617a50460479eabbdf5c2a17da76`
2. `a6659dde3e925b6e028b126b67a24d499983afae3ddb114a3056d5ed397b0123`、`b80af1ea1a306473569806c29e7da542591bb48d2c659fa956aef928c9b43ee5`
3. `b530fa945062cb805020815ee208ad7499e55e0edf1c65ff9b6c8b91332fc69d`、`4ea0453c669de740f1a8b9f2bf0b1cd091af950ef38eca0a528eb3e22b0e9091`

## Epoch 末

最初の設定は `verification_max_updates: 2` かつ epoch 内 3 グループで、成功 2 回の時点で上限に達し、失敗グループと Validation の前に止まった。これは上限の置き方が境界に届かない設定で、製品の欠測ではない。記録から外していない。

続けて上限を 3 にした。epoch 内 3 グループ、最後のグループは両 rank が PESQ 失敗。結果は次のとおりである。

| 時点 | `global_step` | `official_step` | 成功 | 失敗 | Optimizer |
|---|---|---|---|---|---|
| 失敗グループの直後、Validation 前の step 2 checkpoint | 2 | 6 | 4 | 2 | 2 |

Validation はその境界で 1 回確定した。損失は 0.4218636006116867。Plateau の最良値は同じ数、stalls は 0、Generator の学習率は `5e-4` のまま。件数は smoke の rank あたり 2 件で、824 件の分割ではない。

残余勾配は別実行である。1 グループの 2 個目を失敗させ、`accu_count` が 1 のまま epoch が終わった。次の epoch 先頭の `start_epoch(resume_mid_epoch=False)` は、蓄積 1 と勾配ありを、蓄積 0 と勾配なしに戻した。その epoch でも Validation が走り、損失 1.2487203478813171 が Plateau に入った。続く更新のあと 0.5897715613245964 に更新された。学習率は半分になっていない。

## DDP の異常終了

検証プロセスだけを対象にした。process group の timeout は 20 秒。本番の `setup_process` は timeout を渡さない。この環境の NCCL 既定は 10 分、gloo 既定は 30 分である。

| 条件 | 経過 | 終了コード | 相手 rank |
|---|---|---|---|
| rank 0 が forward 前に例外 | 11.3 秒 | 1 | Validation に入ったあと終了 |
| rank 1 が forward 前に例外 | 9.3 秒 | 1 | 同上 |
| rank 0 が SyncBatchNorm の forward で例外 | 11.0 秒 | 1 | 両 rank が forward に入ったあと終了 |
| rank 1 が SyncBatchNorm の forward で例外 | 10.5 秒 | 1 | 同上 |

driver が 180 秒で殺した回は無い。無期限の待ちにはならなかった。本番経路の待ち上限は、今回の 20 秒ではなく NCCL 既定の 10 分である。

checkpoint を書く前の `SIGKILL` では、`step_*.pt` も一時ファイルも残らなかった。`status.json` も無い。書いたあとの `SIGKILL` は再開の文書に書いた。

## メモリ

R5 の rank 0 で観測した `max_memory_allocated` の最大は 12,805,912,576 バイトだった。epoch 末の Trainer 実行ログに出た peak allocated は 23,491,702,784 バイトで、rank 間の重み比較のための複製を含む。

## 原因

illegal memory access の発生箇所は、今回の境界ログでは特定できない。6 回の系列と 824 件 Validation と Trainer の更新は、同期点を超えて戻った。CPU で落ちなかったことは根拠にしていない。これらは CUDA 上の結果である。
