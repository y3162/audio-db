# P10-C Fix.2-D GPU 検証

日付: 2026-11-08。ホスト `tensor`。GPU は物理 1, 2, 6, 7（A100 40GB）。本格 epoch は実行していない。本格 YAML は書き換えていない。スモークの上限は別 YAML に置いた。

重みは scratch の初期値から、再開試験は 3 optimizer 更新、validation 試験は 1 optimizer 更新のあとである。損失は収束後の性能ではない。

共通の入口は `data/results/training/reproduction/fix2d/gpu_driver.py`。これは公式の `execute` を呼び、rank ごとの case id とパラメータ checksum だけを追加で書く。`CUDA_LAUNCH_BLOCKING` は付けていない。`AUDIO_DB_STEP_SYNC=1`。`deterministic: false`。

CUDA illegal memory access は、今回の 4 モデルの実行では出ていない。`NOT_REPRODUCED` のままである。OOM は無い。

## 環境

| 項目 | 値 |
|---|---|
| HEAD | `b2a3549a12c983b5b702d38727dba6c5a1c62913` |
| Python | 3.12.13 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| DuckDB | 2.0.0 |

## MP-SENet

世界 2。`CUDA_VISIBLE_DEVICES=1,2`。seed 1234。global 4、micro 2、蓄積 1。公式損失。Generator と Discriminator。

| 実行 | 出力 | 結果 |
|---|---|---|
| 連続 3 更新 | `data/results/training/reproduction/fix2d/mp_senet_resume_continuous` | `INTERRUPTED`。`complete.json` は無い。peak allocated 11,400,257,024 |
| 別プロセス再開 | 同じ YAML で `step_00000001.pt` から step 3 | case id は rank 0 と rank 1 の step 2 と 3 で連続実行と一致。両 rank の checksum 差は 0 |
| 全 824 validation | `.../fix2d/mp_senet_valid` | rank 0 が 824 件。ユニーク 824。重複 0。192.8 秒。PESQ `ok`、count 824、値 1.183。Generator 損失 1.817。peak reserved 33,428,602,880 |

再開後のパラメータ総和は、連続実行と完全には一致しない。step 3 の Generator 総和差は約 0.046。両 rank の差は同じで、rank 間の分岐ではない。`deterministic: false` の範囲である。

ExponentialLR は epoch 末だけ進む。1 更新の validation checkpoint の学習率は 0.0005 のまま、scheduler の `last_epoch` は 0。

TensorBoard は rank 0 だけが `mp_senet_valid/tensorboard/events.out.tfevents.*` と、再開試験側の連続実行ディレクトリに書いた。

全長 validation の回帰は `tests/training/test_mp_senet_validation_full_length.py`。GPU の 824 件は hop 切り詰めなしの経路で PESQ まで終わった。1 件の再描画は長さ 27,861 で、hop 倍数へ切っていない。

## MossFormerGAN

世界 2。`CUDA_VISIBLE_DEVICES=6,7`。seed 777。global 4、micro 1、蓄積 2。

`distribute` の記録: SyncBatchNorm 8。`validation_requires_all_ranks=true`。`validation_order=rank_mod_fixed_v1`。Generator の BatchNorm は 0 のままである。

| 実行 | 結果 |
|---|---|
| 連続 3 更新 | `official_step` 6、`optimizer_updates` 3、失敗 micro-batch 0。学習率は Generator 5e-4、Discriminator 1e-3。peak allocated 12,628,829,696。rank 間の checksum 差は 0。case id の rank 間交差は 0 |
| 別プロセス再開 | step 2 と 3 の case id が連続実行と一致。rank 間 checksum 差は 0 |
| 824 件 validation | `.../fix2d/mossformergan_valid`。103.2 秒。rank 0 と rank 1 が各 412、ユニーク、交差 0、和 824 |

MossFormer は data epoch の末で validation する。`validation_interval: 1` だけでは 1 更新のあと検証しない。スモーク YAML だけ `data.max_updates_per_epoch: 1` を足し、その 1 更新で epoch を終えてから全 824 件を見た。本格 YAML にはこの上限は無い。

validation の被覆は `per_rank_batches=[412, 412]`、`included_batches=823`、`pesq_failed_batches=1`。Generator 損失は有限で count 824。失敗 batch は損失 0 としては記録していない。PESQ 集計そのものは `na`（ゲートであり plateau の指標ではない）。全 rank スキップの仕様は維持している。

## TF-GridNet

世界 2。`CUDA_VISIBLE_DEVICES=6,7`。seed 0。global 4、micro 2、蓄積 1。`num_cache_chunks=1024`。公式 `mr_l1_tfd`。Adam、学習率 0.001。

| 実行 | 結果 |
|---|---|
| 連続 3 更新 | rank 間 checksum 差 0。case 交差 0。peak allocated 2,743,491,584 |
| 別プロセス再開 | step 2 と 3 の case id が一致。総和差は 1e-5 から 3.5e-4。rank 間は一致 |
| validation | `.../fix2d/tf_gridnet_valid`。355.5 秒。peak allocated 2,723,641,856 |

validation 計画は更新 2,887、破棄 batch `(0, 21)`、`per_rank_cache=512`、plan hash `734a3f63a5b5d554b1ee3319d3378c5eed6d4a8d2094da1aff312473b91f46d9`。実行した batch は各 rank 2,887。chunk は各 rank 5,774。集約の重みは 11,548。`valid.loss` は有限。

valid 1,835 件のうち実行した case は 1,829。計画の保持集合と一致した。残り 6 件は整列前の raw 計画にだけあり、短い rank に合わせる `iterator_stop` で落ちた末尾である。欠落ではなく、その規則である。公式 WAV とのビット一致は測っていない。

scheduler は validation のあと 1 回進み、`last_epoch` は 1。最初の validation なので学習率は 0.001 のままである。`ave_5best` は作らず、`valid.loss.ave_1best.pt` を `1epoch.pt` へ向けた。

## TF-Locoformer

世界 4。`CUDA_VISIBLE_DEVICES=1,2,6,7`。seed 0。global 4、micro 1、蓄積 1。研究用 snapshot。公式幅。`num_cache_chunks=1024`。過去の GPU smoke の 128 とは違う。

3 更新の train plan hash は `4bd4f9b0e9295c218256878102e483cc62eb03f0b806d790afa0854823da4941`。`per_rank_cache` は 256。これは cache 1,024 を 4 で割った値である。

| 実行 | 結果 |
|---|---|
| 連続 3 更新 | 4 rank の checksum 差 0。case 交差 0。学習率は更新後 2.5e-7、5e-7、7.5e-7。warmup 4,000 の `0.001 * step / 4000` と一致。peak allocated 16,470,292,992 |
| `rng_by_rank` | 4 要素。CUDA device は各 rank の 0, 1, 2, 3 |
| 別プロセス再開 | 4 rank とも step 2 と 3 の case id が一致。rank 間 checksum 差は 0。連続実行との総和差は 1e-7 台。peak allocated 16,531,107,840 |

### 1,024 件の validation

出力は `data/results/training/reproduction/fix2d/tf_locoformer_valid`。train は `max_updates: 1` の 1 回だけ。validation cap は無い。モデルは未学習に 1 回の AdamW を足した状態である。この損失を収束後の性能として使わない。

計画 hash `c3d9de17ef756c15a58a4319ebefa496b02f30f086450d7e8eb2c0ac13c1fe32`。破棄 batch は 0, 0, 0, 0。`num_cache_chunks` は 1024。

| 項目 | 結果 |
|---|---|
| 実行 batch | 各 rank 3,584。計画と一致 |
| chunk | 各 rank 3,584。合計の重み 14,336 |
| case | 各 rank ユニーク 256。各 case はちょうど 14 chunk。和 1,024。rank 間の交差 0。欠落 0 |
| `valid.loss` | 有限。9,057.878724438804。count 14,336 |
| 非有限 | 完走した。非有限で停止していない |
| renderer | エラーで停止していない |
| OOM | 無し。peak allocated 16,326,225,920 |
| 壁時計 | 628.9 秒 |
| 終了 | 4 rank とも正常。`INTERRUPTED`。`complete.json` は無い |

実行した case id は `logs/tf_locoformer_valid/validation_cases_rank{0,1,2,3}.jsonl` にある。計画ハッシュとの一致に加え、このログの集合が 1,024 件であることを確認した。

`valid.loss.ave_5best.pt` は無い。有限な validation が 1 件なので `valid.loss.ave_1best.pt` が `1epoch.pt` を指し、`valid.loss.ave.pt` はそれを指す。

## n-best と ave_5best

150 epoch は実行していない。`data/results/training/reproduction/fix2d/nbest_evidence.py` が、実 TF-Locoformer の推論 state dict 174 キーを、Trainer の `_retain_chunk_models` に 6 epoch 分渡した。

| 確認 | 結果 |
|---|---|
| 順位 | `valid.loss` の昇順。epoch 4, 2, 5, 1, 3。損失 0.1, 0.2, 0.4, 0.5, 0.9。epoch 6（損失 1.5）は平均に入らずファイルも消える |
| 平均 | 選ばれた 5 ファイルから再計算した平均と、`valid.loss.ave_5best.pt` の最大絶対差は 0 |
| 読み込み | 別インスタンスの `load_state_dict(..., strict=True)` のあと、最大絶対差は 0 |
| 再開 | 4 epoch の履歴を渡して 5 epoch 目を足すと、同じ 5 件の平均ができる |
| 保持 | `prune_run_checkpoints` は latest 3（step 4, 5, 6）、best の step 1、n-best の epoch 1–5、平均ファイルを残し、step 2 と 3 と epoch 6 を落とす。別ディレクトリは触らない |
| 中断 | 完成済み checkpoint の横に不完全な一時ファイルを置いて消しても、完成ファイルのバイト列は残る |

これは fixture である。本格 run の checkpoint は使っていない。
