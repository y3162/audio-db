# Phase 4-E：MossFormerGAN 公式学習アルゴリズムの移植

判定: **GO（CPU / Gloo）。** CUDA / NCCL の SyncBatchNorm は短い 2 GPU スモークで確認した。公式データ全量の学習と論文スコアの再現はしていない。

基準 commit は `6d800dc`（Phase 4-D）。対象ブランチは `feature/experiment-framework`。

## 1. 変更ファイル

- `src/experiments/models/mossformergan.py`：`MossFormerGANGenerator`（G1）と `MossFormerGANDiscriminator`（D4）
- `src/experiments/mossformergan.py`：`MossFormerGANTrainer`
- `src/experiments/runtime.py`：`clip_repeated` と `average_new_gradients`。既存の `clip` と `average_gradients` の意味は変えていない
- `configs/experiments/mossformergan_se_16k.yaml`
- `tests/experiments/test_mossformergan.py`
- `tests/experiments/mossformergan_data.py`
- `tests/experiments/mossformergan_worker.py`

`src/training/` の現行 Task、Loss、Discriminator は変更していない。Factory にモデル名の分岐は足していない。

## 2. G1/D4 初期化

ClearerVoice の `train.py` は wrapper を 2 つ作り、各 wrapper が SyncANet と Discriminator を作る。残すのは最初の Generator（G1）と 4 番目の Discriminator（D4）である。

Builder の stage 0 はコンポーネント id の順で作るので、Discriminator が Generator より先にできる。各クラスは自分の `seed_model_init(seed)` の中で必要な乱数だけを消費する。

- Generator は G1 だけを作る
- Discriminator は G1、D1、G2、D2、G3、D3、G4 を作って捨て、D4 を残す
- 捨てたモジュールは State Registry に登録しない
- G と D の Parameter は共有しない。共有すると `BuildError`

`MossFormerGANTask` が D2 と G3 の間で行う `generator.to(device)` は乱数を消費しない。Discriminator のコンストラクタでは繰り返していない。

seed 7、公式の `n_fft=400`、`num_channel=64`。Discriminator を先に作ってから Generator を作った。初期 Parameter と Buffer の最大絶対誤差は Generator も Discriminator も 0。初期化の前後でグローバル RNG は戻る。

## 3. 公式 Loss

現行の `MossFormerGANTask._generator_terms` をそのまま呼ぶ。式は変えていない。

\[
L_G = 0.1 L_{RI} + 0.9 L_{Mag} + 0.2 L_{Time} + 0.05 L_{GAN}
\]

Time Loss は clean の real と予測 imaginary から波形を作る現行の式である。Enhanced waveform の L1 には置き換えていない。

Discriminator の教師は real が 1、fake が `(PESQ - 1) / 3.5`。PESQ は `src.training.se.mossformergan.metric_targets` を呼ぶ。

AdamW の weight decay は PyTorch 既定の 0.01。YAML の 0.01 はコンストラクタへ渡す。0.01 でない Optimizer は `BuildError`。Generator の学習率は 0.0005、Discriminator はその 2 倍の 0.001。betas は `(0.9, 0.999)`。eps は `1e-8`。amsgrad は false。Scheduler オブジェクトは作らない。

## 4. Forward と Backward

成功した 1 micro-batch の順は現行 Task と同じである。

1. Generator forward と、detach しない fake magnitude に対する Discriminator forward。ここで Generator Loss を計算する
2. PESQ
3. PESQ が成功したら、detach した fake と real の Discriminator forward。Discriminator Loss を計算する
4. Generator Loss を蓄積回数で割って backward
5. 新しい勾配増分だけを rank 間平均
6. Generator の勾配を clip
7. Discriminator Loss を蓄積回数で割って backward
8. 新しい勾配増分だけを rank 間平均
9. Discriminator の勾配を clip
10. 成功数が蓄積回数に達したら Generator の AdamW、そのあと Discriminator の AdamW

両 backward が終わる前に、どちらの Optimizer も step しない。MP-SENet の D 先行 step は使わない。forward と backward は `sync_context(sync=False)` の中で行い、DDP の all-reduce は使わない。

`amp_fp16` は拒否する。micro-batch ごとに clip し、二つの AdamW は一つの GradScaler 決定にならない。

## 5. PESQ Skip

次のとき micro-batch を skip する。

- Generator Loss が非有限
- PESQ が無い、または計算に失敗して `metric_targets` が None
- Discriminator Loss が非有限
- 他 rank が同じ micro-batch を skip する（`runtime.agree(..., policy="any")`）

Skip で進むもの: data cursor、`official_step`、`micro_batches_processed`、`micro_batches_failed`。

進まないもの: 勾配、`accu_count`、Optimizer、`logical_step`。

失敗を成功として数えず、ゼロ Loss として backward しない。後続の成功は同じ蓄積窓を続ける。

カウンタの意味:

| 名前 | 意味 |
|---|---|
| `logical_step` | G と D の両方が step した回数 |
| `official_step` | 試した micro-batch |
| `data.update_index` | 次に読む micro-batch |
| `accu_count` | 開いている窓の成功数 |
| `optimizer_updates` | G と D それぞれの成立回数。両方同時に増える |

全 micro-batch が PESQ 失敗の有限 epoch は、その epoch の終わりで `halt_reason=no_successful_micro_batch` として停止する。次の epoch へ戻ってループしない。

## 6. Accumulation

成功した Loss は蓄積回数で割ってから backward する。Skip は窓を進めないので、窓は計画上の batch 境界をまたぐ。

epoch の終わりでは未完成の窓を捨てる。`zero_grad(set_to_none=True)` と `accu_count=0` である。現行の `start_epoch(resume_mid_epoch=False)` と同じである。

## 7. Gradient Clip

各成功 micro-batch の backward のあと、そのモジュールだけを clip する。蓄積の最後に一度だけ clip する方式ではない。

`runtime.clip` は同じ Parameter 集合の二度目を拒否する。この制約は残した。繰り返しは `runtime.clip_repeated` が行う。`clip_repeated` は AMP を unscale せず、`amp_fp16` を拒否する。

CPU / Gloo、world size 2 の小さい Linear で、micro-batch ごとの clip と最後の一回だけの clip は一致しなかった。rank 間の勾配は一致した。

## 8. DDP 同期

CPU / Gloo、world size 2 を確認した。

- `find_unused_parameters=True`
- `broadcast_buffers=True`
- Generator と Discriminator の両方を DDP 対象にする
- skip は `agree(policy="any")`。片 rank だけ backward しない
- 新しい勾配だけを `average_new_gradients` で平均する。既に平均した蓄積勾配はもう一度平均しない
- 勾配が None の Parameter はゼロテンソルにしない

`average_gradients` は全勾配を平均し、欠けた勾配をゼロとして参加させる。この意味は変えていない。

world size 1 は現行 Task と数値一致する。world size 2 は増分を平均する安全な研究用実装である。公式 2 GPU の Discriminator が 2 回目の backward で all-reduce しない既知の重み乖離とは、一致すると主張しない。

checkpoint の直前に Buffer を rank 0 から broadcast する。次の DDP forward が行う broadcast と同じ状態をファイルに残すためである。world size 1 では Buffer は変わらない。

## 9. SyncBatchNorm

CUDA かつ world size 2 以上だけ `PreparePolicy.sync_batchnorm` が BatchNorm を SyncBatchNorm に変換する。CPU / Gloo は BatchNorm のままである。

2026-11-09 の短い実行:

- GPU: NVIDIA A100-PCIE-40GB を 2 枚（物理 index 1 と 2）。各 40960 MiB
- PyTorch 2.6.0+cu124、CUDA 12.4、dtype FP32、world size 2
- 波形長 2400、logical step 1、accumulation 1
- `deterministic: false`。`adaptive_max_pool2d_backward_cuda` は決定的実装が無い
- Discriminator の SyncBatchNorm は 8
- `average_calls` は 2（Generator backward と Discriminator backward）
- checkpoint の `parameter_ranks_agree` は True
- 検証の plateau action は `improve`

この実行は CPU とのビット一致を主張しない。`deterministic: true` の CUDA backward は上記の pooling で拒否された。

## 10. Validation

データ epoch の終わりで検証する。`logical_step` の間隔では検証しない。

- Generator は `eval()`
- Discriminator は `train()`。`eval()` にしない
- 公式 Generator Loss を使う
- PESQ 失敗と非有限 Loss は数値の和に足さない。分母の batch 数には残す。欠測を 0 として記録しない
- 整列用 forward は分母に入れない
- 集計は rank ごとの平均を、rank 間で等しい重みで平均する。波形数による加重平均ではない

検証のあとに BatchNorm の Buffer を rank 0 から broadcast してから plateau の snapshot を取る。検証は学習 cursor を進めない。学習 RNG は検証の前後で戻す。

## 11. Plateau

`mossformer_validation_decision` を使う。`ReduceLROnPlateau` には置き換えていない。

- 厳密に改善すると stall は 0 になり、G/D と両 Optimizer の snapshot を残す
- 非改善は stall を増やし、比較用の best もその Loss へ書き換える。歴代最小は保持しない
- 5 回目の非改善で snapshot を復元し、Generator の学習率だけを半分にする。Discriminator の学習率は半分にしない。stall は 5 のまま
- 10 回目の非改善で `stop_training`。追加の半減はしない

seed 7 の実モデルで、現行 `step_schedulers` と Trainer の stall、学習率、復元後の重みが一致した。4 回目の stall から 5 回目の半減、9 回目から 10 回目の停止を、状態を戻してから同じ Loss を適用する再開と比較し、一致した。

## 12. Checkpoint

形式は `audio-db-experiment-v1`。State owner は Generator、Discriminator、Optimizer G、Optimizer D、Trainer、Data、Runtime。Scheduler は無い。

保存できる境界:

- 両 Optimizer の step が終わったあと
- PESQ skip のあと
- 蓄積の途中。`accu_count` と、存在する勾配だけを保存する。None だった Parameter はゼロにしない

G の step と D の step の間は保存しない。テストがこの区間で例外を起こすと、最後に完成した checkpoint から再開できる。

同じ `logical_step` のファイル名は一つである。あとから書いた方が置き換わる。蓄積途中は `step_00000000.pt` のように、まだ増えていない logical step のファイルになる。

## 13. Resume

蓄積 2、成功 1 件の checkpoint から再開した 1 回目の Optimizer 更新は、連続実行の Generator、Discriminator、両 AdamW と一致した（最大絶対誤差 0）。logical step は 1、data cursor は 2。

壊した checkpoint は `CheckpointError`。Skip log と metrics は、復元した件数より後ろを捨て、二重に足さない。

## 14. 数値比較

CPU / FP32、seed 7、公式 STFT。比較相手は現行 `MossFormerGANTask`。

eval の RI、magnitude、time、GAN、generator は一致した。

train、スタブ PESQ、最大絶対誤差 0:

- accumulation 1 の全成功
- accumulation 2 の全成功
- 成功、失敗、成功
- 失敗、成功、成功
- 全失敗
- 非有限の noisy のあと成功

実 PESQ、8,000 サンプルの正弦、accumulation 1、1 更新。Optimizer は 1 回成立し、更新後の Generator と Discriminator の最大絶対誤差は 0。

Loss だけの一致では GO にしていない。勾配、clip、Optimizer state、カウンタも含む。

## 15. GPU 検証

第 9 節の 1 step。長時間学習はしていない。負荷試験用の GPU 1 と 2 を短時間空け、終了後に同じ負荷プロセスを戻した。GPU 6 と 7 は止めていない。

## 16. 回帰

変更前（`6d800dc` の収集開始後に計測。件数は Phase 4-D の記録と同じ）:

| 対象 | 件数 | 結果 |
|---|---:|---|
| `tests/experiments` | 153 | OK、skipped 1 |
| `tests/training` | 297 | FAILED、failures 1、errors 6、skipped 12 |
| `tests/reproduction` | 135 | FAILED、errors 1、skipped 2 |

変更後:

| 対象 | 件数 | 結果 |
|---|---:|---|
| `tests/experiments` | 166 | OK、skipped 1。増分 13 は Phase 4-E |
| `tests/training` | 297 | FAILED、failures 1、errors 6、skipped 12 |
| `tests/reproduction` | 135 | FAILED、errors 1、skipped 2 |

失敗の名前は変更前と同じで、DeepFilterNet3 の `libdf_rfft.so` だけである。新規の Failure、Error、Skip は無い。skipped 1 は変更前からある CUDA AMP の NOT_RUN である。

Phase 4-E の単体は `tests/experiments/test_mossformergan.py` の 13 件が OK だった。

## 17. Phase 4-F への残課題

Phase 4-F は MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer の横断検証である。

残るものは、DeepFilterNet3、SEMamba++、URGENT24 の TF-GridNet、FSDP、sharded checkpoint、AMP BF16、複数 Optimizer の AMP、`audio-db-training-v1` からの再開、公式データ全量と論文スコアである。CUDA の 1 step は長時間学習の検証ではない。公式 2 GPU の重み乖離の再現でもない。
