# P10-B 検証（2026-11-07）

公式幅の Forward / Backward、DDP、接頭辞 validation、checkpoint 再開を、4 モデルで確認した。本格学習には進んでいない。新しい Dataset と生成済み WAV は作っていない。

MP-SENet と TF-GridNet は TSUBAME 4.0 の `r23n5`（H100 × 2、MIG Disabled、メモリ 95,830 MiB、ドライバ 580.105.08）。MossFormerGAN と TF-Locoformer は、その後の SHINODA-LAB `tensor`（A100-PCIE-40GB、ドライバ 535.183.01）。研究室側には対象スナップショットが無かったので、TSUBAME の次の 2 ディレクトリを `data/db/snapshots/` へコピーした。中身は変更していない。

- MossFormerGAN: `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe`
- TF-Locoformer: `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe`

`reproduction.duckdb` は学習が開かないためコピーしていない。`resource.duckdb` は変更していない。

## A. 総合判定

**CONDITIONAL GO。**

4 モデルとも、公式の world size と global batch で Optimizer を 3 回更新し、接頭辞 validation、checkpoint 保存、別プロセスからの再開、中断なしランとの case id 一致まで通った。全件 validation、公式 epoch、`ave_5best` の実平均、CUDA のビット一致は未達である。P10-C の本格学習はこの記録だけでは始めない。

## B. GPU 利用

### TSUBAME

`qrsh -now n -g tga-shinoda -p -5`。`qsub` は使っていない。確保時間は `qacct` の `wallclock`。5 分未満は予算上 5 分。待ち時間は含めない。開始前に消した `9291348` と `9291350`、待機中に消した `9293477`（MossFormerGAN）と `9293479`（TF-Locoformer）は GPU を割り当てられていないので 0 分。

`qacct` の `hard_resources` には `gn_mig=4` と `slots 96` が出る。ノード上の `nvidia-smi` は H100 が 2 枚、MIG Disabled だった。ここでの GPU 数は 2 である。`qacct` にポイント欄は無い。ポイントは未算出。

| Job | Model | 結果 | h_rt | qacct wallclock | スクリプト | 予算 |
|---|---|---|---:|---:|---:|---:|
| 9291345 | MP-SENet | 失敗。`tensorboard` が無かった | 10 分 | 105.390 秒 | 80 秒 | 5 分 |
| 9291347 | MossFormerGAN | 同じ失敗 | 12 分 | 104.631 秒 | 82 秒 | 5 分 |
| 9293476 | MP-SENet | 成功 | 10 分 | 92.510 秒 | 68 秒 | 5 分 |
| 9293478 | TF-GridNet | 成功 | 10 分 | 95.910 秒 | 68 秒 | 5 分 |

予算に載せる確保時間は 20 分。上限 180 分に対する残りは 160 分。成功 2 本の `ru_wallclock` は 67.667 秒と 68.217 秒。

GPU-seconds は wallclock × 2。合計 796.882 GPU-seconds（0.221 GPU-hours）。

成功時の HEAD は `8af54b7`、dirty 7。その学習差分は後で `7b3072f` に入っている。実装ハッシュはログの `IMPL_SHA256 b949c662bbaafc2f5499f71baef3932c367f0a723709eb279452bfa6372c047d`。

### SHINODA-LAB

共有 GPU のうち、使用メモリが 1,024 MiB 未満のものだけを `CUDA_VISIBLE_DEVICES` で指定した。GPU 3 と GPU 4 は他ユーザーが使用中だったので使っていない。ジョブ終了後、こちらのプロセスは残っていない。

| 実行 | Model | GPU | スクリプト経過 | 結果 |
|---|---|---|---:|---|
| tensor、2026-11-07 12:35:03–12:37:26 | MossFormerGAN | A100 40GB × 2（物理 6, 7） | 143 秒 | 成功 |
| tensor、2026-11-07 12:37:45–12:43:59 | TF-Locoformer | A100 40GB × 4（物理 0, 1, 2, 6） | 374 秒 | 成功 |

GPU-seconds は 143 × 2 + 374 × 4 = 1,782（0.495 GPU-hours）。TSUBAME の 180 分予算には足さない。

HEAD は `7b3072f`、dirty 1（共有ホスト向けのスクリプト変更）。`IMPL_SHA256 6e4c30f703761141a89414aec81c7af9dd92350f61821d6d43a65cc7fe32a0dd`。

出力:

- `data/results/training/p10b/20261107-p10b-r2/mp_senet/`（TSUBAME）
- `data/results/training/p10b/20261107-p10b-r2/tf_gridnet/`（TSUBAME）
- `data/results/training/p10b/20261107-p10b-lab/mossformergan/`
- `data/results/training/p10b/20261107-p10b-lab/tf_locoformer/`

## C. モデル別

判定の「接頭辞」は、全件 validation ではない。

| 項目 | MP-SENet | MossFormerGAN | TF-GridNet | TF-Locoformer |
|---|---|---|---|---|
| 公式幅の初期化 | PASS | PASS | PASS | PASS |
| Forward / Backward | PASS | PASS | PASS | PASS |
| Optimizer 更新 2 回以上 | PASS。3 回 | PASS。3 回 | PASS。3 回 | PASS。3 回 |
| 公式 world size の DDP | PASS。2 | PASS。2 | PASS。2 | PASS。4 |
| Validation 経路 | PASS。先頭 2 batch | PASS。先頭 2 batch | PASS。rank あたり 1 batch | PASS。rank あたり 1 chunk |
| Checkpoint 保存 | PASS | PASS | PASS | PASS |
| Checkpoint 再開 | PASS。step 3 へ進んだ | PASS。step 3 へ進んだ | PASS。step 3 へ進んだ | PASS。step 3 へ進んだ |
| Dataset identity | PASS | PASS | PASS。`data_state.dataset` | PASS。`data_state.dataset` |
| TensorBoard / JSONL | PASS | PASS | JSONL は PASS。TensorBoard は TSUBAME 成功ランのディレクトリが存在する | PASS。scalar を読んだ |
| Data provenance | PASS | PASS | PASS | PASS |

Virtual rank は使っていない。`data_provenance.json` の `ddp_mode` は MossFormerGAN と TF-Locoformer で `real`。

### G1–G2

有限損失で Optimizer が更新された。peak allocated は rank 0 のログである。

| Model | batch plan | peak allocated | peak reserved |
|---|---|---:|---:|
| MP-SENet | global 4、micro 2、world 2、蓄積 1 | 11,389,884,416 | 13,753,122,816 |
| TF-GridNet | global 4、micro 2、world 2、蓄積 1 | 2,743,491,584 | 3,026,190,336 |
| MossFormerGAN | global 4、micro 1、world 2、蓄積 2 | 12,770,233,856 | 12,952,010,752 |
| TF-Locoformer | global 4、micro 1、world 4、蓄積 1 | 16,509,144,064 | 17,764,974,592 |

A100 40GB では MossFormerGAN と TF-Locoformer のどちらも OOM していない。パラメータ数は保存済み state dict の要素数で、MossFormerGAN の Generator は 3,128,639、Discriminator は 8,125,019、TF-Locoformer は 14,973,506。MP-SENet は既監査の G 2.263M、D 0.182M。TF-GridNet の公開 checkpoint 監査は 2,552,790。今回の初期化後の本数は数え直していない。

TF-Locoformer の JSONL 学習率は更新後で、step 1 / 2 / 3 が `2.5e-7`、`5e-7`、`7.5e-7` だった。`0.001 / 4000 * step` と一致する。最初の更新自体は構築時の 0.001 を使う。

### G3

各ランの `data_state.world_size` は 2、2、2、4。TF-GridNet と TF-Locoformer の計画は `espnet_chunk_v1`。TF-Locoformer の `per_rank_batch` は 1、`num_samples_per_epoch` は 1,000、`chunk_length` は 64,000。MossFormerGAN の `order_policy` は `rank_mod_fixed_v1`。デッドロックは起きていない。

### G4

MP-SENet の検証は公式順の先頭 2 batch。PESQ は有限（work のログは 1.324699）。best は `best_generator.pt`。これはスモークがゲートを epoch 0 にした結果である。

MossFormerGAN の work 検証は generator 0.4566850960254669、RI 0.8478517532348633、magnitude 0.2808321937918663、time 0.5957547426223755、GAN 4.0986012805888095e-08。加重和との差は 1e-8 未満で、time 単独ではない。PESQ 指標自体は `na`（plateau の指標ではない）。2 batch とも分母に入っているので、PESQ ゲートは両方を採用している。`best_checkpoint` は null。epoch 0 は公開ゲート 40 より前だからである。step 3 の checkpoint の `plateau.best_val_loss` は同じ 0.4566850960254669、stall は 0、最良 snapshot はあり、`sched_g` は null。半減はしていない。

TF-GridNet の検証損失は 7,931.48388671875。scope は rank あたり 1 batch。best は `1epoch.pt`。学習率は 0.001 のままである。最初の検証が最良値を作るだけで、patience による半減は 2 epoch 目の末に達していない。

TF-Locoformer の検証は 4 rank の重み合計 4、generator 6,875.3177490234375、SI-SDR −4.634327992796898 dB。`1epoch.pt`、`valid.loss.best.pt`、`valid.si_snr.best.pt` がある。`valid.loss.ave_1best.pt` は `1epoch.pt` へのリンクで、`valid.loss.ave.pt` はそこを向く。`valid.loss.ave_5best.pt` は無い。

### G5

再開は step 2 の checkpoint から step 3 へ進んだ。中断なしランと case id は step 1、2、3 で一致した。plan hash、dataset id、snapshot hash、world size も一致した。損失は step 1 でほぼ一致し、その後は非決定的 CUDA で開く。

| Model | step 1 の generator 差 | step 2 | step 3 |
|---|---:|---:|---:|
| MP-SENet | 2.4e-7 | 2.4e-6 | 4.4e-3 |
| TF-GridNet | 4.9e-4 | 9.8e-4 | 2.1e-2 |
| MossFormerGAN | 0 | 1.5e-5 | 5.8e-7 |
| TF-Locoformer | 0 | 6.8e-3 | 6.0e-2 |

chunk 計画の dataset id は `data_state` の直下ではなく `data_state.dataset` にある。比較 JSON の直下フィールドが null でも、来歴 JSON と `data_state.json` には入っている。

## D. 修正

原因とテストは [P10_B_AUDIT](P10_B_AUDIT.md)。GPU 前に、MossFormerGAN の検証損失、スモーク上限、TF-GridNet の xavier 初期化、`complete.json` の退避を入れた。研究室側では、スクリプトが TSUBAME の絶対パスへ `cd` していたため、リポジトリ相対に変えた。共有 GPU を掴む前に `CUDA_VISIBLE_DEVICES` の個数を確認する。

`tests.training.test_mossformer_validation` は 3 件成功（20.6 秒、CPU）。検証損失の加重和、PESQ 失敗時の 0、上限キーの任意指定を見ている。

## E. 公式との対応

監査の未解決表と同じ。加えて、TensorBoard の音声プレビューはスモーク YAML で無効である。正本は `metrics.jsonl`。

## F. 回帰

`CUDA_VISIBLE_DEVICES=""` で `tests/training` を CPU 実行した。230 件、失敗 11、SKIP 12、743 秒。

TensorBoard の DDP 8 件と再開境界 1 件は、スイート一括では `tests.training` の import 失敗か event の順序不一致になった。`test_two_ranks_write_once` と `test_interruption_at_checkpoint_and_jsonl_boundaries` を単独で再実行すると PASS だった。子プロセスへ埋め込んだ `sys.path` がスイート中に伸びた結果で、今回の学習差分が原因ではない。

`test_checked_in_config_uses_the_official_optimizer` は `configs/training/mp_senet.yaml` の `deterministic: false` を拒否する。この値は `b95bbd6` で、今回の差分ではない。`configs/training/trial100` は無く、5 モデルの試験は 0 件になる。このディレクトリは今回削除していない。

`libdf` が `.venvs/se` に無いことによる SKIP は、今回の差分とは別である。SKIP は PASS に数えない。DeepFilterNet3 と SEMamba++ の公式 Dataset は blocked のままである。

## G. 残課題

- 修正可能: 比較 JSON が chunk の dataset id を直下だけ見ていた。スクリプトは `data_state.dataset` も読むようにした。既存の比較 JSON は作り直していない。
- 公式情報が必要: `cv_synthetic_small`、TF-GridNet の shape file 順、MP-SENet の公式 16 kHz 変換器、DNS の公式 librosa 設定。
- 時間不足ではない: 全件 validation と `ave_5best` は、このスモークの設計として切っている。
- P10-C: 上限キーを外した公式 epoch、全件に近い validation、5 epoch 以上での `ave_5best`。投入は別承認。

## H. P10-C

| Model | 判定 | 理由 |
|---|---|---|
| MP-SENet | 条件付きで開始できる | 2 GPU の更新、検証接頭辞、再開は通った。400 epoch と epoch 40 の best ゲートは未実施 |
| MossFormerGAN | 条件付きで開始できる | 2 GPU、生成器損失、plateau 状態の保存は通った。半減と 824 件検証は未実施 |
| TF-GridNet | 条件付きで開始できる | 2 rank の chunk 計画と `valid.loss` の 1 epoch は通った。1,835 件と plateau の半減は未実施 |
| TF-Locoformer | 条件付きで開始できる | 4 rank、64,000 chunk、受理 Dataset、`ave_1best` は通った。36,000 件と `ave_5best` は未実施 |

4 モデルとも、P10-C をこのセッションでは開始していない。
