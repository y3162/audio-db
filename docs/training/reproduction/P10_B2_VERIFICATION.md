# P10-B.2 検証

ホストは `tensor`。GPU は A100-PCIE-40GB。Python は `.venvs/se`（3.12.13、torch 2.6.0+cu124）。HEAD は `7979d931a9696f7a2b323309c2f68313a7248e49`。作業開始時の dirty は 0。GPU の学習計測は `AUDIO_DB_STEP_SYNC=1`。Run ID は `20261107-p10b2`。`20261107-p10b-lab` と `20261107-p10b1` は上書きしていない。

本格再現学習は開始していない。新しい Dataset、生成済み WAV、波形キャッシュは作っていない。`resource.duckdb` は変更していない。

## スナップショット

`load_snapshot` は manifest の論理ハッシュと `snapshot_hash` を検査する。その後、参照する元素材を読み、記録された SHA-256 と比べた。

### MP-SENet

`mp_senet.vb28.ondemand.official@1`

| 項目 | 結果 |
|---|---|
| dataset id | `36ab4590b113d0aa00884c44041eaa97d65085cee57bcc9c9e06135741fc76d7` |
| snapshot hash | `265f256b938a89746dcaa62cd44daa2c6bcca3bce4cc1b1719eb2170717d33d5` |
| protocol | `69dcd7c2bd50f353777cacb76a86ccf3c44d7cc133a3bc355995727b9f0269a7` |
| 件数 | train 11,572、valid 824、test 824。valid と test は同じ集合。train との重なり 0 |
| `selection_uses_test` | true |
| 元素材 | 一意 24,792。ハッシュしたバイト数 6,890,139,300。欠落 0、不一致 0。22.1 秒 |
| epoch 0 | world 2、global 4、micro 2、蓄積 1。更新 2,893。policy `torch_distributed_sampler_v1`。plan hash `6eb387c88cacb9e35c4a037dd07aad7eca611ec362d63c7bcc06aa45a8321578` |
| 再構築 | rank 0 の先頭 batch を 2 回描き、noisy と clean の最大絶対誤差は 0。検証の先頭は `(1, 27861)` の全文。有限 |

### TF-GridNet

`tf_gridnet.ms_snsd.espnet_recipe@1`

| 項目 | 結果 |
|---|---|
| dataset id | `13c6058881ef86fe7506f705b1bd5a5bc68b313181410d8fdaaee20c33ef3fc6` |
| snapshot hash | `a69a477af70bc43dbcabdafbd5c97ec8a26aac4debd51158079545de1e4f0757` |
| protocol | `9f5ffa7c14a43b23b5db0a0c931c8a80e91cb666518aa10c826813c90abf7e97` |
| 件数 | train 7,345、valid 1,835、test 305。分割の重なり 0 |
| `selection_uses_test` | false |
| 元素材 | 一意 24,354。バイト数 3,067,712,184。欠落 0、不一致 0。107.7 秒 |
| 長さ | 音声は未読。train 最小 160,138、最大 333,557。valid 最小 160,016、最大 313,910。48,000 以下は 0 |
| 再構築 | train と valid の先頭 chunk を 2 回描き、最大絶対誤差は 0。形状 `(2, 48000)`。有限 |

MossFormerGAN の同じ確認は、dataset id `153096c312f8027bba93fa635224bff8681170841bbe40164d189f923d16aebe`、snapshot `ba0c77c94babbbc2c3e70b149554f8a922a68cf6fbe71ccf833927dc71dff555`、train 11,572、valid 824、test 824、valid と test は同一、元素材 24,792 の不一致 0。

TF-Locoformer の受理 Dataset は dataset id `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe`、snapshot `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`。train 324,000 と valid 36,000 の長さはすべて 480,000。`selection_uses_test` は false。

## MP-SENet の中断と再開

物理 GPU 1 と 2。公式 YAML に `verification_max_updates` だけを足した。`max_updates_per_epoch` は無い。6 更新で `INTERRUPTED`、`complete.json` は無い。別プロセスが step 16 まで再開した。中断なしランも 16 更新で `INTERRUPTED`。所要 131 秒。

| 項目 | 結果 |
|---|---|
| case id | step 1 から 16 まで work と continuous が一致 |
| plan hash | `6eb387c88cacb9e35c4a037dd07aad7eca611ec362d63c7bcc06aa45a8321578`。CPU の epoch 0 と同じ |
| dataset / snapshot | 上の表と同じ |
| 位置 | data epoch 0、`update_in_epoch` 16。world size 2 |
| 学習率 | Generator 0.0005、Discriminator 0.0005。16 step とも不変。Scheduler `last_epoch` は 0 |
| 損失 | step 1 の generator は 2.25682688 と 2.25682759。step 16 は 1.17610 と 1.18005 |
| 重み | step 16 の最大絶対差は Generator 9.11e-3、Discriminator 2.61e-2。`deterministic: false` |
| Checkpoint | step 6 は 29,689,087 バイト、0.49 秒。step 16 は約 29.7 MB、0.83 秒と 0.95 秒 |
| peak allocated | 11,390,721,536 バイト |

定常 15 step（step 1 を除く）の同期更新時間は、中央値 0.454 秒、平均 0.452 秒、最小 0.322 秒、最大 0.657 秒。データ待ちの中央値は 0.069 秒、平均 0.083 秒。更新と待ちを足した 1 step の平均は 0.534 秒、中央値 0.514 秒。

## TF-GridNet の中断と再開

物理 GPU 1 と 2。`chunk_iterator.max_updates` は外した。6 更新で epoch は終わらないので、全件検証は走らない。`INTERRUPTED`。`complete.json` は無い。再開と中断なしランは 16 更新。所要 152 秒。

| 項目 | 結果 |
|---|---|
| case id | step 1 から 16 まで一致 |
| plan hash | `550fdeabd54e01ad62e5f75bcdeda7ea487d6ff26f6e2d7da8072a68adc43144`。CPU の epoch 1 と同じ |
| dataset / snapshot | 上の表と同じ |
| 位置 | data epoch 0、`update_in_epoch` 16。world size 2 |
| 学習率 | 0.001。16 step とも不変。epoch 末ではないので plateau は進んでいない |
| 損失 | step 1 は 6,968.6904296875 で一致。step 16 は 6,075.592 と 6,075.658 |
| 重み | step 16 の最大絶対差は 1.83e-3（`blocks.1.attn_conv_V_1.2.gamma`） |
| Checkpoint | 31,071,895 バイト、書き込み 0.63 秒。再開側の step 16 は 31,075,927 バイト、0.62 秒 |
| peak allocated | 2,743,491,584 バイト |

定常 15 step の同期更新時間は、中央値 0.252 秒、平均 0.248 秒、最小 0.145 秒、最大 0.399 秒。データ待ちの中央値は 0.117 秒、平均 0.143 秒。更新と待ちの平均は 0.391 秒、中央値 0.364 秒。この設定の `prefetch` は false なので、待ちは更新と直列である。

## MossFormerGAN の全件検証

物理 GPU 6。world size 2 の公式順を 1 プロセスの rank 0 が見た。学習更新はしていない。824 件、237.94 秒。

| 項目 | 結果 |
|---|---|
| 件数 | 824。形状はすべて `(1, 32000)` |
| Generator | 1.1346609779787296。count 824 |
| RI | 1.099149186585103。count 823 |
| magnitude | 0.5043251261314239。count 823 |
| time | 2.7705812845311217。count 823 |
| GAN | 0.3623175526519882。count 823 |
| PESQ ゲート | included 823、0 として残した件数 1。PESQ 値そのものは `na` |
| パラメータ | Generator も Discriminator も最大差 0 |
| Discriminator buffer | `cnn.bn6.running_mean` の最大差 1.136。`num_batches_tracked` も増えた。公式の検証が Discriminator を `train` のままにするため |
| Optimizer | 状態は空。学習率は検証の前後とも G 0.0005、D 0.001 |
| plateau | 改善。stall 0。最良 snapshot あり。生きている重みの差は 0。半減していない |
| peak allocated | 609,113,600 バイト。reserved 692,060,160。プロセス RSS 1,695,160 KB |

## MP-SENet の全件検証

物理 GPU 6。rank 0、全文、batch 1。学習更新はしていない。824 件、778.323 秒。先頭 64 件は 100.724 秒（1 件 1.574 秒）だった。この接頭辞には 156,302 サンプルまでの長い発話があり、全件平均の 0.945 秒より長い。全件の実測を使う。

PESQ は 824 件とも有限。平均 1.1981693374589808。Generator 損失は 1.9328542169436669。time 0.52649、magnitude 0.13614、metric 0.23480。パラメータと buffer の最大差は 0。Discriminator も `eval`。Optimizer は進んでいない。学習率は 0.0005。peak allocated は 8,219,724,800 バイト、reserved は 21,611,151,360 バイト。RSS は 1,622,356 KB。

これは step 5,000 の検証経路と同じ損失と PESQ である。16 更新の中断ランでは epoch が終わっていないので、ExponentialLR は進んでいない。best のゲート epoch 40 より前であり、この検証だけでは `best_generator.pt` を公式の選択点とは呼ばない。

## TF-GridNet の全件検証

物理 GPU 1 と 2。検証計画の 2,887 batch を各 rank が処理した。手動の STFT、本体、損失は、batch 0 で `validation_report` と差 0。

| 段階 | rank 0 | rank 1 |
|---|---:|---:|
| 初期化 | 7.06 秒 | 5.21 秒 |
| 計画 | 0.60 秒 | 0.44 秒 |
| 再構築 | 312.85 秒 | 300.80 秒 |
| STFT | 1.95 秒 | 2.41 秒 |
| 本体 | 260.32 秒 | 255.28 秒 |
| 損失 | 11.09 秒 | 10.19 秒 |
| forward 一式 | 276.12 秒 | 270.73 秒 |
| all-reduce | 0.93 秒 | 20.55 秒 |

rank 1 の all-reduce が長いのは、先にループを終えて rank 0 を待ったためである。同期した 1 回は、遅い rank のループ 588.97 秒に all-reduce を足して約 592 秒。集約損失は 7,132.230456657132。重み合計は 11,548。plan hash は CPU の検証計画と同じ。peak allocated は 525,130,240 バイト。

512 batch の接頭辞も同じ plan hash と、batch 0 の差 0 だった。全件を採用する。

## TF-Locoformer の検証時間

物理 GPU 0、1、2、6。world size 4。接頭辞は rank あたり 2,048 chunk。全件 126,000 ではない。batch 0 の手動損失と `validation_report` の差は 0。plan hash は CPU の検証計画と同じ。peak allocated は 607,085,568 バイト。

最も遅い rank の、再構築と forward を足した 1 chunk あたりの時間:

| 接頭辞 | 秒/chunk |
|---:|---:|
| 128 | 0.160 から 0.169 |
| 512 | 0.158 |
| 2,048 | 0.1602 |

2,048 chunk の内訳（rank 0）は、再構築 102.92 秒、STFT 0.487 秒、本体 220.83 秒、損失 3.295 秒、all-reduce 1.59 秒。計画は 8.7 から 9.6 秒。初期化は 1.6 秒以下。1 chunk あたりの時間は接頭辞の長さで変わらない。全件は `126000 * 0.16018359375 + 計画と all-reduce` で 20,194 秒、5.61 時間。

同じ 4 GPU で 12 更新を同期計測した。step 1 は 1.535 秒。定常 11 step の更新時間は中央値 0.374 秒、平均 0.387 秒、最小 0.368 秒、最大 0.431 秒。データ待ちの平均は 0.272 秒。更新と待ちの中央値は 0.667 秒、平均は 0.659 秒。学習率は `0.001 / 4000 * step` で、step 12 は 3e-6。終了は `INTERRUPTED`。`complete.json` は無い。Checkpoint は 179,910,164 バイト、書き込み 2.29 秒。peak allocated は 16,470,292,992 バイト。

## 速度とメモリ

| モデル | 定常更新の中央値 | 待ちの中央値 | 1 step の平均 | peak allocated | Checkpoint |
|---|---:|---:|---:|---:|---:|
| MP-SENet | 0.454 秒 | 0.069 秒 | 0.534 秒 | 11.39 GB | 29.7 MB、約 0.5 から 0.9 秒 |
| MossFormerGAN | 1.038 秒 | 0.008 秒 | 1.089 秒 | 12.58 GB | 136,454,907 バイト、1.86 秒 |
| TF-GridNet | 0.252 秒 | 0.117 秒 | 0.391 秒 | 2.74 GB | 31.1 MB、約 0.63 秒 |
| TF-Locoformer | 0.374 秒 | 待ち平均 0.272 秒 | 0.659 秒 | 16.47 GB | 179.9 MB、2.29 秒 |

MossFormerGAN の 12 更新も `INTERRUPTED` で、`complete.json` は無い。学習率は G 0.0005、D 0.001 のままである。検証 1 回は上の 237.94 秒。

## 回帰

`tests/training` と `tests/reproduction` を `CUDA_VISIBLE_DEVICES=""` で再実行している。結果が揃い次第、この節を実数に置き換える。SKIP は PASS に数えない。
