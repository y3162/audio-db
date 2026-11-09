# Phase 5: 本格学習前の readiness

判定: **CONDITIONAL GO。** これは設計書の Phase 5（LoRA、難例 Sampler、複数 Expert、Distillation、勾配を通す ASR）ではない。損失、Optimizer の更新順、DataModule の式、checkpoint schema は変えていない。公式データ全量も論文スコアも実行していない。

2026-11-10。ホストは tensor。GPU は NVIDIA A100-PCIE-40GB（40,960 MiB）を物理番号 1、2、6、7 に限って使った。0、3、4、5 は使っていない。Python 3.12.13、PyTorch 2.6.0+cu124、CUDA 12.4、glibc 2.31、搭載メモリ 503 GB。DeepFilterNet3 の native library は Phase 4-J の規則で `build/df-rfft/release/libdf_rfft.so`（ABI 1、SHA-256 `e1eb0815e5242191e98ca5785ca2a868a01b421d91be7b0fc6e79b994dc77e77`）を選んだ。追跡中の `libdf_rfft.so` は GLIBC 2.33 が無く、置き換えていない。SEMamba++ の fused は `.venvs/semambapp` で、`scan_configured=fused` かつ `scan_executed=fused` を metrics で確認した。

## 1. 変更ファイル

学習式は変えていない。`AUDIO_DB_RESOURCE_PROFILE=1` のときだけ `resource_profile.json` を書く。既定の回帰はこのファイルを書かず、CUDA 同期もしない。

- `src/experiments/profile.py`
- `src/experiments/runtime.py`
- `src/experiments/checkpoint.py`
- `src/experiments/api.py`
- `scripts/experiments/phase5_launch.py`
- `tests/experiments/test_phase5_profile.py`
- `configs/experiments/readiness/`
- `configs/experiments/production/`

Phase 4 の `configs/experiments/*.yaml` は変えていない。本格候補は起動していない。短い結果は `data/results/experiments/phase5-readiness/` で、既存の `data/results/training/` とは別である。

## 2. 測り方

`scripts/experiments/phase5_launch.py` が既存の `python -m src.experiments.cli` を起動する。world size は見える GPU 数で、1 枚なら単一プロセス、2 枚以上は `torch.distributed.run` である。nvidia-smi の使用メモリは 2 秒間隔のサンプルで、短い peak を取りこぼし得る。表の allocated / reserved は PyTorch の peak で、trainer 開始時に reset した値である。backward と optimizer の秒数は profiling 中だけ CUDA synchronize した CPU 区間である。rank 比較と checkpoint 書き込みは同期していない。初回起動の wall は import と構築を含む。`resource_profile.json` の elapsed は構築のあとである。MP-SENet と MossFormerGAN の profile は、同じディレクトリの 2 step 目（resume）で上書きされている。1 step 目の wall は launch report に残っている。

プロセス RSS の最大値は測っていない。

## 3. 公式幅

宣言は Phase 4 の公式 YAML と同じである。幅は変えていない。

| モデル | 幅 | 今回の学習条件 |
|---|---|---|
| MP-SENet | dense channel 64、TSC Conformer 4 | world 2、micro-batch 2、2 秒、FP32、D のあと G。global batch 4 |
| MossFormerGAN | `se_16k`、`num_channel=64` | world 2、micro-batch 1、accumulation 2、2 秒、FP32、SyncBatchNorm。global batch 4 |
| TF-GridNet | MS-SNSD architecture | world 2、chunk 48,000、micro-batch 2、global batch 4、Adam、MultiResolutionL1 |
| TF-Locoformer | DNS medium architecture | world 4、chunk 64,000、micro-batch 1、global batch 4、AdamW、WarmupReduceLROnPlateau |
| DeepFilterNet3 | 出荷 DfNet | world 1、3 秒、native RFFT、batch schedule 付き。実行 batch は下に書く |
| SEMamba++ | `hid_feature=48`、TF-Mamba 4 | 3 秒、fused、micro-batch 1、2、4。batch 8 は学習していない |

Preflight は optimizer 更新をしない。成功しても学習成功ではない。

## 4. GPU 学習

| 実行 | 結果 | 1 プロセス wall 秒 | nvidia peak MiB |
|---|---|---:|---|
| MP-SENet preflight | PASS、`training_executed=false` | 12.7 | 1,148 / 728 |
| MP-SENet logical step 1 | INTERRUPTED（`stop_after` 1）。D と G が applied | 17.2 | 14,340 / 13,920 |
| MP-SENet resume で step 2 | COMPLETED。`step_00000002.pt` | 19.3 | 14,360 / 13,940 |
| MossFormerGAN preflight | PASS | 12.9 | 1,180 / 760 |
| MossFormerGAN logical step 1 | INTERRUPTED。skip なし。G と D の loss は有限 | 21.2 | 13,856 / 13,174 |
| MossFormerGAN resume で step 2 | COMPLETED | 27.4 | 13,684 / 13,526 |
| TF-GridNet preflight | PASS | 13.2 | 1,174 / 728 |
| TF-GridNet logical step 1 | COMPLETED。loss 6,968.69。1 epoch の途中で止めている | 19.1 | 4,090 / 3,670 |
| TF-Locoformer preflight | PASS。world 4 | 81.7 | 2,161 / 906 / 906 / 886 |
| TF-Locoformer logical step 1 | COMPLETED。loss 7,947.01。validation には到達していない | 93.5 | 20,205 / 18,950 / 18,950 / 18,930 |
| DeepFilterNet3 preflight | PASS。出荷 cap 64 に対する batch 16 の警告あり | 6.6 | 542 |
| DeepFilterNet3 logical step 1 | COMPLETED。loss 6.97。sample は 8 件 | 12.7 | 2,354 |
| DeepFilterNet3 完了ディレクトリの resume | 拒否。`complete.json` がある | — | — |
| DeepFilterNet3 SIGTERM | 下の Stop を参照 | — | — |
| SEMamba++ fused micro 1 | COMPLETED。`scan_executed=fused`。loss 12.50 | 19.1 | 7,832 |
| SEMamba++ fused micro 2 | COMPLETED。loss 12.24 | 19.0 | 14,536 |
| SEMamba++ fused micro 4 | COMPLETED。loss 12.44 | 21.2 | 28,316 |
| SEMamba++ batch 8 | 学習していない。preflight だけ PASS | 10.8 | 470 |
| leakage-safe MP-SENet | 学習していない。preflight だけ PASS | 14.9 | 1,148 / 740 |

TF-GridNet と TF-Locoformer の損失が大きいのは、この 1 step の値であり、論文のスコアではない。CUDA の bit 一致は主張しない。`deterministic: false` である。

DeepFilterNet3 の YAML は micro-batch 16 を要求した。metrics の sample id は 8 件で、smoke の epoch が 16 に満たない短い batch だった。公式 batch 16 の学習成功とは記録しない。`batch_schedule` 付き world size 2 は起動していない。拒否は外していない。

SEMamba++ の micro 4 は、最初の起動が bandwidth の `kaiser_best` で `resampy` 不足により描画前に落ちた。`.venvs/semambapp` に既 pin の `resampy==0.4.3` を入れてから同じ micro 4 を 1 回再実行し、その回が完了した。micro 1 と 2 はその劣化を引いていなかった。batch 8 は Phase 4-H の約 45.56 GiB OOM の記録を残し、今回は再実行していない。micro 4 の reserved 27.1 GiB から batch 8 を成功扱いにしない。

## 5. Dataset

素材パスは各 snapshot から 30 件を標本し、欠けは 0 だった。全ファイルの存在確認ではない。

| Snapshot | 状態 | train | 備考 |
|---|---|---:|---|
| `mp_senet.vb28.ondemand.official@1` | ready | 11,572 | `selection_uses_test=true`。test 824。独立な test 評価ではない |
| `mp_senet.vb28.ondemand.safe@1` | ready | 10,700 | `selection_uses_test=false`。valid 872。今回は preflight のみ |
| `mossformergan.vb28.ondemand.official@1` | ready | 11,572 | `selection_uses_test=true` |
| `mossformergan.vb28.ondemand.safe@1` | ready | 10,700 | Trainer は `official_faithful` 以外を拒否する。研究用 YAML は作っていない |
| `tf_gridnet.ms_snsd.espnet_recipe@1` | ready | 7,345 cases | `selection_uses_test=false`。論文実験の再現とは呼ばない |
| `tf_locoformer.dns.accept_recipe.research_valid_fixed@1` | ready | 324,000 | valid は 1,024。`cv_synthetic_small` ではない |
| `dfn3.online.smoke@1` | ready | cases ではなく material | 代替素材。`dfn3.online.official@1` は無い |
| `semambapp.gsr.smoke@1` | ready | material | 代替素材。`semambapp.gsr.official@1` は無い |

Provenance は rank 0 の `data_provenance.json` と checkpoint sidecar に書いた。MP-SENet と MossFormerGAN の official は limitations に test を選択に使う旨がある。DeepFilterNet3 と SEMamba++ は `train.cases is not bound` がある。これを independent の証明にはしない。

## 6. Peak memory

PyTorch peak。resume で上書きされた MP-SENet と MossFormerGAN は 2 step 目の値である。

| モデル | allocated GiB | reserved GiB | OOM |
|---|---:|---:|---|
| MP-SENet world 2 | 10.61 | 12.83 | なし |
| MossFormerGAN world 2 | 11.76 | 12.16 | なし |
| TF-GridNet world 2 | 2.54 | 2.80 | なし |
| TF-Locoformer world 4 | 15.20 | 17.72 | なし。4 rank とも allocated 15.20 |
| DeepFilterNet3 world 1、実行 batch 8 | 1.42 | 1.72 | なし。公式 batch 16 は未達 |
| SEMamba++ micro 1 | 6.47 | 7.07 | なし |
| SEMamba++ micro 2 | 12.74 | 13.62 | なし |
| SEMamba++ micro 4 | 25.21 | 27.07 | なし |
| SEMamba++ batch 8 | — | — | 今回は未実行。過去の OOM 記録は残す |

## 7. 更新時間

profile の秒。MP-SENet と MossFormerGAN は resume の step である。

| モデル | backward | optimizer | 比較 | checkpoint 書き込み | profile elapsed |
|---|---|---|---:|---:|---:|
| MP-SENet | 0.066、0.561 | 0.051、0.004 | 0.355 | 1.290 | 3.35 |
| MossFormerGAN | 0.453、0.038、0.248、0.042 | 0.069、0.002 | 1.684 | 3.553 | 8.88 |
| TF-GridNet | 0.561 | 0.063 | 0.652 | 2.659 | 6.66 |
| TF-Locoformer | 0.701 | 0.056 | 1.973 | 6.753 | 14.19 |
| DeepFilterNet3 | 0.560 | 0.091 | 0.000（world 1） | 1.132 と 0.450 | 6.52 |
| SEMamba++ micro 4 | 0.117、2.011 | 0.053、0.035 | 0.000（world 1） | 1.721 | 11.93 |

MP-SENet の長い backward は Generator 側である。MossFormerGAN は micro 2 回 × G/D の 4 backward と、その後の G、D の optimizer である。SEMamba++ の長い backward は Discriminator 側である。比較は検出を弱めていない。world 1 の比較がほぼ 0 なのは、通信が無いためである。

音声秒 / profile 秒は、checkpoint を含んだ 1 step の比率である。MP-SENet は 8 / 3.35、MossFormerGAN は 8 / 8.88、TF-GridNet は 12 / 6.66、TF-Locoformer は 16 / 14.19、DeepFilterNet3 は 24 / 6.52、SEMamba++ micro 4 は 12 / 11.93。定常速度の保証ではない。

## 8. Checkpoint

| モデル | ファイル MiB | sidecar bytes | 比較 秒 | 書き込み 秒 |
|---|---:|---:|---:|---:|
| MP-SENet | 28.33 | 8,760,857 | 0.355 | 1.290 |
| MossFormerGAN | 130.17 | 8,760,879 | 1.684 | 3.553 |
| TF-GridNet | 29.65 | 7,573,815 | 0.652 | 2.659 |
| TF-Locoformer | 171.62 | 43,273,704 | 1.973 | 6.753 |
| DeepFilterNet3 | 32.85 | 16,589 | 0 | 1.132 |
| SEMamba++ | 86.04 | 10,669 | 0 | 1.721 |

CPU メモリの peak は NOT_MEASURED。

## 9. 推定

checkpoint 書き込みを引いた profile 秒を 1 更新とみなす。validation、PESQ の skip、warmup、epoch 境界、cache は入っていない。少数 step の外挿であり、長時間の安定性は保証しない。

| モデル | 更新の推定秒 | 予算 | 推定 |
|---|---:|---|---|
| MP-SENet 2 GPU | 2.1 | 2,893 更新 × 400 epoch = 1,157,200 | 1 epoch 約 1.7 時間。400 epoch 約 670 時間。5,000 更新ごとの validation は別 |
| MossFormerGAN 2 GPU | 5.3 | 2,893 更新 × 121 pass = 350,053 | 1 pass 約 4.3 時間。全 pass 約 520 時間。今回の 2 step に skip は無かった |
| TF-GridNet 2 GPU | 4.0 | 計画 553,521 更新、35 epoch | 合計約 620 時間。epoch ごとの validation は別。1 epoch の更新数は割り切れないのでここには書かない |
| TF-Locoformer 4 GPU | 7.4 | 14,000 更新 × 150 epoch = 2,100,000 | 1 epoch 約 29 時間。150 epoch 約 4,300 時間 |
| DeepFilterNet3 公式 22,711 更新 | — | VoiceBank 10,700 発話 | smoke の短い batch から外挿しない |
| SEMamba++ 公式 batch 8 | — | — | micro 4 の 10 秒を batch 8 の時間にしない |

## 10. 本格候補 YAML

`configs/experiments/production/`。schema は `audio-db-experiment/v1`。起動していない。

| ファイル | 意味 |
|---|---|
| `mp_senet_gan_official.yaml` | world 2、micro 2、1,157,200 step。test を選択に使う |
| `mp_senet_gan_leakage_safe.yaml` | safe snapshot、1,070,000 step、validation 2,675。preflight のみ |
| `mossformergan_se_16k_official_2gpu.yaml` | world 2、350,053 step。scratch |
| `tf_gridnet_ms_snsd.yaml` | world 2、553,521 step、35 epoch。mid-epoch 保存は間隔 1,000,000 |
| `tf_locoformer_dns_research.yaml` | world 4、2,100,000 step。research valid。`cv_synthetic_small` ではない |
| `deepfilternet3_smoke_schedule.yaml` | world 1、schedule 付き、smoke。120 epoch が停止条件。22,711 更新ではない |
| `semambapp_gsr_micro4.yaml` | 測れた最大。fused、micro 4、accumulation 1。batch 8 ではない |
| `semambapp_gsr_micro1.yaml` | より小さい測定済み形状 |
| `semambapp_gsr_batch8.yaml` | 公式の one-forward 形状。40GB では起動しない前提。preflight のみ |

出力先は `data/results/experiments/phase5-production/` で、既存 results と重ならない。

## 11. Stop と resume

DeepFilterNet3 で `max_logical_steps: 8` の実行に、開始 15 秒後に SIGTERM を送った。status は `INTERRUPTED`、`complete.json` は無く、metrics は step 1 と 2 で止まっている。開始 6 秒の試行は import 中で handler が無く、プロセスが死んだ。こちらは境界停止の成功に数えない。

MP-SENet と MossFormerGAN は step 1 の checkpoint から step 2 へ再開し、`COMPLETED` になった。完了済み DeepFilterNet3 ディレクトリの resume は `complete.json` を理由に拒否した。GAN の checkpoint は logical step の完了後であり、G と D の片側だけを保存したファイルは作っていない。MossFormerGAN の今回の step に PESQ skip は無く、skip 後 cursor の新しい測定はしていない。Phase 4-J の試験は残っている。

## 12. Provenance

新規 checkpoint は `audio-db-data-provenance-v1` を持つ。公式 MP-SENet と MossFormerGAN の `selection_uses_test` は true である。leakage-safe の preflight は学習も provenance ファイルも書いていない。

## 13. Preflight

上の表の preflight は PASS で、`training_executed` は false である。SEMamba++ の micro 2 と 4 は、micro 1 と同じモジュールを batch だけ変えて学習した。batch 8 の preflight はモジュール構築であり、batch 8 の forward ではない。必要 GPU が足りない条件を world size 1 に書き換えて成功にはしていない。TF-Locoformer は 4 GPU で実行した。

## 14. 回帰

`CUDA_VISIBLE_DEVICES=""` で GPU 試験のあと直列に実行した。新規の Failure、Error、Skip は無い。`tests/experiments` は Phase 5 の profile 試験 2 件が増え、220 件から 222 件になった。Skip は 1 のままである。

| Suite | 件数 | 結果 | 時間 |
|---|---:|---|---:|
| `tests/experiments` | 222 | OK、Skip 1 | 1040 秒 |
| `tests/training` | 297 | OK、Skip 12 | 903 秒 |
| `tests/reproduction` | 135 | OK、Skip 2 | 456 秒 |

## 15. 判定

| モデル | 公式再現の開始 | 研究用比較の開始 |
|---|---|---|
| MP-SENet | READY。2×A100 40GB、world 2、micro 2、official snapshot。test 選択は独立評価ではない | CONDITIONAL READY。safe snapshot は preflight まで。同じアルゴリズムの GPU 更新は official snapshot で済んでいる |
| MossFormerGAN | READY。2×A100、world 2、accumulation 2、official snapshot | BLOCKED。safe snapshot はあるが、Trainer が `official_faithful` 以外を拒否する |
| TF-GridNet | READY。対象は ESPnet 公開レシピ。2×A100、world 2。論文実験の再現ではない | 同じ snapshot を研究用の別条件としては切っていない |
| TF-Locoformer | BLOCKED。`cv_synthetic_small` は未解決 | READY。4×A100、world 4、chunk 64,000、research valid 1,024。150 epoch は未開始 |
| DeepFilterNet3 | BLOCKED。`dfn3.online.official@1` は無い | CONDITIONAL READY。smoke、world 1、native はローカル再ビルド。実行 batch は 8 で、要求した 16 ではない。schedule 付き複数 GPU は未対応のまま |
| SEMamba++ | BLOCKED。公式素材は無く、batch 8 は未実行 | CONDITIONAL READY。smoke、fused、micro 4 まで。batch 8 と accumulation による代替は主張しない |

## 16. 残課題

- 30 分を超える本格学習は、別の明示的な起動指示が要る。
- SEMamba++ batch 8 は 40GB では未実行。80GB 級が要る可能性は、過去の 45.56 GiB の記録から残る。
- DeepFilterNet3 の公式 batch 16 以上と、schedule 付き複数 GPU は未測定。
- MossFormerGAN の leakage-safe は Trainer の claim 制約で始められない。
- TF-Locoformer の公式 `cv_synthetic_small` は未解決。
- 推定時間に validation と skip は入っていない。
- CPU メモリ peak は未測定。
- `.venvs/semambapp` の `resampy` は lock に入っていない。
