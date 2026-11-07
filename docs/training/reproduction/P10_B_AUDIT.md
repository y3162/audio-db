# P10-B 監査（2026-11-07）

公式幅の短時間 GPU スモークに使う 4 モデルの学習設定と、その時点の実装差である。本格再現学習ではない。epoch 全体、全件 validation、収束は見ていない。

対象 Dataset は次のとおり。DeepFilterNet3 と SEMamba++ の公式 Dataset は blocked のままである。

| Model | Dataset | 公式の実効 GPU |
|---|---|---|
| MP-SENet | `mp_senet.vb28.ondemand.official@1` | 2 GPU、global batch 4 |
| MossFormerGAN | `mossformergan.vb28.ondemand.official@1` | 2 GPU、GPU あたり micro 1 × 蓄積 2、global batch 4 |
| TF-GridNet | `tf_gridnet.ms_snsd.espnet_recipe@1` | 2 rank、global batch 4 |
| TF-Locoformer | `tf_locoformer.dns.accept_recipe@1` | 4 rank、rank あたり batch 1、global batch 4 |

スモーク設定は `configs/training/reproduction/p10b/`。`max_steps` は 2 で、再開比較のため実行スクリプトが 3 まで上げる。`max_updates_per_epoch`、`validation_max_batches`、`chunk_iterator.max_updates`、`chunk_iterator.validation_max_updates` はスモークの上限であり、公式 epoch ではない。

## 公式設定と実装

### MP-SENet

公式は README の 2 GPU、global batch 4、AdamW（lr 5e-4、betas 0.8 / 0.99、weight decay 0.01）、ExponentialLR gamma 0.99、公式損失、segment 32,000（2 秒）である。Dataset は test を validation に使う（`selection_uses_test` true）。学習設定の `allow_test_selection: true` と組で開始する。

スモーク YAML の `micro_batch_size_per_gpu: 4` は上限である。2 GPU では実効 micro 2、蓄積 1 になり、global batch 4 を保つ。TSUBAME のログは `global=4 micro=2 world=2 accumulation=1` だった。

`best_checkpoint_start_epoch` はスモークだけ 0 である。公開ゲートは epoch 40 なので、この best ファイルを公式の選択規則とは呼ばない。validation は公式順の先頭 2 batch である。

### MossFormerGAN

公式 `train.sh` は `n_gpu=2`。YAML は `batch_size: 1`、`accu_grad: 2`、`effec_batch_size: 2`（GPU あたり）。2 GPU で global batch 4。AdamW の実効 weight decay は 0.01、識別器の学習率は生成器の 2 倍。segment は 32,000（公式の 2 秒 crop）。開始は scratch で、初期 checkpoint は渡さない。

検証損失は時間領域 L1 ではない。生成器の公式目的 `0.1 * RI + 0.9 * magnitude + 0.2 * time + 0.05 * GAN` を、rank 0 の公式順の先頭 2 batch で平均する。PESQ が計算できない batch は 0 として分母に残る。識別器は `train` のまま、どちらの Optimizer も進めない。Plateau に渡す値はこの生成器損失である。5 回非改善で生成器の学習率を半分にし、10 回で止める処理は checkpoint の `plateau` に残る。今回の 1 回の検証では半分にする回数に達しない。

`scheduler.name` は `none` である。公開 Solver に `lr_scheduler` オブジェクトは無い。

### TF-GridNet

公式 MS-SNSD recipe は 2 rank、global batch 4、rank あたり batch 2、Adam（lr 1e-3）、勾配クリップ 5、`mr_l1_tfd`、ReduceLROnPlateau（factor 0.7、patience 1）。`num_samples_per_epoch: 5000` は rank あたりの元発話数であり、Optimizer 更新数ではない。チャンクは 48,000、shift 0.5。`init: xavier_uniform`。検証の選択は `valid.loss`。`si_snr` は公式 stats の選択キーではない。

スモークは `max_updates: 2` と `validation_max_updates: 1` で、epoch 末の経路を 1 回だけ通す。この 1 batch は `cv_ms_snsd` でも、valid 1,835 件全体でもない。

### TF-Locoformer

公式 DNS 構成は medium 相当の既定（`n_layers` 6、`emb_dim` 128、`attention_dim` 128、`ffn_hidden_dim` 384、`n_fft` 256、`hop` 128）。YAML の `architecture: {}` と `stft: {}` はこの既定を使う。4 rank、global batch 4、rank あたり batch 1、チャンク 64,000、shift 0.5。`num_samples_per_epoch: 1000` は rank あたりの元発話数。AdamW（lr 1e-3、weight decay 0.01）、勾配クリップ 5、`mr_l1_tfd`（時間重み 0.5）。Scheduler は `WarmupReduceLROnPlateau`（warmup 4,000、factor 0.5、patience 3）。最初の Optimizer 更新は構築時の学習率を使い、その後 `initial_lr / 4000 * step_num` を書く。JSONL と TensorBoard の学習率は更新後の値である。

検証は sorted uttid の末尾 1/10（36,000 件）であり、`cv_synthetic_small` ではない。スモークは rank あたり先頭 1 chunk だけを集約する。有限な `valid.loss` が 5 件未満のときは平均ファイルを書かず、`valid.loss.ave_1best.pt` を最良 epoch へのリンクにする。`valid.si_snr.best.pt` は SI-SDR（dB）を最大化する研究側のファイルで、公式 stats の `si_snr` キーではない。

Dataset は `tf_locoformer.dns.accept_recipe@1`。snapshot hash `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` は [P10_A1_VERIFICATION](P10_A1_VERIFICATION.md) と一致した。`selection_uses_test` は false。

## 今回入れた修正

作業ツリーの学習差分は `7b3072f`（`[WIP]`）にある。

- MossFormerGAN の `validation_report` を、公開 validator の生成器損失に合わせた。時間領域 L1 だけを検証損失にしない。
- スモーク用の任意キーを設定パーサに足した。キーが無い既存 YAML の意味は変えていない。
- TF-GridNet は `model.init` があるとき `apply_espnet_initialization` を呼ぶ。`dim > 1` を xavier し、`nn.LayerNorm` と `nn.GroupNorm` だけをモジュール既定へ戻す。独自の 4 次元正規化は `nn.LayerNorm` ではないので gamma は xavier のまま残る。これは以前の公式監査と同じ規則である。今回、公開 checkpoint の gamma 統計は測り直していない。
- `max_steps` に達すると `complete.json` が書かれ、同じディレクトリの再開は拒否される。GPU スクリプトは再開の前にこれを `complete.phase-train.json` へ移す。
- スクリプトの作業ディレクトリは、スクリプト自身から見たリポジトリルートである。共有ホストでは `CUDA_VISIBLE_DEVICES` の個数を要求 GPU 数と一致させる。

## 未解決の差

このスモークが公式再現だと主張しないもの。

| 項目 | 状態 |
|---|---|
| 全件 validation | 未実施。各モデルは固定した接頭辞だけ |
| 公式 epoch 数と収束 | 未実施。更新は 3 回まで |
| CUDA のビット一致 | `deterministic: false`。同じ case id でも 2 step 目以降の損失は完全には一致しない |
| Mixed precision | 学習ステップは GradScaler も autocast も使わない。今回、公式 YAML の AMP 指定は開き直していない |
| MP-SENet の 16 kHz 変換器 | 公式の変換器は不明のまま。学習時は soxr HQ と PCM_16 |
| MP-SENet の best ゲート | スモークは epoch 0 から best を書く。公開は epoch 40 |
| MossFormerGAN の構築回数 | 公式はネットワークを複数回構築して最初の Generator と最後の Discriminator を使う。ここは seed を分けて 1 回ずつ構築する |
| MossFormerGAN の plateau 半減 | 状態は保存した。5 回非改善の半減は GPU では起こしていない |
| TF-GridNet の shape file 順 | 生成物が無く、基底順は呼び出し側の発話列のまま |
| TF-GridNet の未シード生成 | 公式は未シード。audio-db の seed は文書化した差 |
| TF-Locoformer の `cv_synthetic_small` | UNKNOWN のまま |
| TF-Locoformer の `ave_5best` | 検証 epoch が 1 回なので平均ファイルは無い。`ave_1best` リンクだけ |
| 公式の共有乱数カウンタ | DNS 受理レンダラは案件ごとの Generator。ビット一致は主張しない |

## 開始条件

本格学習の入口は、上の Dataset と公式幅の batch / world size を維持した別 YAML である。`configs/training/reproduction/p10b/` の上限キーを本格学習に残さない。`h_rt` が 30 分以上になる TSUBAME ジョブは、この監査だけでは投入しない。

DeepFilterNet3 の `dfn3.online.official@1` と SEMamba++ の `semambapp.gsr.official@1` は blocked のままである。今回の共通コード変更は、その状態を ready にしない。
