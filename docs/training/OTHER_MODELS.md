# 4 モデルの学習対応

共通 Trainer はモデルの損失を知らない。損失と更新順は `src/training/se/` にある。`loss: waveform_l1` は TF-GridNet の基盤テスト用で、公式の学習ではない。

| レベル | 意味 |
|---|---|
| L0 | 推論のみ |
| L1 | forward / backward、Optimizer 更新、checkpoint、学習後の重みで推論 |
| L2 | 公式の主要な損失、更新順、前処理 |
| L3 | 公式との学習ステップ比較 |

| モデル | レベル | 公式学習との違い |
|---|---|---|
| MP-SENet | L3 | 公式幅を、公式と同じ Discriminator→Generator の順で比較する。PESQ を Discriminator の偽例目標に使う。詳細は [VERIFICATION.md](VERIFICATION.md) |
| TF-GridNet | L3 | `ms-snsd` は ESPnet の `mr_l1_tfd`（窓 256/512/768/1024、分散正規化なし、SI-SNR の重み 0）。Adam、weight decay 0、勾配クリップ 5。同じ波形に対する損失差は 0。`urgent24` は分散正規化ありで、今回の公式ステップ比較はしていない。`waveform_l1` は公式ではない。詳細は [VERIFICATION.md](VERIFICATION.md) |
| SEMamba++ | L3 | 振幅 0.9、位相 0.3、複素 0.1、整合 0.1、mel 0.1、敵対 1、特徴整合 1（マップ平均の和を 2 倍し、スケール数では割らない）。更新は Discriminator のあと Generator。周波数経路は `n_fft=400` の 100 bin に固定なので、それ以外の `n_fft` では動かない。参照ツリーはコピーしていない。Discriminator は `nnAudio` の `CQT2010v2`（hop `(512, 256, 256)`、8 octaves、bins `(12, 24, 36)`、実行時の幅 128）と STFT 解像度 `(1024, 120, 600)`、`(2048, 240, 1200)`、`(512, 50, 240)` である。設定に書かれた幅 32 は層へ渡っていない。時間 L1 は合計に入っていない。位相は `torch.diff` ではなく、最終ビンに `-1` を含む差分行列である。Hann 窓は CPU で作ってから GPU へ移す。詳細は [VERIFICATION.md](VERIFICATION.md)。VoiceBank train 10,700 件の 200 step 比較もそこにある。3 秒・batch 1 の損失差は 1e-6 まで確認した。native batch 8 は 16 GB では動かない。4 プロセス probe は hid_feature 8、1 block で、公式 Triton RMSNorm は CPU では動かない。GPU fused の 1 step は [VERIFICATION.md](VERIFICATION.md)。 |
| MossFormerGAN | L3 | 公開幅 `num_channel=64`。損失は 0.1 RI + 0.9 振幅 + 0.2 時間 + 0.05 GAN。Hamming `periodic=False`、圧縮 0.3。STFT のフレームは `center=True` と同じで、reflect は index で行う。更新は Generator backward、Discriminator backward、そのあと両方 step。Discriminator の学習率は 2 倍。PESQ が取れないバッチは 0 にせず止める。時間波形の再合成は、公開損失と同じく clean の実部と予測の虚部を使う。公開 Discriminator は NISQA の AdaptCNN のあと Conformer を 2 段（dim 384、head 4、kernel 31）と attention pool、LearnableSigmoid である。公式 `SyncANet` と同じ state dict を読む。CPU の 4 step では公式と独立の損失・勾配・重みが一致した。GPU の 1 step は再実行差より公式/独立の勾配差が大きい。20 step と accumulation 2 は未計測。詳細は [VERIFICATION.md](VERIFICATION.md)。 |
| DeepFilterNet3 | L3 | 出荷 `config.ini`（SHA-256 `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290`）の損失。multi-resolution 振幅 500、複素 500、gamma 0.3、FFT 256/512/1024/2048、local SNR 1e-3。mask、単一スペクトル、SI-SDR、ASR は 0。同じ波形に対する公式 `MultiResSpecLoss` との差は 0、local SNR の目標差も 0。公式 `DfNet` へ同じ state dict を読むと forward の最大絶対誤差は 0。4 独立プロセスの CPU 20 step では、同一実装の再実行は損失差 0 のまま、公式と独立の損失差は step 13 以降 0.05 を超える。AdamW は lr 0.001、betas `(0.9, 0.999)`、weight decay `1e-12`、`amsgrad=True`。per-step cosine は `deepfilter_cosine` で、`steps_per_epoch` を YAML に明示する。`libdf` の解析 STFT は比較していない。フロントエンドは独立 `DFStream` で、発話ごとに reset する。詳細は [VERIFICATION.md](VERIFICATION.md) |

起動は `python -m src.training.se.cli --config configs/training/<model>.yaml`。チェックポイントは `checkpoint_dir` で、`data/checkpoints/` には書けない。

SEMamba++ の CUDA 学習は `model.architecture.scan_backend: fused` で明示選択できる。既定は `reference`、CPU は指定にかかわらず reference である。fused は対応 GPU 用に構築した外部 `mamba-ssm` の `mamba_inner_fn` と Triton RMSNorm を使い、checkpoint のキーは変わらない。3 秒・2 GPU の実測は [VERIFICATION.md](VERIFICATION.md) を参照する。20 step の勾配スナップショットは公式の再実行でも反転するので、それを長時間学習後の品質一致の証拠にしない。
