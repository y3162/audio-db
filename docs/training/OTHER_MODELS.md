# 6 モデルの学習対応

共通 Trainer はモデルの損失を知らない。損失と更新順は `src/training/se/` にある。`loss: waveform_l1` は TF-GridNet の基盤テスト用で、公式の学習ではない。

| レベル | 意味 |
|---|---|
| L0 | 推論のみ |
| L1 | forward / backward、Optimizer 更新、checkpoint、学習後の重みで推論 |
| L2 | 公式の主要な損失、更新順、前処理 |
| L3 | 公式との学習ステップ比較 |

| モデル | レベル | 公式学習との違い |
|---|---|---|
| MP-SENet | L3 | CPU で公式モジュールと 1 step を比較した。STFT・Generator 出力・Discriminator 出力・損失の最大絶対誤差は 0。Generator 勾配の最大絶対誤差は 6.43e-8。AdamW 更新後の重みの最大絶対誤差は 4.05e-4。この重み差は、初回 AdamW の閉形式と一致する。詳細は [VERIFICATION.md](VERIFICATION.md) |
| TF-GridNet | L2 | 学習損失は ESPnet の `mr_l1_tfd`。`urgent24` は分散正規化あり、`ms-snsd` はなし。SI-SNR は損失に足さない。`ms-snsd` の公式幅（`n_fft=512`、4 層、`emb_dim=32`、LSTM 128）でも 1 step を確認した。`urgent24` の単体テストは公式より狭い。`waveform_l1` は残してあるが公式ではない |
| SEMamba++ | L2 | 振幅 0.9、位相 0.3、複素 0.1、整合 0.1、mel 0.1、敵対 1、特徴整合 1（中で 2 倍）。更新は Discriminator のあと Generator。周波数経路は `n_fft=400` の 100 bin に固定なので、それ以外の `n_fft` では動かない。参照リポジトリに LICENSE が無い。公開設定の CQT Discriminator は hop `(512, 256, 256)`、8 octaves、bins `(12, 24, 36)` で、`nnAudio` の `CQT2010v2` を使う。lab に `nnAudio` は無く、コードはコピーしていない。ここにあるのは独立した multi-resolution / sub-band Discriminator で、公式の CQT Discriminator ではない。時間 L1 は公式の合計に入っていない。CUDA の `mamba_ssm` 融合カーネルとの学習比較はしていない |
| MossFormerGAN | L2 | 公開幅 `num_channel=64`。損失は 0.1 RI + 0.9 振幅 + 0.2 時間 + 0.05 GAN。Hamming `periodic=False`、圧縮 0.3。STFT のフレームは `center=True` と同じで、reflect は index で行う。更新は Generator backward、Discriminator backward、そのあと両方 step。Discriminator の学習率は 2 倍。PESQ が取れないバッチは 0 にせず止める。時間波形の再合成は、公開損失と同じく clean の実部と予測の虚部を使う。公開 Discriminator は NISQA の AdaptCNN のあと Conformer を 2 段（dim 384、head 4、kernel 31）と attention pool、LearnableSigmoid である。ここにあるのは畳み込みの magnitude pair で、その Conformer は無い。損失係数が同じであることは公式再現ではない |
| DeepFilterNet3 | L2 | 出荷 `config.ini`（SHA-256 `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290`）の損失。multi-resolution 振幅 500、複素 500、gamma 0.3、FFT 256/512/1024/2048、local SNR 1e-3。mask、単一スペクトル、SI-SDR、ASR は 0。公式幅 `conv_ch=64` で 1 step した。AdamW は lr 0.001、betas `(0.9, 0.999)`、weight decay `1e-12`、`amsgrad=True`。エポック内ステップ数に依存する cosine（warmup 3 epoch、1e-4 から 1e-3、下限 1e-6、weight decay は 0.01 へ）は未適用。`emb_num_layers=1` は ErbDecoder を作れない。発話ごとに `DFStream` を reset する。公式モジュールとの数値比較はしていない |
| UniPASE | L1 | 4 段階を別 Optimizer にした。DeWavLM は層特徴の MSE。Vocoder は mel 15、敵対 0.5、特徴 0.5。Adapter は特徴 MSE 200、敵対 1、特徴 1。PostNet は mel 15、敵対 0.5、特徴 0.5。更新は Generator のあと Discriminator。AdamW は betas `(0.8, 0.99)`、weight decay 0.01。単体テストは縮小ネットである。Adapter の入力は凍結した線形写像で、DeWavLM-Omni ではない。WavLM-Large が無いので公式の教師では学習していない。線形 warmup cosine は未適用 |
| StuPASE | L1 | Vocoder の係数は 15 / 1 / 1、CFM は `forward` の流速 MSE で ODE の `sample` は呼ばない。更新順は Generator のあと Discriminator。mel の帯と窓は公開リスト。周期は `(2, 3, 5, 7, 11)`、STFT 解像度は `(512, 1024, 2048)`。畳み込み幅は HiFi-GAN より狭い。DeWavLM は縮小 WavLM の層 MSE だけ実行した。WavLM-Large.pt、DeWavLM.pt、Vocoder_L24.pt は無く、公式教師では学習していない。テキスト条件はゼロ。公式幅の Vocos と CFM の学習 step は未実行 |

起動は `python -m src.training.se.cli --config configs/training/<model>.yaml`。チェックポイントは `checkpoint_dir` で、`data/checkpoints/` には書けない。
