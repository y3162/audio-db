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
| MP-SENet | L3 | CPU で公式モジュールと 1 step を比較した。STFT・Generator 出力・Discriminator 出力・損失の最大絶対誤差は 0。Generator 勾配の最大絶対誤差は 6.43e-8。AdamW 更新後の重みの最大絶対誤差は 4.05e-4（勾配差が Adam の epsilon で増幅された） |
| TF-GridNet | L2 | 学習損失は ESPnet の `mr_l1_tfd`。`urgent24` は分散正規化あり、`ms-snsd` はなし。SI-SNR は損失に足さない。単体テストのネットワーク幅は公式より狭い。`waveform_l1` は残してあるが公式ではない |
| SEMamba++ | L2 | 振幅 0.9、位相 0.3、複素 0.1、整合 0.1、mel 0.1、敵対 1、特徴整合 1（中で 2 倍）。更新は Discriminator のあと Generator。周波数経路は `n_fft=400` の 100 bin に固定なので、それ以外の `n_fft` では動かない。参照リポジトリにライセンスが無いため、Multi-resolution / sub-band Discriminator は独立実装。公式の CQT Discriminator そのものではない。時間 L1 は公式の合計に入っていない |
| MossFormerGAN | L2 | 公開幅 `num_channel=64`。損失は 0.1 RI + 0.9 振幅 + 0.2 時間 + 0.05 GAN。Hamming `periodic=False`、`center=True`、圧縮 0.3。更新は Generator backward、Discriminator backward、そのあと両方 step。Discriminator の学習率は 2 倍。PESQ が取れないバッチは 0 にせず止める。時間波形の再合成は、公開損失と同じく clean の実部と予測の虚部を使う。Discriminator 本体に Conformer は無い |
| DeepFilterNet3 | L1 | 発話ごとに `DFStream` を reset する。`emb_num_layers=1` は ErbDecoder の GRU 層数が 0 になり構築できない。学習 ini が無いので、mask 1、スペクトル振幅 1、SI-SDR 0.2、local SNR 5e-4 はソースの例であって、出荷された config ではない。公式幅 `conv_ch=64` の学習 step は未実行 |
| UniPASE | L0 | DeWavLM 段階は WavLM-Large.pt が無く実行しない。Vocoder / Adapter / PostNet は凍結した DeWavLM が要る。そのグラフの 1 step は実行していない |
| StuPASE | L1 | mel vocoder の GAN（15 / 1 / 1）と CFM の流速 MSE は、縮小ネットワークで 1 step できた。CFM は `forward` のみで、`sample` の ODE は呼ばない。テキスト条件はゼロで、DeWavLM-R ではない。DeWavLM 段階は WavLM-Large が無く実行しない。公式幅の Vocos と CFM.pt の学習 step は未実行 |

起動は `python -m src.training.se.cli --config configs/training/<model>.yaml`。チェックポイントは `checkpoint_dir` で、`data/checkpoints/` には書けない。
