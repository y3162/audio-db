# 6 モデルの学習対応

共通の Trainer から起動できるのは MP-SENet と、TF-GridNet の時間 L1 である。時間 L1 は公式の学習ではない。レベルは次のとおりである。

| レベル | 意味 |
|---|---|
| L0 | 推論のみ |
| L1 | forward / backward、Optimizer 更新、checkpoint、学習後の重みで推論 |
| L2 | 公式の主要な損失、更新順、前処理 |
| L3 | 公式との学習ステップ比較 |

| モデル | レベル | 公式学習との違い |
|---|---|---|
| MP-SENet | L2 | 損失、Discriminator のあと Generator、ExponentialLR をエポック終了時に進める。公式の `cudnn.benchmark` やファイルリスト学習は使わない。ステップ数値の比較はしていない |
| TF-GridNet | L1 | `loss: waveform_l1`。ESPnet の SI-SNR や URGENT の学習スケジュールは未実装。`ms-snsd` は波形を直接、`urgent24` は複素スペクトルを通す |
| DeepFilterNet3 | L0 | 公式は `df/train.py` の複数損失とエポックループ。縮小設定の `DfNet` 構築は `ErbDecoder` で失敗し、学習タスクは無い |
| MossFormerGAN | L0 | 公式は GAN。公開幅 `num_channel=64`、`n_fft=400`、800 サンプルの時間 L1 で 1 step の勾配は有限だった（CPU、1053 個の勾配）。縮小幅は MossFormer の `view` で失敗する。checkpoint と推論への再読込は未実装 |
| SEMamba++ | L0 | 公式は GAN と mel / STFT / PESQ 損失。ソースにライセンスファイルが無いので、その学習コードはコピーしていない。縮小設定の波形 L1 は入力によって非有限になるため、学習タスクにしていない |
| UniPASE | L0 | 公式は段階学習。Vocoder は mel 損失と GAN（`train/train_vocoder.py`）。構築に事前学習チェックポイントが要り、学習タスクは無い |
| StuPASE | L0 | 公式は DeWavLM、flow、vocoder を別々に学習する。flow は MSE（`train/train_flow.py`）。構築に事前学習チェックポイントが要り、学習タスクは無い |

TF-GridNet は `model.name: tf_gridnet` と `loss: waveform_l1` で `python -m src.training.se.cli` から起動する。チェックポイントの `model` は `separator.` 付きで、`load_enhancer("tf_gridnet")` が読む。

CPU で確認したのは、縮小した `ms-snsd` の 1 step、同じ seed の初期化、凍結したパラメータが動かないこと、学習後の重みを `load_enhancer` で読むことである。GPU と DDP ではこのタスクを動かしていない。
