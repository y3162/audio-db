# モデルの実装状況

## MP-SENet

[論文](https://arxiv.org/abs/2305.13686)、[公式コード](https://github.com/yxlu-0102/MP-SENet)、参照 commit `89932cfe90d1dacb8e170e4a331d762462c21792`、MIT（ライセンスは `src/se/models/mp_senet/LICENSE`）。公式 Generator の encoder、二軸 Transformer と magnitude/phase decoder および Transformer を必要最小限移植した。学習損失と Metric Discriminator は `src/se` には無く、`src/experiments/` にある。16 kHz モノラル音声強調、`generator` キーを持つ PyTorch checkpoint。公式の VB と DNS の両重みで検証。依存は PyTorch と torchaudio。重みはユーザーが明示指定する。詳細な検証限界は VALIDATION.md を参照。

## DeepFilterNet3

[論文/コード](https://github.com/Rikorose/DeepFilterNet)、参照 commit `d375b2d8309e0935d165700c91da9de862a99c31`、MIT または Apache-2.0（`src/se/models/deepfilternet3/LICENSE`）。48 kHz、ERB 32、deep filter order 5。実行時に `libdf` は import しない。Vorbis 窓、ERB 帯域、指数正規化は PyTorch で、解析 FFT と ERB の `log10` と unit norm は `realfft` 3.3.0 の共有ライブラリ（`make df-rfft`）で計算する。ライブラリが無いときは PyTorch に戻り、libdf とのビット一致は成り立たない。重みは `model_120.ckpt.best` を `strict=True` で読む。`config.ini` も渡せる。

## MossFormerGAN

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`、Apache-2.0（`src/se/models/mossformergan/LICENSE`）。推論対象は SyncANet。Discriminator は含まない。公開チェックポイントの `mask_decoder.prelu_out` は長さ 201、推論 YAML の `fft_len` は 400 なので `n_fft=400`。argparse の 512 はこの重みでは使わない。`last_best_checkpoint.pt` の `model` を `strict=True` で読む。10 秒を超える発話は公式 decode と同じく 10 秒窓、ストライド 75% でつなぐ。CPU の 20 秒は最大絶対誤差 0。同じ GPU 上では 25 秒まで最大 2.33e-6。

## SEMamba++

参照 commit `cd369c89a8106072c077b468417b765b227afd74`。ソースツリーにライセンスファイルはなく、重みカードは GPL-2.0 表示。ソースはコピーしていない。`src/se/models/semambapp/` は公開設定（hidden 48、4 bottleneck、圧縮 0.3）とチェックポイントのモジュール名に合わせた再実装で、Mamba の選択的スキャンは Apache-2.0 の Mamba 漸化式を PyTorch だけで計算する。`semambapp.pth` の `generator` を `strict=True` で読む。実チェックポイントでは missing も unexpected も 0。既定の scan は `mamba_ssm` を使わない。`scan_backend: fused` 指定時は CUDA 上で外部 `mamba-ssm` の `mamba_inner_fn` と Triton RMSNorm を使い、CPU では従来の PyTorch scan に戻る。公式 CUDA 融合カーネル（`mamba-ssm==2.3.2.post1`、`causal-conv1d==1.7.0`）との波形差は 20 秒で 1.80e-5。単一 mixer では融合カーネルと PyTorch 漸化式の差は 3.28e-7。

## TF-Locoformer

[論文](https://arxiv.org/abs/2409.16674)、[公式コード](https://github.com/merlresearch/tf-locoformer) tag `v1.1.0`、commit `7a615460d347ff7334a13dbb831d16280da72cdc`、Apache-2.0（`src/se/models/tf_locoformer/LICENSE`）。対応するのは DNS-Interspeech 2020 の単一話者強調（`dns`）である。standalone の separator を引用し、Hann STFT（`n_fft=256`、`hop=128`）と ESPnet の `normalize_variance`（入力を不偏標準偏差で割る）を独立構成した。不偏標準偏差が `1e-6` 未満の入力は、学習では拒否し、推論では零波形を返す。ESPnet は実行時に import しない。2 話者分離の重みは読み込まない。公開重みの CPU 比較は最大絶対誤差 0、空の Quadro RTX 5000 では 20 秒まで `1e-5` 未満。手順と表は [TF-Locoformer](../../.agents/ARCHITECTURE.md)。

## 追記: 以前の未実装調査

上の記述のうち、短音声だけを確認したという部分は `VALIDATION.md` の途中記録である。最終的な波形比較は同ファイルの末尾を参照する。公式 CLI のファイル名や PCM 16 書き出しは、波形が一致していても別項目のまま残している。

## 追記: TF-GridNet（部分実装）

[論文](https://arxiv.org/abs/2211.12433)、[ESPnet](https://github.com/espnet/espnet) commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、Apache-2.0（`src/se/models/tf_gridnet/LICENSE`）。ESPnet の v1/v3 の推論用 GridNet ブロックと正規化層を引用し、前後処理とネットワークを独立構成した。ESPnet、Trainer、データローダーへの実行時依存はない。入力はモノラル、出力は単一話者。`urgent24`（v3、可変サンプルレート）と `ms-snsd`（v1、16 kHz）に対応し、2話者分離 `wsj0-2mix` は未対応。公式 checkpoint は ESPnet の `separator.` 接頭辞付き state dict、パスは明示指定。チェックポイント自体の利用条件は未確認。urgent24 と ms-snsd は 20 秒まで公式 separator と最大絶対誤差 0。ESPnet の推論コマンド全体は起動していない。

## ライセンスと依存（調査結果）

- SEMamba++: 参照 commit にライセンスファイルはない。[重みカード](https://huggingface.co/yongjoonlee/semambapp) は GPL-2.0。ソースはコピーせず、チェックポイント互換の再実装にした。`mamba_ssm` は使っていない。
- MossFormerGAN: コードと[重みカード](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K) は Apache-2.0。推論は SyncANet のみ。
- DeepFilterNet3: コードは MIT または Apache-2.0。解析カーネルは `realfft` 3.3.0（MIT）と `rustfft` 6.4.1（MIT OR Apache-2.0）を静的に含み、libdf は再配布しない。
