# 音声強調モデル

登録名と variant は [ARCHITECTURE.md](ARCHITECTURE.md)。公式との最大絶対誤差は [VALIDATION.md](VALIDATION.md)。ここには数値を繰り返さない。

学習の損失と更新順は `src/se` に無い。`src/experiments/` にある。

## MP-SENet

[論文](https://arxiv.org/abs/2305.13686)、[公式コード](https://github.com/yxlu-0102/MP-SENet)、参照 commit `89932cfe90d1dacb8e170e4a331d762462c21792`、MIT（`src/se/models/mp_senet/LICENSE`）。

16 kHz のモノラル強調。公式 Generator の encoder、二軸 Transformer、magnitude/phase decoder を移植している。チェックポイントは `generator` キーを `strict=True` で読む。variant は `vb` と `dns`。省略できる。重みのパスは呼び出し側が渡す。

`enhance` は torchaudio で 16 kHz にし、エネルギーが float32 で正のときだけ RMS 正規化する。無音と、二乗がアンダーフローする振幅はそのまま返す。`torch.stft` の reflect padding のため、16 kHz 換算で `n_fft // 2` サンプル以下は拒否する。振幅は `view_as_real().pow(2).sum()` の順で計算する。iSTFT には元の長さを渡す。

## TF-GridNet

[論文](https://arxiv.org/abs/2211.12433)、[ESPnet](https://github.com/espnet/espnet) commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、Apache-2.0（`src/se/models/tf_gridnet/LICENSE`）。

ESPnet の v1 / v3 の推論用 GridNet を独立構成している。ESPnet は実行時に import しない。入力はモノラル、出力は 1 話者。`n_srcs` が 1 でない構築は拒否する。2 話者分離の `wsj0-2mix` は variant に無い。

| variant | ネットワーク | フロントエンド |
|---|---|---|
| `urgent24`（省略時） | v3 | 入力サンプルレートに合わせて 8 kHz の STFT を拡縮する。`n_fft = 256 * rate / 8000`、hop はその半分、Hann。標準偏差 0 の無音は入力を返し、それ以外の定数は零波形を返す |
| `ms-snsd` | v1 | 既定は `n_fft=512`、stride 256、Hann。16 kHz 以外は torchaudio で 16 kHz にして戻す。標準偏差 0 は零波形 |

チェックポイントは `model` キーで、全キーが `separator.` で始まる state dict だけを受ける。接頭辞を外して `strict=True`。チェックポイント自体の利用条件は、コードの Apache-2.0 とは別にカードへ書かれていない。

## TF-Locoformer

[論文](https://arxiv.org/abs/2409.16674)、[公式コード](https://github.com/merlresearch/tf-locoformer) tag `v1.1.0`、commit `7a615460d347ff7334a13dbb831d16280da72cdc`、Apache-2.0（`src/se/models/tf_locoformer/LICENSE`）。

対応するのは DNS-Interspeech 2020 の単一話者強調（variant `dns`。省略できる）である。standalone の separator を同梱し、Hann STFT（既定 `n_fft=256`、hop 128）と、入力を不偏標準偏差で割る正規化を独立構成している。ESPnet は実行時に import しない。`num_spk` が 1 でない構築は拒否する。

不偏標準偏差が `1e-6` 未満の入力は、`forward` では拒否し、`enhance` では零波形を返す。16 kHz 以外は torchaudio で 16 kHz にして戻す。`n_fft // 2` 以下は拒否する。

チェックポイントは `model`。全キーが `separator.` で始まるときだけ接頭辞を外し、separator へ `strict=True` で読む。公開重みとの数値比較の条件は VALIDATION.md の未検証に残している。

## DeepFilterNet3

[論文/コード](https://github.com/Rikorose/DeepFilterNet)、参照 commit `d375b2d8309e0935d165700c91da9de862a99c31`。上流は MIT または Apache-2.0。このパッケージの `src/se/models/deepfilternet3/LICENSE` は MIT。

48 kHz、ERB 32、deep filter order 5（`DFN3Params` の既定）。実行時に `libdf` は import しない。Vorbis 窓、ERB 帯域、指数正規化は PyTorch。解析 FFT、ERB の `log10`、unit norm の平方根は `libdf_rfft.so`（`realfft` 3.3.0、ABI 1）である。カーネルは contiguous な CPU float32 を要求する。ネットワーク本体は `load_enhancer` の device へ載せる。

ライブラリの選択は `src/se/models/deepfilternet3/dsp.py`。

- `DF_RFFT_LIBRARY` が最優先。
- 無ければ、追跡ファイル `src/se/models/deepfilternet3/libdf_rfft.so` が ABI 1 で読めるときそれを使う。
- 追跡ファイルが存在するが読めないときは、`build/df-rfft/release/libdf_rfft.so` が ABI 1 で読めるときだけそれを使う。追跡ファイルが残っていてローカルビルドも使えないときは、PyTorch に戻さず `RuntimeError`。
- ファイルが無いときだけ、`DF_RFFT_REQUIRED=1` でなければ `torch.fft.rfft`、`torch.log10`、`torch.sqrt` に戻る。この戻りでは `libdf` との一致は成り立たない。`DF_RFFT_REQUIRED=1` なら `FileNotFoundError`。
- `make df-rfft` は、追跡ファイルと内容が違うビルド成果物で追跡ファイルを置き換えない。

`config.ini` も渡せる。重みは `model_120.ckpt.best` を `strict=True` で読む。各 `enhance` の開始時に STFT の overlap と指数正規化状態を捨てる。16 kHz など別レートは torchaudio で 48 kHz にして戻す。

学習側の `src/experiments/official/deepfilternet3` は `src.se.build_model("deepfilternet3")` を使う。解析カーネルは同じ `dsp.py` である。

カーネルパッケージの `Cargo.toml` は `MIT OR Apache-2.0`。`Cargo.lock` の依存は `realfft` 3.3.0 と `rustfft` 6.4.1。以前の文書は realfft を MIT、rustfft を MIT OR Apache-2.0 と書いている。クレートのライセンス本文はこのリポジトリに無い。`libdf` は再配布しない（`requirements/official-assets.toml`）。

## MossFormerGAN

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`、Apache-2.0（`src/se/models/mossformergan/LICENSE`）。重みカードも Apache-2.0。

推論は SyncANet だけである。Discriminator は含まない。公開チェックポイントの `mask_decoder.prelu_out` は長さ 201、推論の `n_fft` は 400（窓 400、hop 100、periodic Hamming）。argparse の既定 512 はこのローダでは使わない。`last_best_checkpoint.pt` の `model` を `strict=True` で読む。variant は `se_16k`。省略できる。

10 秒を超える発話は、公式 decode と同じく 10 秒窓、ストライド 75% でつなぐ。エネルギー正規化は分割前の発話全体で一度だけ計算する。無音はそのまま返す。16 kHz 以外は torchaudio で変換する。`n_fft // 2` 以下は拒否する。

## SEMamba++

参照 commit `cd369c89a8106072c077b468417b765b227afd74`。ピンしたソースツリーにライセンスファイルは無く、[重みカード](https://huggingface.co/yongjoonlee/semambapp) は GPL-2.0。ソースはコピーしていない。`src/se/models/semambapp/NOTICE` を参照。

16 kHz、`n_fft=400`、hop 100、窓 400、Hann、圧縮 0.3、hidden 48、bottleneck 4。`semambapp.pth` の `generator` を `strict=True` で読む。variant は `semambapp`。省略できる。ピーク正規化のあと STFT し、出力は入力長に戻す。無音はそのまま返す。`n_fft // 2` 以下は拒否する。16 kHz 以外は torchaudio で変換する。

選択的スキャンの既定は `scan_backend: reference` で、PyTorch の漸化式だけを使う。`mamba_ssm` は import しない。`scan_backend: fused` は CUDA 上で `mamba_inner_fn` と Triton RMSNorm を使う。CPU では参照スキャンに戻る。fused には `.venvs/semambapp` の `mamba-ssm` と `causal-conv1d` が要る。`src/se` の実行時依存には入れていない。
