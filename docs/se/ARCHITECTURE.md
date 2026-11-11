# 音声強調 API

`src.se` は推論と、学習前の `nn.Module` 構築だけを公開する。学習は `src/experiments/` だけが行う。`enhance` は推論専用で、損失も更新順も含まない。`src/se` は他の `src/` を import しない。

```python
from src.se import available_enhancers, build_model, load_enhancer, model_metadata

print(available_enhancers())
model = build_model("mp_senet")
enc = load_enhancer("mp_senet", checkpoint="/path/to/g_best_vb", device="cpu")
enhanced = enc.enhance(noisy, sample_rate=16000)
model_urgent = build_model("tf_gridnet", variant="urgent24")
model_ms = build_model("tf_gridnet", variant="ms-snsd")
print(model_metadata("tf_gridnet"))
```

| 関数 | 戻り |
|---|---|
| `available_enhancers()` | 登録名のタプル |
| `model_metadata(name)` | `ModelSpec`（構築関数、ローダ、サンプルレート、別名、variant） |
| `build_model(name, config=None, *, variant=None)` | 勾配を維持する `nn.Module` |
| `load_enhancer(name, checkpoint, device="cpu", config=None, *, variant=None)` | `enhance` を持つオブジェクト |

登録名は `mp_senet`、`tf_gridnet`、`tf_locoformer`、`deepfilternet3`、`mossformergan`、`semambapp`。別名は `mp-senet`、`tf-gridnet`、`tf-locoformer`、`deepfilternet`、`dfn3`、`mossformergan_se_16k`、`semamba++`、`se_mamba_pp`。

`variant` を省略すると、TF-GridNet は `urgent24` になる。それ以外は、省略が各モデルの既定構成になる。TF-GridNet の `urgent24` と `ms-snsd` は構造が違うので、ms-snsd では `variant` を渡す。`wsj0-2mix` は登録していない。`ModelSpec.sample_rate` は登録値で、TF-GridNet は 8000 である。`ms-snsd` の処理レートは 16 kHz である。

## 入出力

`enhance(waveform, sample_rate)` は 1 次元の有限な浮動小数点 Tensor と正の整数サンプルレートを受け、元の長さ、dtype、デバイスの Tensor を返す。空、複数チャンネル、NaN、Inf は拒否する。ストリーミング用の API は無い。

`enhance_batch(waveforms, sample_rate, lengths=None)` は `(batch, time)` を発話ごとに `enhance` し、`(zero-padded output, lengths)` を返す。バッチ内でサンプルレートは一つである。

チェックポイントはパスか、必要な名前からパスへの辞書である。名前の過不足は拒否する。state dict は `strict=True`。設定は辞書、JSON、YAML。YAML には PyYAML が要る。flat mapping であり、公式の ESPnet YAML 全体を読む機能ではない。DeepFilterNet3 は `config.ini` も受ける。

モデルごとのレート、窓、無音、最短長は [MODELS.md](MODELS.md)。公式との最大絶対誤差は [VALIDATION.md](VALIDATION.md)。

## モデルごとの処理

共通なのは、別レートの入出力を torchaudio で変換することと、出力を入力の長さに戻すことである。16 kHz への変換、RMS 正規化、複素 STFT は MP-SENet の処理であり、全モデルの処理ではない。

| モデル | チェックポイントの名前 | 処理の要点 |
|---|---|---|
| MP-SENet | `generator` | 16 kHz。RMS 正規化、振幅圧縮、Hann。無音と極小振幅はそのまま返す |
| TF-GridNet `urgent24` | `model`（`separator.` 接頭辞） | 入力レートに合わせて STFT を拡縮する。内部で 16 kHz にしない |
| TF-GridNet `ms-snsd` | 同上 | 16 kHz、`n_fft=512`、stride 256 |
| TF-Locoformer | `model` | 16 kHz、`n_fft=256`、hop 128。不偏標準偏差が `1e-6` 未満なら零波形 |
| DeepFilterNet3 | `model` | 48 kHz。各 `enhance` の開始時に STFT の overlap と指数正規化を捨てる。解析 FFT は `libdf_rfft.so`。ファイルが無いときの戻り先は MODELS.md |
| MossFormerGAN | `model` | 16 kHz、`n_fft=400`、periodic Hamming。10 秒超は 10 秒窓、ストライド 75%。エネルギー正規化は発話全体で一度 |
| SEMamba++ | `generator` | 16 kHz、`n_fft=400`、hop 100、Hann。既定のスキャンは PyTorch。`scan_backend: fused` のときだけ CUDA 上で `mamba_ssm` を使う |

新規のモデルは、独立した `nn.Module` と前後処理を足し、`src/se/core/registry.py` の `ModelSpec` へ登録する。
