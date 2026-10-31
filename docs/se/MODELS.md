# モデルの実装状況

## MP-SENet

[論文](https://arxiv.org/abs/2305.13686)、[公式コード](https://github.com/yxlu-0102/MP-SENet)、参照 commit `89932cfe90d1dacb8e170e4a331d762462c21792`、MIT（ライセンスは `src/se/models/mp_senet/LICENSE`）。公式 Generator の encoder、二軸 Transformer と magnitude/phase decoder および Transformer を必要最小限移植した。学習損失・discriminator は含まない。16 kHz モノラル音声強調、`generator` キーを持つ PyTorch checkpoint。公式の VB と DNS の両重みで検証。依存は PyTorch と torchaudio。重みはユーザーが明示指定する。詳細な検証限界は VALIDATION.md を参照。

## DeepFilterNet3

[論文/コード](https://github.com/Rikorose/DeepFilterNet)、参照 commit `d375b2d8309e0935d165700c91da9de862a99c31`、MIT または Apache-2.0（`src/se/models/deepfilternet3/LICENSE`）。48 kHz、ERB 32、deep filter order 5。`libdf` は使わず、公開されている Rust の Vorbis 窓・非正規化 real FFT・ERB パワー・指数正規化を PyTorch で実装した。重みは `model_120.ckpt.best` を `strict=True` で読む。`config.ini` も渡せる。

## MossFormerGAN

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`、Apache-2.0（`src/se/models/mossformergan/LICENSE`）。推論対象は SyncANet。Discriminator は含まない。公開チェックポイントの `mask_decoder.prelu_out` は長さ 201、推論 YAML の `fft_len` は 400 なので `n_fft=400`。argparse の 512 はこの重みでは使わない。`last_best_checkpoint.pt` の `model` を `strict=True` で読む。10 秒を超える発話は公式 decode と同じく 10 秒窓、ストライド 75% でつなぐ。CPU の 20 秒は最大絶対誤差 0。同じ GPU 上では 25 秒まで最大 2.33e-6。

## SEMamba++

参照 commit `cd369c89a8106072c077b468417b765b227afd74`。ソースツリーにライセンスファイルはなく、重みカードは GPL-2.0 表示。ソースはコピーしていない。`src/se/models/semambapp/` は公開設定（hidden 48、4 bottleneck、圧縮 0.3）とチェックポイントのモジュール名に合わせた再実装で、Mamba の選択的スキャンは Apache-2.0 の Mamba 漸化式を PyTorch だけで計算する。`semambapp.pth` の `generator` を `strict=True` で読む。実チェックポイントでは missing も unexpected も 0。`mamba_ssm` は未使用。公式 CUDA 融合カーネル（`mamba-ssm==2.3.2.post1`、`causal-conv1d==1.7.0`）との波形差は 20 秒で 1.80e-5。単一 mixer では融合カーネルと PyTorch 漸化式の差は 3.28e-7。

## UniPASE

参照 commit `857b60ad05d37a2cf6d7a89883ec9fc4fc164b45`、MIT。第三者部品は各ファイル先頭のライセンス表記に従う。推論グラフは DeWavLM、Adapter、Vocoder_DWO-L1、PostNet。`omegaconf` と `espnet2` は実行時に不要な import を外し、活性化名の解決だけを同梱した。20 秒以下は公式 `inference.py` の全文経路、20 秒を超える入力は `inference_long.py` の 8 秒窓・4 秒 hop・重なり切り捨て。出力は `max(|y|)+1e-8` で入力ピークへ戻す。16 kHz 超は PostNet を通す。

## StuPASE

PASE 参照 commit `45614efae73dc49bf36018d1ca6a9319b3050038`、Apache-2.0 と同梱の個別ライセンス。DeWavLM-R、CFM、Vocoder_Mel-16k。`torchdiffeq` と `x-transformers` は lab にないため、Euler 積分と rotary を `flow/compat.py` に置いた。rotary は x-transformers 2.14.2 の隣接ペア回転。CFM の `load_state_dict` は上流と同じ `strict=False`。配布中の `CFM.pt` と `Vocoder_Mel-16k.pt` は、公式ローダでも独立ローダでも missing も unexpected も 0 件。公式に長時間用の分割はない。1 秒から 20 秒まで、seed を揃えると波形も ODE 軌跡も最大絶対誤差 0。

## 追記: 以前の未実装調査

上の記述のうち、短音声だけを確認したという部分は `VALIDATION.md` の途中記録である。最終的な波形比較は同ファイルの末尾を参照する。公式 CLI のファイル名や PCM 16 書き出しは、波形が一致していても別項目のまま残している。

## 追記: TF-GridNet（部分実装）

[論文](https://arxiv.org/abs/2211.12433)、[ESPnet](https://github.com/espnet/espnet) commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、Apache-2.0（`src/se/models/tf_gridnet/LICENSE`）。ESPnet の v1/v3 の推論用 GridNet ブロックと正規化層を引用し、前後処理とネットワークを独立構成した。ESPnet、Trainer、データローダーへの実行時依存はない。入力はモノラル、出力は単一話者。`urgent24`（v3、可変サンプルレート）と `ms-snsd`（v1、16 kHz）に対応し、2話者分離 `wsj0-2mix` は未対応。公式 checkpoint は ESPnet の `separator.` 接頭辞付き state dict、パスは明示指定。チェックポイント自体の利用条件は未確認。urgent24 と ms-snsd は 20 秒まで公式 separator と最大絶対誤差 0。ESPnet の推論コマンド全体は起動していない。

## ライセンスと依存（調査結果）

- SEMamba++: 参照 commit にライセンスファイルはない。[重みカード](https://huggingface.co/yongjoonlee/semambapp) は GPL-2.0。ソースはコピーせず、チェックポイント互換の再実装にした。`mamba_ssm` は使っていない。
- MossFormerGAN: コードと[重みカード](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K) は Apache-2.0。推論は SyncANet のみ。
- DeepFilterNet3: コードは MIT または Apache-2.0。DSP は `libdf` なしの PyTorch 実装。
- UniPASE: コードは MIT。同梱の第三者コードには個別ライセンスがある。[重みカード](https://huggingface.co/Xiaobin-Rong/unipase) は Apache-2.0。
- StuPASE: PASE は Apache-2.0 と追加の個別ライセンス。推論既定は steps=8、CFG=0.5、sway=-1.0。Euler と rotary は同梱の代替実装。
