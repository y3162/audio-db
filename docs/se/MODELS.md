# モデルの実装状況

## MP-SENet

[論文](https://arxiv.org/abs/2305.13686)、[公式コード](https://github.com/yxlu-0102/MP-SENet)、参照 commit `89932cfe90d1dacb8e170e4a331d762462c21792`、MIT（ライセンスは `src/se/models/mp_senet/LICENSE`）。公式 Generator の encoder、二軸 Transformer と magnitude/phase decoder および Transformer を必要最小限移植した。学習損失・discriminator は含まない。16 kHz モノラル音声強調、`generator` キーを持つ PyTorch checkpoint。公式の VB と DNS の両重みで検証。依存は PyTorch と torchaudio。重みはユーザーが明示指定する。詳細な検証限界は VALIDATION.md を参照。

## DeepFilterNet3

[論文/コード](https://github.com/Rikorose/DeepFilterNet)、参照 commit `d375b2d8309e0935d165700c91da9de862a99c31`、MIT または Apache-2.0（`src/se/models/deepfilternet3/LICENSE`）。48 kHz、ERB 32、deep filter order 5。`libdf` は使わず、公開されている Rust の Vorbis 窓・非正規化 real FFT・ERB パワー・指数正規化を PyTorch で実装した。重みは `model_120.ckpt.best` を `strict=True` で読む。`config.ini` も渡せる。

## MossFormerGAN

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`、Apache-2.0（`src/se/models/mossformergan/LICENSE`）。推論対象は SyncANet。Discriminator は含まない。公開チェックポイントの周波数次元は 201 なので `n_fft=400`。`last_best_checkpoint.pt` の `model` を `strict=True` で読む。

## SEMamba++

参照 commit `cd369c89a8106072c077b468417b765b227afd74`。ソースツリーにライセンスファイルはなく、重みカードは GPL-2.0 表示。ソースはコピーしていない。`src/se/models/semambapp/` は公開設定（hidden 48、4 bottleneck、圧縮 0.3）とチェックポイントのモジュール名に合わせた再実装で、Mamba の選択的スキャンは Apache-2.0 の Mamba 漸化式を PyTorch だけで計算する。`semambapp.pth` の `generator` を `strict=True` で読む。`mamba_ssm` は未使用。

## UniPASE

参照 commit `857b60ad05d37a2cf6d7a89883ec9fc4fc164b45`、MIT。第三者部品は各ファイル先頭のライセンス表記に従う。推論グラフは DeWavLM、Adapter、Vocoder_DWO-L1、PostNet。`omegaconf` と `espnet2` は実行時に不要な import を外し、活性化名の解決だけを同梱した。

## StuPASE

PASE 参照 commit `45614efae73dc49bf36018d1ca6a9319b3050038`、Apache-2.0 と同梱の個別ライセンス。DeWavLM-R、CFM、Vocoder_Mel-16k。`torchdiffeq` と `x-transformers` は lab にないため、Euler 積分と rotary を `flow/compat.py` に置いた。CFM の `load_state_dict` は上流と同じ `strict=False`。

## 追記: 以前の未実装調査

上の5モデルは短音声の構築・ロード・推論まで進んでいる。公式CLIとの数値一致は未了。詳細は VALIDATION.md と IMPLEMENTATION_STATUS.md。

## 追記: TF-GridNet（部分実装）

[論文](https://arxiv.org/abs/2211.12433)、[ESPnet](https://github.com/espnet/espnet) commit `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a`、Apache-2.0（`src/se/models/tf_gridnet/LICENSE`）。ESPnet の v1/v3 の推論用 GridNet ブロックと正規化層を引用し、前後処理とネットワークを独立構成した。ESPnet、Trainer、データローダーへの実行時依存はない。入力はモノラル、出力は単一話者。`urgent24`（v3、可変サンプルレート）と `ms-snsd`（v1、16 kHz）に対応し、2話者分離 `wsj0-2mix` は未対応。公式 checkpoint は ESPnet の `separator.` 接頭辞付き state dict、パスは明示指定。チェックポイント自体の利用条件は未確認。長時間音声と完全な公式 CLI との比較は未実施。

## ライセンスと依存（調査結果）

- SEMamba++: 参照 commit にライセンスファイルはない。[重みカード](https://huggingface.co/yongjoonlee/semambapp) は GPL-2.0。ソースはコピーせず、チェックポイント互換の再実装にした。`mamba_ssm` は使っていない。
- MossFormerGAN: コードと[重みカード](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K) は Apache-2.0。推論は SyncANet のみ。
- DeepFilterNet3: コードは MIT または Apache-2.0。DSP は `libdf` なしの PyTorch 実装。
- UniPASE: コードは MIT。同梱の第三者コードには個別ライセンスがある。[重みカード](https://huggingface.co/Xiaobin-Rong/unipase) は Apache-2.0。
- StuPASE: PASE は Apache-2.0 と追加の個別ライセンス。推論既定は steps=8、CFG=0.5、sway=-1.0。Euler と rotary は同梱の代替実装。
