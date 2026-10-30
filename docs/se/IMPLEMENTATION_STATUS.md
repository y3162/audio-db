# 実装状況（引き継ぎ用）

判定は次のとおり。構築・重み・短音声推論が動いても、公式出力との数値比較が無いものは公式一致を空欄のままにせず「未計測」と書き、判定は PARTIAL にする。

| モデル | 構築 | 重みロード | 推論 | 公式一致 | official非依存 | 判定 |
|---|---|---|---|---|---|---|
| UniPASE | 16 kHz 経路と PostNet 構築 | 4 ファイル | CPU 短音声、有限 | 未計測（上流クラスを移植） | 確認 | PARTIAL |
| TF-GridNet | urgent24, ms-snsd | 両 variant | CPU・CUDA | 短音声 CPU で最大絶対誤差 0。CLI 全体は未計測 | 確認 | PARTIAL |
| SEMamba++ | 公開設定の再実装 | `generator` strict | CPU 短音声、再実行差 0 | 未計測（`mamba_ssm` なし） | 確認 | PARTIAL |
| MP-SENet | VB, DNS | 両チェックポイント | CPU・CUDA | 短音声 CPU で波形差 0。CLI 全体と GPU 公式比較は未計測 | 確認 | PARTIAL |
| MossFormerGAN | SyncANet, n_fft 400 | `model` strict | CPU・CUDA | 未計測（公式モジュールは einops 未導入） | 確認 | PARTIAL |
| DeepFilterNet3 | DfNet | `model_120.ckpt.best` strict | CPU・CUDA、発話ごとに状態リセット | 未計測（`libdf` なし）。STFT 往復の最大絶対誤差 4.17e-7 | 確認 | PARTIAL |
| StuPASE | DeWavLM + CFM + vocoder | 3 ファイル。CFM は上流どおり strict=False | CPU 16 kHz。種を固定すると再実行差 0 | 未計測（Euler と rotary は代替実装） | 確認 | PARTIAL |

## 実行

```bash
cd /net/spring/work/seki/exp/audio-db
.venvs/lab/bin/python -m unittest discover -s tests/se -v
```

結果: 19 tests OK。ホストの `lab` は Python 3.13.9、PyTorch 2.6.0+cu124。CUDA スモークは `CUDA_VISIBLE_DEVICES=1`（Quadro RTX 5000）。

CPU と CUDA の最大絶対差（公式比較ではない）:

- DeepFilterNet3、48 kHz、4800 samples: 1.28e-9
- MossFormerGAN、16 kHz、1600 samples: 1.91e-6

`src/se/official` を除いた一時ディレクトリで 5 モデルのロードと短音声推論を確認した。

## 未完了

- 公式 CLI または公式拡張（`libdf`、`mamba_ssm`、`torchdiffeq`、`x-transformers`）との波形差
- 長時間音声、8/16/48 kHz の横断比較、SEMamba++ / UniPASE / StuPASE の CUDA
- MP-SENet の CPU/CUDA 差（最大約 0.00442）の原因
- TF-GridNet `wsj0-2mix`（2 話者分離のため単一強調 API の対象外）
- SEMamba++ のソース許諾。重みカードの GPL-2.0 表示はソース全体の許諾として扱っていない

## 次に必要な作業

1. `libdf` をビルドできる環境で DeepFilterNet3 の公式 `enhance` と波形を比較する。
2. `mamba_ssm` がある環境で SEMamba++ の同一入力スペクトルを比較する。
3. `rotary-embedding-torch==0.8.3` と `einops` がある環境で MossFormerGAN の公式 decode と比較する。
4. `torchdiffeq` と `x-transformers` がある環境で StuPASE のサンプル列を比較する。
5. UniPASE は 16 kHz 以外、とくに PostNet を通る 48 kHz 出力を公式スクリプトと比較する。
