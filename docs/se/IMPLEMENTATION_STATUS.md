# 実装状況（引き継ぎ用）

判定は、公式チェックポイントを載せた公式処理と独立実装の最終波形を同じ入力で比べた結果に基づく。短音声の有限値だけでは PASS にしない。

ホストは turing。独立実装の通常テストは `.venvs/lab`（Python 3.13.9、PyTorch 2.6.0+cu124）。公式パッケージの比較は `.venvs/ref`（Python 3.10.12、同じ PyTorch 2.6.0+cu124、NumPy 1.26.4）。GPU は空きの GPU 1（Quadro RTX 5000、`CUDA_VISIBLE_DEVICES=1`）。GPU 0 は他ユーザーが使用中。

| モデル | 公式ネットワーク | 公式波形 | 長時間 | official非依存 | 判定 |
|---|---|---|---|---|---|
| UniPASE | 16 kHz と 48 kHz PostNet を含む `forward` が最大絶対誤差 0 | 同左（ピーク合わせ前） | 0.5 秒と、48 kHz 指定の 16000 samples。20 秒と `inference_long` の 8 秒分割は未実施 | 確認 | PARTIAL |
| TF-GridNet urgent24 | separator と公式 STFT・分散正規化 | 8/16 kHz の 1 秒と 20 秒で 0 | 20 秒で 0 | 確認 | PASS |
| TF-GridNet ms-snsd | 公式 `TFGridNet.forward` | 16 kHz の 1 秒と 20 秒で 0 | 20 秒で 0 | 確認 | PASS |
| SEMamba++ | 公式モジュールは strict で過不足 0。参照スキャンの mixer は 0 | 1 秒波形 2.38e-8。CUDA 融合カーネルは未実施 | 1 秒のみ | 確認 | PARTIAL |
| MP-SENet VB | 公式 `mag_pha_stft` | CPU 1・5・20 秒で 0。GPU 同士は 3.73e-9 | 20 秒で 0 | 確認 | PASS |
| MP-SENet DNS | 同左 | CPU 1・5・20 秒で 0。GPU 同士は 2.33e-10 | 20 秒で 0 | 確認 | PASS |
| MossFormerGAN | `n_fft=400` の公式 decode | 1 秒と 11 秒で 0 | 11 秒（10 秒窓の境界）で 0。20 秒は未計測 | 確認 | PASS |
| DeepFilterNet3 | `libdf` の `enhance` | 1 秒 4.10e-8、20 秒 6.52e-8 | 20 秒で 6.52e-8 | 確認 | PASS |
| StuPASE | rotary 修正後の `forward` が 0 | 16 kHz・0.5 秒、seed 5 で 0 | 0.5 秒。20 秒は未実施 | 確認 | PARTIAL |

UniPASE、SEMamba++、StuPASE の長時間比較が残っているため、全体は GO ではない。

## 修正

- StuPASE の rotary は周波数を半分に切って入れ替えていた。x-transformers 2.14.2 は隣接ペア `(f0, f0, f1, f1, ...)` なので、`src/se/models/stupase/flow/compat.py` をそちらに合わせた。修正後、公式 `StuPASE.forward` と最大絶対誤差 0。
- MossFormerGAN は 10 秒を超えると公式が 10 秒窓・ストライド 75% で分割する。独立実装は全文を一度に見ていたため、11 秒で最大絶対誤差 0.151 だった。同じ窓と、発話全体で一度だけ計算するエネルギー正規化を入れたあと、11 秒は最大絶対誤差 0。1 秒（分割なし）も 0 のまま。
- `n_fft` はチェックポイントの周波数次元 201 と推論 YAML の `fft_len: 400` が一致する。argparse の既定 512 は、この重みの推論では使われない。

## CFM の strict=False

StuPASE の公式 `CFM.from_pretrained` は `strict=False`。今回の `CFM.pt` と `Vocoder_Mel-16k.pt` は、公式ローダでも独立ローダでも missing 0、unexpected 0。欠落した重みで推論しているわけではない。`strict=False` は上流のローダがそう書いているため残している。

## 実行

```bash
cd /net/spring/work/seki/exp/audio-db
.venvs/lab/bin/python -m unittest discover -s tests/se -v
```

公式比較は `tests/se/reference/run_*.py` を明示実行する。`.venvs/ref` が要るスクリプトは lab では走らない。SKIP を PASS とは数えない。

## 未完了

- UniPASE、StuPASE、SEMamba++ の 20 秒。MossFormerGAN の 20 秒（11 秒の窓境界は 0）。
- UniPASE `inference_long` の 8 秒硬切り出しと、エンハンサのピーク式（`clamp_min(1e-8)` と公式の `max+1e-8`）。
- SEMamba++ の `mamba_ssm` CUDA 融合カーネル。`causal-conv1d==1.5.0.post8` はビルド隔離で torch が見えず失敗。参照漸化式との波形差は 2.38e-8。
- TF-GridNet の ESPnet 推論コマンド自体は起動していない。比較は公式 separator と、ESPnet `Stft` が呼ぶ `torch.stft` 条件。
- MP-SENet の CPU/GPU 差は公式同士でも出る（VB 最大 4.16e-3、DNS 最大 1.29e-4）。同じ装置では公式と独立が 1e-9 以下。
- DeepFilterNet3 の 48 kHz 以外は torchaudio リサンプルであり、`libdf` の sinc とは比較していない。
