# 実装状況（引き継ぎ用）

判定は、公式チェックポイントを載せた公式処理と独立実装の最終波形を同じ入力で比べた結果に基づく。短音声の有限値だけでは PASS にしない。

この表は turing 上の波形比較の記録である。現在の通常テストは `.venvs/se`、公式パッケージ比較は `.venvs/official`、SEMamba++ の fused CUDA は `.venvs/semambapp` を使う。

| モデル | 公式波形の最大絶対誤差 | 最大検証長 | official非依存 | 判定 |
|---|---|---|---|---|
| TF-GridNet urgent24 | 0 | 20 秒 | 確認 | PASS |
| TF-GridNet ms-snsd | 0 | 20 秒 | 確認 | PASS |
| SEMamba++ | CUDA 融合カーネルとの波形差は 1 秒 3.97e-5、20 秒 1.80e-5 | 20 秒 | 確認 | PASS |
| MP-SENet VB | CPU で 0、GPU 同士 3.73e-9 | 20 秒 | 確認 | PASS |
| MP-SENet DNS | CPU で 0、GPU 同士 2.33e-10 | 20 秒 | 確認 | PASS |
| MossFormerGAN | CPU 20 秒で 0。GPU は 15/20/25 秒で最大 2.33e-6 | 25 秒 | 確認 | PASS |
| DeepFilterNet3 | 48 kHz・20 秒で 6.52e-8。16 kHz API 往復は 3.86e-6 | 20 秒 | 確認 | PASS |

全体の L2 は GO。根拠と未検証範囲は `docs/se/VALIDATION.md`。

## 修正

- MossFormerGAN は 10 秒を超えると公式が 10 秒窓・ストライド 75% で分割する。独立実装は全文を一度に見ていたため、11 秒で最大絶対誤差 0.151 だった。同じ窓と、発話全体で一度だけ計算するエネルギー正規化を入れたあと、11 秒は最大絶対誤差 0。1 秒（分割なし）も 0 のまま。
- `n_fft` はチェックポイントの周波数次元 201 と推論 YAML の `fft_len: 400` が一致する。argparse の既定 512 は、この重みの推論では使われない。

## 実行

```bash
cd /path/to/audio-db
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/se -v
```

公式比較は `tests/se/reference/run_*.py` を明示実行する。`.venvs/official` が要るスクリプトは `.venvs/se` では走らない。SKIP を PASS とは数えない。

## 今回足したこと

- SEMamba++ の独立実装は PyTorch 漸化式のまま。公式 CUDA 融合カーネルとの差は `VALIDATION.md`。ライブラリは `mamba_ssm` を import しない。

## 未検証のまま残したもの

- TF-GridNet の ESPnet 推論コマンド自体は起動していない。比較は公式 separator と、ESPnet `Stft` が呼ぶ `torch.stft` 条件。
- MP-SENet の公式 CLI のファイル入出力と PCM 16 書き出し。波形推論とは別。CPU/GPU 差は公式同士でも出る（VB 最大 4.16e-3、DNS 最大 1.29e-4）。
- DeepFilterNet3 の 16 kHz 入力は torchaudio のリサンプル。`libdf` の sinc ではない。往復後の差は 3.86e-6。
- SEMamba++ を融合カーネルへ差し替える最適化はしていない。20 秒の RTF は公式 0.039、独立実装 0.33。Peak VRAM は公式 1.3 GiB、独立実装 6.5 GiB。
