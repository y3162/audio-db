# TF-Locoformer

DNS-Interspeech 2020 の単一話者強調（`dns`）だけを対象にする。2 話者分離は読み込まない。公式参照は `merlresearch/tf-locoformer` tag `v1.1.0`、commit `7a615460d347ff7334a13dbb831d16280da72cdc`。再実行は `scripts/se/verify_tf_locoformer.py`。通常の `unittest` は公開重みを読まない。公式 checkout または公開重みが無いと比較行は SKIP であり、SKIP は PASS ではない。

事前に固定した波形許容は、最大絶対誤差 `1e-5` 以下かつ相対 L2 `1e-5` 以下。公開重みの 1 秒 GPU 差として既に記録されていた `1.11e-9` より広く、フロントエンドや重みの取り違えは落ちる幅である。

## 推論（2026-11-05、turing）

環境は `.venvs/se`、PyTorch `2.6.0+cu124`、torchaudio `2.6.0+cu124`、`rotary-embedding-torch==0.8.3`。チェックポイント `data/checkpoints/tf-locoformer/dns/valid.loss.ave_5best.pth` の SHA-256 は `6ffa518f2b73bed289e38349366eb777c94b9130acc32bf4c085cb049d9d1666`。seed は `20261105`。公式 standalone separator と独立実装に同じ state dict を読み、同じ 16 kHz 波形を入れた。

CPU（turing）は実音声・合成 1/5/20 秒・8 kHz と 48 kHz から 16 kHz へリサンプルした 1 秒のすべてで最大絶対誤差 0。20 秒の経過時間は 175 秒（RTF 8.75）。

GPU は空の Quadro RTX 5000（16 GB、使用メモリ 15 MiB の装置）。最大絶対誤差は次のとおり。

| 入力 | 長さ | 最大絶対誤差 | 相対 L2 | 秒 | peak allocated |
|---|---:|---:|---:|---:|---:|
| libri_demand_snr0 先頭の clean | 32,000 | 5.96e-8 | 8.21e-8 | 1.54 | 404 MB |
| 同じケースの noisy | 32,000 | 4.47e-8 | 7.97e-8 | 0.41 | 404 MB |
| 合成 1 秒 | 16,000 | 1.49e-8 | 1.56e-7 | 0.21 | 272 MB |
| 合成 5 秒 | 80,000 | 2.98e-8 | 1.89e-7 | 1.14 | 801 MB |
| 合成 10 秒 | 160,000 | 2.24e-8 | 3.69e-7 | 2.36 | 1.46 GB |
| 合成 20 秒 | 320,000 | 2.24e-8 | 4.21e-7 | 5.48 | 2.79 GB |
| 8 kHz から 16 kHz | 16,000 | 1.44e-8 | 2.91e-7 | 0.21 | 271 MB |
| 48 kHz から 16 kHz | 16,000 | 8.15e-9 | 4.04e-7 | 0.21 | 271 MB |

10 秒行は別プロセス（`report-turing-10s.json`）。他の GPU 行は同じ許容で PASS した最初の測定。20 秒推論の peak は 2.79 GB で、16 GB の推論は完走する。チャンク分割は足していない。

## 学習アルゴリズム

公式 DNS レシピと研究設定の対応は次のとおり。

| 項目 | 公式 DNS | 研究 YAML | 判定 |
|---|---|---|---|
| ネットワーク | standalone、`normalize_variance: true` | 同じ | GO |
| 損失 | `mr_l1_tfd`、時間重み 0.5、損失内の分散正規化なし、SI-SNR 重み 0 | 同じ | GO |
| Optimizer | AdamW、lr 1e-3、betas `(0.9, 0.999)`、weight decay 1e-2、clip 5 | 同じ | GO |
| 初期化 | `xavier_uniform` | `model.init: xavier_uniform`。省略時は PyTorch 既定で、スモーク YAML はこちら | GO。研究ファイルは明示 |
| scheduler | warmup 4,000 step、その後 factor 0.5、patience 3 の plateau | `warmup_reduce_on_plateau`。更新ごとに線形 warmup、epoch 末の validation loss で plateau | GO。式は ESPnet `WarmupReduceLROnPlateau` |
| 最初の更新の lr | 構築時の 1e-3。scheduler は `optimizer.step` のあと | 同じ | GO |
| チャンク | 64,000 サンプル、shift 0.5 | 32,000 サンプル | intentional。VoiceBank 16 kHz では 10,700 件中 1,205 件だけが 64,000 より長い |
| バッチ | 4 GPU で各 forward batch 4、蓄積 1 | 1 GPU、forward batch 4、蓄積 1 | intentional。複数 GPU には分けない |
| epoch | `num_iters_per_epoch: 1000`、最大 150、early stop patience 10 | 発話 10,700、最大 150、patience 10 | intentional。データが DNS ではない |
| 入力音量 | `speech_volume_normalize: null` | `data.rms_normalize: true` | intentional。他の VoiceBank 研究設定に合わせ、スモークも true を明示 |
| モデル内正規化 | `normalize_variance: true` | true | GO。発話 RMS とは別 |
| データ | DNS-Interspeech 2020 | `voicebank_28_train` / `voicebank_28_valid` | intentional |
| best の SI-SNR | ESPnet は SI-SNR を最大化 | validation の `si_snr` は正の SI-SDR（高い方が良い） | GO。学習ログの `si_sdr` は負の損失のまま |
| 決定性 | 保存された DNS 設定は `cudnn_deterministic: true` | 研究 YAML は `deterministic: false` | intentional。他の VoiceBank 長時間設定と同じ |

スモークは `configs/training/tf_locoformer.yaml`（CPU、2 更新、1 秒、scheduler なし、PyTorch 既定初期化）。研究設定は `configs/training/tf_locoformer_voicebank.yaml`。150 epoch の投入はこの測定だけでは始めない。

ゼロ分散の混合は学習 forward が `ValueError` を出し、タスクが `TrainingError` にするため optimizer は進まない。推論は不偏標準偏差が `1e-6` 未満のとき零波形を返す。通常振幅の公式一致はこのしきい値の外である。定数 `0.2` の float32 標準偏差は 0 ではなく約 `1.5e-8` なので、厳密な 0 比較では抜けていた。

縮小モデルの CPU 検証:

- Xavier 初期化は、同じ seed で公式 separator に同じ手続きを適用した結果と差 0。
- warmup の lr は step 1..4 で `base * step / 4`、step 5 で base、その後 patience を超えるまで据え置き、次の悪い validation で 0.5 倍。
- 同じ初期重み・同じ 5 バッチで、公式 separator と独立タスクの損失・lr・更新後重みの差は 0。
- 連続 20 更新と、10 更新で保存して再開した 20 更新の重み・AdamW state・lr・scheduler step の差は 0。

公開幅の 1 更新は tensor の空いている A100 80GB（物理 GPU 4、測定開始時 13 MiB）で行った。合成波形、Xavier、AdamW、勾配クリップ 5。

| チャンク | batch | peak allocated | 秒 | 損失 |
|---|---:|---:|---:|---:|
| 64,000 | 1 | 16.24 GB | 12.43 | 有限 |
| 64,000 | 4 | 64.52 GB | 3.55 | 有限 |
| 32,000 | 1 | 8.18 GB | 1.13 | 有限 |
| 32,000 | 4 | 32.48 GB | 0.63 | 有限 |

turing の 16 GB では 64,000・batch 1 の backward が OOM した。研究 YAML の 32,000・batch 4 は peak allocated 32.48 GB なので、測定した 80 GB では載る。40 GB カードでは未測定。seed 17 の epoch 1 と epoch 2 はどちらも 3,834 optimizer 更新だった。0.63 秒はデータ読込を含まない 2 回目の合成更新なので、1 epoch の計算下限はおおよそ 3,834 × 0.63 秒（約 40 分）である。読込、validation、checkpoint は含まない。

## 評価

公開 DNS 重み、Whisper large-v3（`temperature=0`、language en）、`.venvs/asr`、turing の空いている Quadro RTX 5000（GPU 0、開始時 15 MiB）、seed 17。学習セットは VoiceBank で、ここでの発話は LibriSpeech test-clean なので学習には入っていない。

`libri_demand_snr0` は 2 件すべてを処理した（error 0、skip 0）。音声は短縮されているため WER は `partial_audio` で na であり、0 にはしていない。PESQ は noisy 1.316 から enhanced 3.950、STOI は 0.964 から 0.994、SI-SDR は -0.009 から 22.94。保存先は `data/results/se-evaluation/tf-locoformer-dns-libri-snr0-20261105/`。

`libri_testclean_full_smoke` は全文の 8 件（error 0、skip 0）。micro WER（`wer-norm-v1`）は clean / noisy / enhanced とも 0.015748（正規化トークン 127 に対し置換 2）。観測波形は参照と一致し、noisy の SI-SDR は数値的に無限大なので na、STOI は 1、PESQ は 4.644 である。強調後の PESQ は 4.521、STOI は 0.999983、SI-SDR は 39.54、WER は変わらない。WER が動いていないことは、公開重みの波形一致とは別の結果である。保存先は `data/results/se-evaluation/tf-locoformer-dns-libri-full-20261105/`。
