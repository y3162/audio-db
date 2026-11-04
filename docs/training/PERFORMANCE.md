> 訂正：DeepFilterNet3 の schedule は `min(schedule値, 設定 batch_size)`。出荷 batch_size 64 では epoch 10 以降の実効 batch は 64 であり、9999 をそのまま投入しない。batch 64 の実行可否は未測定。公式 `df/train.py:236`。

> 訂正（2026-11-04）：TF-GridNet の 5,000 更新は固定更新の研究案であり、ESPnet chunk iterator の公式 epoch ではない。このファイルの 0.191 秒/step は長さ 48,000 の固定 crop の測定で、可変チャンク epoch の合計時間には換算しない。詳細は [監査](OFFICIAL_TRAINING_AUDIT.md)。

# 学習 step の内訳と入力を変えない高速化（2026-11-04）

モデル構造、batch、segment、損失、Optimizer、STFT、selective scan は変えていない。変えたのは発話の並列読み込み、次の Optimizer 更新分の先読み、4 行以上の wideband PESQ のプロセス並列である。`torch.compile`、CUDA Graph、AMP、TF32 の切り替えは入れていない。

## 計測環境

| 項目 | 値 |
|---|---|
| ホスト | `r4n11`。対話ジョブ `115720`（codeserver）。新しい `qsub` / `qrsh` は出していない |
| GPU | NVIDIA H100 MIG 4g.47gb。PyTorch の容量は 46.38 GiB（`total_bytes` 49,794,777,088）。compute capability 9.0。`CUDA_VISIBLE_DEVICES` はその MIG UUID |
| 使用率 | `nvidia-smi` の GPU Utilization は `[N/A]`。0% ではない。親 GPU の `memory.used` は他の MIG 切片を含むことがあり、このセッションの開始時は 18,139 MiB、高速化の再測定前は 4,596 MiB だった。計算プロセスは無く、見えている MIG 切片は空だった。親メモリが 1,024 MiB 以上だと既存の空き検査が拒否するため、測定プロセスだけ `allow_busy` にした。これは他の利用者の GPU を使う許可ではない |
| ソフトウェア | `.venvs/clean-20261104`。PyTorch 2.6.0+cu124。`deterministic: false`。cuDNN benchmark は切ったまま |
| データ | `data/db/voicebank.duckdb` の `voicebank_28_train`。seed 17 |
| 最適化前 | `data/results/training/trial100-20261104/`。読み込みは直列で、先読みは無い |
| 最適化後 | `data/results/training/perf-fast-20261104/` |
| 同日の直列対照 | `data/results/training/perf-serial-20261104/`。20 step、`load_workers: 1`、`prefetch: false` |
| Profiler | `data/results/training/perf-profile-20261104/`。最後の 2 step だけ PyTorch Profiler。その step 時間はオーバーヘッドを含む |

DCGM は使っていない。カーネル時間は Profiler、通常の step 時間は Profiler を外した計測である。Profiler ありの step 中央値は TF-GridNet 7.4 秒、MP-SENet 2.2 秒、MossFormerGAN 17 秒、DeepFilterNet3 32 秒、SEMamba++ 19 秒で、通常 step の比較には使わない。

1 step は 1 回の Optimizer 更新である。定常は 6 step 目以降。CUDA は区切りで同期した。Peak は step ごとの allocated / reserved の最大である。

## 最適化前の内訳

朝の 100 step（`trial100-20261104`）の定常中央値。collate はレシピの読み込み、混合、リサンプル、切り出しである。

| モデル | 秒/step | collate | update | 主因 |
|---|---:|---:|---:|---|
| MP-SENet | 0.611 | 0.028 | 0.582 | GRU backward、bmm、softmax。PESQ は update の中で GPU を待たせる |
| SEMamba++ | 4.709 | 3.010 | 1.703 | 直列の音声読み込み。GPU は convolution backward と selective scan |
| TF-GridNet | 0.462 | 0.274 | 0.188 | 読み込みと LSTM / im2col。GPU 更新は 0.19 秒で安定 |
| DeepFilterNet3 | 2.153 | 0.962 | 1.196 | 48 kHz・batch 16 の読み込みと、スペクトル損失の GPU 計算 |
| MossFormerGAN | 0.769 | 0.014 | 0.755 | convolution backward と 2 発話の PESQ。読み込みは中央値では小さい |

メモリ使用量が小さいことは、それだけでは遅さの理由にしない。DeepFilterNet3 の reserved は約 3 GiB だが、step は読み込みと GPU の両方で長い。

同日に直列のまま 20 step を再実行すると、同じ先頭ファイルでも collate が再び長かった。SEMamba++ の collate 中央値 2.61 秒、update 1.70 秒。DeepFilterNet3 は collate 2.69 秒、update 1.53 秒。TF-GridNet は collate 0.95 秒、update 0.19 秒。ページキャッシュだけで朝の 3 秒が消えた、とは言えない。学習プロセスの直列読み込みが step の外側に残る。

## 採用した変更

1. `RecipeDataset.collate` は `load_workers > 1` のとき、発話をスレッドで読む。既定は 4。`Executor.map` が行順を保つ。
2. Trainer は次の Optimizer 更新 1 回分を、現在の GPU 更新と重ねて読む。epoch を進める前にその読み込みを待つ。
3. wideband PESQ は 4 行以上を spawn プロセス 4 つで計算する。1〜3 行は従来どおり直列。スコアは同じ `pesq(..., "wb")` で、行順は変わらない。

不採用にしたもの:

| 候補 | 理由 |
|---|---|
| `torch.compile`、CUDA Graph、AMP、TF32 の変更 | 数値経路が変わる。今回は調査のみで、学習実装には入れない |
| batch の変更 | 公式の batch を保つ。DeepFilterNet3 の batch 24 は予算計測であり、batch 16 の学習設定は変えていない |
| selective scan や STFT の別実装 | 以前の backward 差の原因になった経路を維持する |
| PESQ のスレッド並列 | GIL の内側で、4 行の 0.072 秒が 0.075 秒になった。プロセスプールは 0.023 秒だったので、こちらだけ採用した |
| pin_memory による転送の重ね | 1 step の波形は数 MB から十数 MB で、読み込みと GPU カーネルより小さい |

## 入力と学習写像

CPU の玩具データでは、workers 4 の波形が直列と `torch.equal` で一致し、prefetch ありの 3 step と再開が直列実行の重み・Optimizer と一致した。4 行の PESQ は直列スコアと一致した。

VoiceBank の TF-GridNet step 2 の 4 発話は、直列と workers 4 で波形ハッシュが一致した（`64fa0568f1de8891`）。

最適化後の 100 step は、5 モデルとも朝の実行と case ID の列が一致した。step 1 の generator 損失は 5 モデルとも朝の値と一致した。step 2 以降は `deterministic: false` の CUDA 再実行でずれる。TF-GridNet の step 2 の相対差は 3.9e-6 で、同日の直列再実行も step 1 から 95173.195 と 95173.188 に分かれた。この差を先読みが作った新しい系統差とは扱わない。公式 4 プロセスの再実行は、この高速化の前後ではやり直していない。

step 50 の checkpoint を別プロセスで読み、step 52 まで進めた。5 モデルとも step 51 と 52 の case ID は連続実行と一致した。読んだ記録は global step 50、epoch 0、消費済み micro-batch、RNG の `python` / `numpy` / `torch` / `cuda` である。更新後の重みハッシュの一致は求めていない。

## 最適化後の 100 step

定常中央値。VRAM は朝の実行と同じバイト数である。

| Model | Before sec/step | After sec/step | Speedup | Before VRAM | After VRAM |
|---|---:|---:|---:|---:|---:|
| MP-SENet | 0.611 | 0.543 | 1.12 | 31.34 GiB | 31.34 GiB |
| SEMamba++ | 4.709 | 1.671 | 2.82 | 25.83 GiB | 25.83 GiB |
| TF-GridNet | 0.462 | 0.191 | 2.42 | 4.98 GiB | 4.98 GiB |
| DeepFilterNet3 | 2.153 | 1.421 | 1.52 | 2.84 GiB | 2.84 GiB |
| MossFormerGAN | 0.769 | 0.757 | 1.02 | 11.29 GiB | 11.29 GiB |

100 step の学習 wall clock（検証の外挿を含まない）:

| モデル | Before | After |
|---|---:|---:|
| MP-SENet | 82.71 秒 | 62.91 秒 |
| SEMamba++ | 463.98 秒 | 177.25 秒 |
| TF-GridNet | 49.17 秒 | 23.93 秒 |
| DeepFilterNet3 | 203.58 秒 | 145.12 秒 |
| MossFormerGAN | 88.98 秒 | 86.77 秒 |
| 合計 | 888.43 秒 | 495.98 秒 |

プロセス全体（import、初期化、8 件の検証を含む）は朝の記録と `measurement.json` の `process_seconds` を比べる。SEMamba++ は 475.5 秒から 183.0 秒、TF-GridNet は 53.2 秒から 28.3 秒、DeepFilterNet3 は 208.5 秒から 150.8 秒になった。

MossFormerGAN の 1.02 倍は、朝の定常分布（最小 0.759、p90 0.809）の中に入る。中央値だけでは効果があったと言わない。同日の直列対照では collate が 0.59 秒まで伸び、step 中央値は 1.35 秒だった。先読み後の p90 は 0.762 秒で、GPU 更新 0.76 秒に張り付く。読み込みが遅いときだけ効く。

MP-SENet の update 中央値は 0.582 秒から 0.541 秒になった。PESQ プールの単体計測は 4 行で 0.073 秒から 0.023 秒である。残りの短縮は、時々入る読み込み待ちが先読みで消えたことによる。p90 は 0.842 秒から 0.556 秒になった。

SEMamba++、TF-GridNet、DeepFilterNet3 は、同日直列対照の collate が 0.95〜2.69 秒で、先読み後の step は GPU 更新の時間まで落ちた。この 3 つが全学習時間の短縮のほとんどを占める。

検証 8 件からの全件外挿も短くなった（MP-SENet 484 秒から 263 秒、SEMamba++ 604 秒から 176 秒、TF-GridNet 123 秒から 56 秒）。発話長で層化していない。全学習時間へ足すときは、遅い方の外挿を保守側に残す。詳細は [GPU_BUDGET.md](GPU_BUDGET.md)。

同じ日の追加で、TF-GridNet の chunk 48,000・batch 4 を 24 step 測り直した。定常中央値は 0.1916 秒で、上の 0.191 秒と一致した。5,000 更新と検証 872 件を含むプロセスは 1,045.70 秒だった。step 時間の中央値だけでは、この 1 epoch の実測にならない。

## DeepFilterNet3 batch 24

出荷スケジュール `0/16,2/24,5/32,10/64,20/128,40/9999` の epoch 2–4 が batch 24 である。`steps_per_epoch` は `floor(10700/24)=445`。測定は step 0 から 40 step で、epoch 2 の学習率そのものではない。計算時間を見るための実行である。設定は `configs/training/measure/deepfilternet3-batch24.yaml`。学習の batch 16 設定は変えていない。

| | batch 16（先読み後） | batch 24 |
|---|---:|---:|
| 定常 秒/step | 1.421 | 2.200 |
| p90 | 1.516 | 2.270 |
| Peak allocated | 2.84 GiB | 4.21 GiB |
| Peak reserved | 3.31 GiB | 4.89 GiB |
| step / epoch | 668 | 445 |
| 更新だけの epoch | 949 秒 | 979 秒 |

OOM はしなかった。1 発話あたりの時間は 0.089 秒と 0.092 秒で、batch を増やしても epoch の計算は短くならない。スケジュール末尾の batch 9999 は実効 64 であり、9999 のままは投入しない。

## DeepFilterNet3 batch 32 と 64

`configs/training/measure/deepfilternet3-batch32.yaml` と `deepfilternet3-batch64.yaml`。各 8 step の予定で、定常は後半 5 点。先読みが collate の印を消すため、秒/step は update と step の中央値を使う。collate の記録は 0 秒で、読み込みは更新と重なっている。

| | batch 32 | batch 64 |
|---|---:|---:|
| 定常 秒/step | 3.471 | 7.025 |
| p90 | 3.525 | 7.061 |
| Peak allocated | 5,991,804,416 バイト（5.58 GiB） | 11,841,566,208 バイト（11.03 GiB） |
| Peak reserved | 6,973,030,400 バイト（6.49 GiB） | 13,832,814,592 バイト（12.88 GiB） |
| 更新 / epoch | 334 | 167 |
| 更新だけの epoch | 0.322 h | 0.326 h |

損失はすべて有限。OOM はしていない。batch は縮めていない。結果は `data/results/training/go-20261104/df-batch32-rerun/` と `df-batch64/`。最初の batch 32 は profiler の時刻印が欠けて step 2 で止まり、`df-batch32/` に残している。

## TF-GridNet の実長

48 kHz の WAV ヘッダフレーム数を `ceil(frames/3)` した長さは、`RecipeDataset._load` の 30 件と一致した。RMS は長さを変えない。epoch 回数は [GPU_BUDGET.md](GPU_BUDGET.md)。公式モードの 4 更新は peak allocated 5,351,750,656 バイト、peak reserved 6,182,404,096 バイトで、RMS ありの 24 step（4.98 GiB / 5.76 GiB）と同じ帯である。

## 再現

```bash
export CUDA_VISIBLE_DEVICES=<このジョブの MIG UUID>
export PYTHONPATH="$PWD"
.venvs/clean-20261104/bin/python tests/training/reference/profile_steps.py \
  --config configs/training/trial100/tf_gridnet.yaml \
  --out data/results/training/perf-profile-20261104/tf_gridnet \
  --steps 8 --profile-last 2 --allow-busy
.venvs/clean-20261104/bin/python tests/training/reference/trial100.py \
  --config configs/training/trial100/tf_gridnet.yaml \
  --checkpoint-dir data/results/training/perf-fast-20261104/tf_gridnet \
  --allow-busy
```

既存の結果ディレクトリは上書きしない。新しい `--checkpoint-dir` を使う。直列対照は `--load-workers 1 --no-prefetch`。

回帰テストは `tests/training/test_dataset.py` のスレッド読み込み、`tests/training/test_mp_senet_train.py` の prefetch と直列の重み一致、PESQ の直列一致である。
