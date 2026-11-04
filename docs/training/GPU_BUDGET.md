# 本格学習の GPU 予算（2026-11-04）

100 optimizer step の実測から外挿した。長時間学習は開始していない。A100 の旧見積もりは [GPU_COST.md](GPU_COST.md)。ここでの秒は、そのファイルの秒へ換算していない。

## 測定環境

| 項目 | 値 |
|---|---|
| 日時 | 2026-11-04 |
| ジョブ | 既存の対話ジョブ `115437`。ホスト `r4n11`。新しい `qsub` / `qrsh` は出していない |
| GPU | NVIDIA H100 MIG 4g.47gb。PyTorch から見た容量は 46.38 GiB（`total_bytes` 49,794,777,088）。compute capability 9.0。同時に使った MIG は 1 |
| ソフトウェア | `.venvs/clean-20261104`。PyTorch 2.6.0+cu124。`deterministic: false` なので TF32 は許可される。Trainer は cuDNN benchmark を切ったまま |
| データ | `data/db/voicebank.duckdb` の `voicebank_28_train` 10,700 件、`voicebank_28_valid` 872 件。本番 `resource.duckdb` は開いていない |
| 音声時間 | clean ファイルを全件 `soundfile.info` した。train 8.708 時間、valid 0.684 時間。ファイルは 48 kHz。学習はモデルごとのサンプルレートへ読み込み時に変換し、segment へ切る |
| 話者 | train は p226 から p282 の 26 話者。valid は p286 と p287。話者と発話の重なりは 0。公式 test 話者はこの DB に無い |
| seed | 17 |
| 精度 | fp32。AMP は使っていない。ESPnet の TF-GridNet レシピも `use_amp: false` |
| DataLoader workers | 独立 Trainer は 0。step 時間はプロセス内のレシピ読み込みを含む。公式の worker 4、8、32 とは違う |
| step | 1 step は 1 回の Optimizer 更新。蓄積の micro-batch は step に数えない。CUDA は計測の区切りで同期した |

結果は `data/results/training/trial100-20261104/`。各モデルの `measurement.json` と `steps.jsonl`。既存の結果ディレクトリは上書きしていない。SEMamba++ の batch 8 失敗は `semambapp/` に残し、成功は `semambapp-accum2/` である。

## 100 step の実測

定常 step は 6 歩目以降 95 歩の分布である。1 歩目はカーネル初期化を含むので定常に入れない。Peak は step ごとの allocated / reserved の最大である。

| モデル | Optimizer 更新 | 定常 秒/step 中央値 | p90 | 1 歩目 秒 | Peak allocated | Peak reserved | 非有限 | 検証 8 件の外挿 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| MP-SENet | 100 | 0.611 | 0.842 | 5.91 | 31.34 GiB | 34.01 GiB | 0 | 484 秒 |
| SEMamba++ | 100 | 4.709 | 5.041 | 15.31 | 25.83 GiB | 27.67 GiB | 0 | 604 秒 |
| TF-GridNet | 100 | 0.462 | 0.554 | 1.30 | 4.98 GiB | 5.76 GiB | 0 | 123 秒 |
| DeepFilterNet3 | 100 | 2.153 | 2.332 | 5.72 | 2.84 GiB | 3.31 GiB | 0 | 213 秒 |
| MossFormerGAN | 100 | 0.769 | 0.809 | 4.01 | 11.29 GiB | 11.73 GiB | 0 | 258 秒 |

SEMamba++ の公式 micro 8 は 1 歩目の discriminator で OOM した。確保のピークは 45.56 GiB。成功した 100 step は global batch 8 を micro 4 × 蓄積 2 で保つ。Generator と Discriminator は、その 1 step の中で各 1 回更新される。蓄積 2 は Discriminator のあと Generator を forward し直す。

MossFormerGAN も 1 step で Generator を 1 回、Discriminator を 1 回更新する。micro-batch は 2。100 step は Optimizer 更新 100 回であり、micro-batch 200 回ではない。

損失はいずれも有限である。100 step の値を強調品質とはみなさない。TF-GridNet の `mr_l1_tfd` は既存の損失のスケールのままである。

検証の外挿は、登録順の先頭 8 件を 1 件ずつ測り、872 件へ比例させた。発話長で層化していない。MP-SENet の最初の全体検証は、長さ 29,083 が hop 100 の iSTFT 長 29,000 と一致せず失敗した。検証経路だけ hop 倍数へ切るようにしたあとの再測定が上の 484 秒である。更新経路の長さ検査は変えていない。

Checkpoint 保存は 2 回測った。中央に近い値は MP-SENet 0.11 秒、SEMamba++ 0.42 秒、TF-GridNet 0.14 秒、DeepFilterNet3 0.09 秒、MossFormerGAN 0.42 秒。

再開は step 50 の checkpoint を別プロセスで読み、step 52 まで進めた。5 モデルとも、続けた入力の case ID は連続実行の step 51 と 52 と一致した。読んだ記録は global step 50、epoch 0、消費済み micro-batch（蓄積 2 のモデルは 100）、RNG の種類 `python` / `numpy` / `torch` / `cuda` である。更新前テンソルのハッシュをこの試験で再比較してはいない。更新後のビット一致は求めていない。

モデル初期化は約 1.0–1.3 秒、データセットの索引は約 1.5 秒である。プロセス全体には Python の import が加わる。長時間ジョブではこの固定分は epoch 時間に対して小さい。

## 1 epoch の式

独立 Trainer の 1 epoch は、train 10,700 件を 1 回走査し、端数の micro-batch を捨てた Optimizer 更新である。

`epoch_seconds = steps_per_epoch × step_seconds + validation_seconds + checkpoint_seconds`

`step_seconds` の標準は定常中央値、楽観はその 0.95 倍、保守は定常 p90 である。検証の楽観は外挿の 0.75 倍、保守は 1.5 倍である。この幅は 8 件からの外挿用であり、新しい実測ではない。1 歩目の初期化は epoch ごとには足さない。ジョブ開始時に 1 回だけ乗る。

| モデル | steps / epoch | 捨てる発話 | 標準 秒/epoch | 標準 時間/epoch |
|---|---:|---:|---:|---:|
| MP-SENet | 2,675 | 0 | 2,118 | 0.588 h |
| SEMamba++ | 1,337 | 4 | 6,900 | 1.917 h |
| TF-GridNet（1 走査） | 2,675 | 0 | 1,360 | 0.378 h |
| TF-GridNet（公式 5,000 iter） | 5,000 | 走査ではない | 2,436 | 0.677 h |
| DeepFilterNet3 batch 16 | 668 | 12 | 1,651 | 0.459 h |
| MossFormerGAN | 5,350 | 0 | 4,371 | 1.214 h |

TF-GridNet の公開レシピは `num_iters_per_epoch: 5000` であり、10,700 件の 1 走査ではない。公式再現の時間は 5,000 iter の行を使う。公平比較で 1 走査を 1 epoch と呼ぶ場合は別行である。

DeepFilterNet3 の出荷 batch は epoch とともに 16、24、32、64、128 へ増える、と [GPU_COST.md](GPU_COST.md) が記録している。今回測ったのは epoch 0 の batch 16 だけである。`config.ini` の `MAX_EPOCHS` はこのセッションで読み直していない。コードの既定は 10、公開重みのファイル名は `model_120` なので、120 は候補であって確定値ではない。batch 16 のまま 120 epoch 続けると標準 55.1 時間だが、それは出荷スケジュールではない。サンプル数が同じなら epoch 時間は batch に比例しない、という仮定を置くと、未測定の大きい batch でも 1 epoch は 0.459 時間の近くに留まる。これは仮定である。保守では epoch 2 以降を測定値の 2 倍にし、120 epoch で約 109 時間とする。

## 公式の epoch 数での総時間

標準シナリオ。1 seed。DeepFilterNet3 は測定できた epoch 0–1 の 2 epoch だけを合計に入れ、120 epoch は脚注に分ける。TF-GridNet は公式の 5,000 iter × 35 epoch。

| モデル | epoch 数 | 標準 | 楽観 | 保守 |
|---|---:|---:|---:|---:|
| MP-SENet | 400 | 235.4 h | 212.9 h | 330.9 h |
| SEMamba++ | 100 | 191.7 h | 178.7 h | 212.4 h |
| TF-GridNet | 35 × 5,000 iter | 23.7 h | 22.3 h | 28.7 h |
| DeepFilterNet3 | 2（batch 16 のみ） | 0.92 h | 0.85 h | 1.04 h |
| MossFormerGAN | 120 | 145.7 h | 136.7 h | 157.2 h |
| 合計（DF は 2 epoch） |  | 597.4 h | 551.4 h | 730.2 h |

1 seed の標準 597 時間は 24.9 日を 1 GPU で直列に走らせた経過時間である。GPU 時間も同じ 24.9 GPU-days である。3 seed はその 3 倍で、標準 1,792 時間（74.7 GPU-days）、楽観 1,654 時間、保守 2,191 時間。

DeepFilterNet3 を batch 16 のまま 120 epoch 足すと、1 seed の標準はさらに 55.1 時間で合計 652 時間になる。2 倍の保守を 118 epoch にかけると、DF だけで約 109 時間である。3 seed の標準にこの 55.1 時間を足すと 1,957 時間である。

早期停止は測っていない。検証が epoch 40 以降 15 epoch 改善しない、という計画をそのまま時間にすると、MP-SENet は 400 ではなく 55 epoch 前後で、標準約 32 時間になる。SEMamba++ は 100 のうち 55 なら約 105 時間、MossFormerGAN は約 67 時間である。これはシナリオであり、止まる epoch の実測ではない。公式再現では各モデルの公開 epoch 数を優先する。

## 構成

| 構成 | 内容 | 標準の GPU 時間 |
|---|---|---:|
| 最小 | 5 モデル × 1 seed。DF は batch 16 の 2 epoch だけ確定 | 597 h |
| 標準 | 5 モデル × 3 seed。同じ DF の扱い | 1,792 h |
| 拡張 | 標準に、DF を batch 16 の速度で 120 epoch まで足し、各モデル 1 回分のやり直し（約 20%）を見る | 約 2,350 h |

拡張の 20% は測定ではなく、失敗したジョブのやり直し枠である。負の SNR 評価セットの作成は CPU である。固定 ASR の WER は未実測なので、この時間に入っていない。Whisper Large v3 を test 824 件 ×（clean、noisy、5 モデル）へ通す時間は、学習予算とは別に pilot してから積む。

フル H100 や A100 への速度換算はしない。MIG 4g.47gb の測定を、メモリが広い装置でも同じ秒数とは書かない。SEMamba++ の batch 8 が載る GPU では、蓄積 2 より短くなる可能性がある。その秒数は無い。

## メモリと GPU の置き方

1 モデルにつき GPU は 1 枚を想定する。同時実行は測っていない。

| モデル | この MIG に載るか | 備考 |
|---|---|---|
| MP-SENet | 載る。reserved 34 GiB | 他の学習と重ねない |
| SEMamba++ 蓄積 2 | 載る。reserved 28 GiB | micro 8 は載らない |
| TF-GridNet | 載る。reserved 6 GiB | |
| DeepFilterNet3 batch 16 | 載る。reserved 3 GiB | batch 64 以上は未測定 |
| MossFormerGAN | 載る。reserved 12 GiB | |

TF-GridNet、DeepFilterNet3、MossFormerGAN の reserved 合計は約 21 GiB で、46 GiB の MIG に数値上は並ぶ。同時実行の速度とピークは測っていないので、予算は 1 枚ずつとした時間のままにする。

## ジョブ時間と実験プロセス時間

上の時間は、学習プロセスが GPU を使っている計算の外挿である。TSUBAME がジョブに確保する時間、課金ポイント、対話ジョブ `115437` の予約とは別である。30 分を超える本学習は、投入の前にそのジョブを含む明示の承認が要る。この文書は投入の承認ではない。

実験プロセス時間の会計は [VERIFICATION.md](VERIFICATION.md) の 2026-11-04 の節。上限 25,200 秒は課金ポイントではない。
