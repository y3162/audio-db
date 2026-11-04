> 訂正：DeepFilterNet3 の schedule は `min(schedule値, 設定 batch_size)`。出荷 batch_size 64 では epoch 10 以降の実効 batch は 64 であり、9999 をそのまま投入しない。batch 64 の実行可否は未測定。公式 `df/train.py:236`。

> 訂正（2026-11-04）：TF-GridNet の 5,000 更新は固定更新の研究案であり、ESPnet chunk iterator の公式 epoch ではない。下の 9.9 時間は研究案の外挿で、公式学習の見積もりには使わない。公式モードの 1 epoch の更新数は未計測。詳細は [監査](OFFICIAL_TRAINING_AUDIT.md)。

> 2026-11-04 のプロトコル実装セッションでは GPU を使っていない。見えていた GPU の使用メモリは 17,957 MiB で、1024 MiB 未満ではなかった。検証予算の加算は 0 秒。DeepFilterNet3 の batch 32 と 64 は引き続き未測定。

# 本格学習の GPU 予算（2026-11-04、高速化後）

100 optimizer step の再測定から外挿した。長時間学習は開始していない。最適化前の同じ式は、このファイルの前の版と `data/results/training/trial100-20261104/` にある。A100 の旧見積もりは [GPU_COST.md](GPU_COST.md)。秒はそこへ換算していない。内訳と採用理由は [PERFORMANCE.md](PERFORMANCE.md)。

## 測定

| 項目 | 値 |
|---|---|
| 最適化後 | `data/results/training/perf-fast-20261104/`。`load_workers: 4`、`prefetch: true`。定常は 6〜100 step の 95 点 |
| 最適化前 | `data/results/training/trial100-20261104/`。直列読み込み |
| GPU | H100 MIG 4g.47gb、46.38 GiB。同時に使った MIG は 1 |
| 環境 | `.venvs/clean-20261104`。fp32。AMP なし。`deterministic: false` |
| step | 1 step は 1 回の Optimizer 更新。中央値は標準、p90 は保守、中央値の 0.95 倍は楽観 |

Peak allocated は最適化の前後で同じバイト数だった。MP-SENet 31.34 GiB、SEMamba++ 25.83 GiB、TF-GridNet 4.98 GiB、DeepFilterNet3 batch 16 は 2.84 GiB、MossFormerGAN 11.29 GiB。

## 1 epoch

`epoch_seconds = steps_per_epoch × step_seconds + validation_seconds + checkpoint_seconds`

標準の `step_seconds` は定常中央値。検証は登録順の先頭 8 件から 872 件への比例で、発話長では層化していない。最適化後の外挿は MP-SENet 263 秒、SEMamba++ 176 秒、TF-GridNet 56 秒、DeepFilterNet3 227 秒、MossFormerGAN 245 秒である。最適化前は 484、604、123、213、258 秒だった。この差は 8 件の再測定であり、全件検証の実測ではない。保守では、最適化前と最適化後の 1.5 倍のうち遅い方を使う。

| モデル | steps / epoch | 前 秒/step | 後 秒/step | 後 時間/epoch |
|---|---:|---:|---:|---:|
| MP-SENet | 2,675 | 0.611 | 0.543 | 0.477 h |
| SEMamba++ | 1,337 | 4.709 | 1.671 | 0.670 h |
| TF-GridNet（公式 5,000 iter） | 5,000 | 0.462 | 0.191 | 0.281 h |
| TF-GridNet（1 走査） | 2,675 | 0.462 | 0.191 | 0.158 h |
| DeepFilterNet3 batch 16 | 668 | 2.153 | 1.421 | 0.327 h |
| DeepFilterNet3 batch 24 | 445 | 未測定 | 2.200 | 0.335 h |
| MossFormerGAN | 5,350 | 0.769 | 0.757 | 1.193 h |

表の「公式 5,000 iter」は固定更新の研究案であり、可変チャンクの公式 epoch ではない。DeepFilterNet3 の batch 24 は 40 step の定常 35 点で、epoch 2 の学習率そのものではない。1 epoch の計算時間は batch 16 とほぼ同じで、batch を増やしたことによる短縮は無い。

## 公開 epoch 数

標準は最適化後の step 中央値と、最適化後の検証外挿である。DeepFilterNet3 は、測った batch 16 の 2 epoch だけを合計に入れる。120 epoch は下の別表である。TF-GridNet の 9.9 時間は固定 5,000 更新 × 35 の研究案であり、可変チャンクの公式学習時間ではない。

| モデル | epoch | 前 標準 | 後 標準 | 後 楽観 | 後 保守 | 削減 |
|---|---:|---:|---:|---:|---:|---:|
| MP-SENet | 400 | 235.4 h | 190.8 h | 175.4 h | 219.2 h | 44.6 h |
| SEMamba++ | 100 | 191.7 h | 67.0 h | 62.6 h | 79.2 h | 124.7 h |
| TF-GridNet | 35 | 23.7 h | 9.9 h | 9.3 h | 10.6 h | 13.8 h |
| DeepFilterNet3 | 2（batch 16） | 0.92 h | 0.65 h | 0.60 h | 0.75 h | 0.27 h |
| MossFormerGAN | 120 | 145.7 h | 143.1 h | 134.4 h | 148.2 h | 2.6 h |
| 合計 | | 597.4 h | 411.4 h | 382.2 h | 458.0 h | 186.0 h |

1 seed の標準 411 時間は、1 GPU で直列に走らせた GPU 時間であり、経過時間も同じである。3 seed は標準 1,234 時間、楽観 1,147 時間、保守 1,374 時間。最適化前の 3 seed 標準は 1,792 時間なので、同じ範囲で約 558 時間少ない。

この 186 時間のうち、step 時間だけを差し替えて検証外挿を前の値に固定すると、約 148 時間分が step の短縮である。残りは 8 件検証の再外挿が短くなった分で、全件検証を測り直した数字ではない。保守列は遅い方の検証を使うので、その不確実性を標準より上に残している。

MossFormerGAN の 1.02 倍は朝の step 分布の中に入る。2.6 時間は効果として主張しない。表には測った中央値を置いている。

早期停止は測っていない。検証が epoch 40 以降 15 epoch 改善しない、という計画を時間にすると、MP-SENet は約 55 epoch で標準約 26 時間、SEMamba++ は約 37 時間、MossFormerGAN は約 66 時間になる。止まる epoch の実測ではない。公式再現では公開 epoch 数を優先する。

## DeepFilterNet3 のスケジュール

| 区間 | batch | 根拠 | 1 epoch | 区間の合計 |
|---|---:|---|---:|---:|
| epoch 0–1 | 16 | 100 step | 0.327 h | 0.65 h |
| epoch 2–4 | 24 | 40 step | 0.335 h | 1.00 h |
| epoch 5–119 | 32, 64, 128, その後 9999 | 未測定 | — | — |

epoch 0–4 を 1 seed に足すと、上の合計 411 時間は 412 時間になる。3 seed では 1,237 時間。batch 24 は実行でき、reserved 4.89 GiB だった。

epoch 5 以降を、batch 24 の 1 発話あたり 0.092 秒のまま 115 epoch 続けると、DeepFilterNet3 だけで約 40 時間、1 seed の合計は約 451 時間になる。これは測定ではなく、大きい batch でもスループットが上がらない、という仮定である。batch 32 以上は 1 step も走らせていない。スケジュールの batch 9999 はこの MIG では実行できない。120 epoch の 40 時間を、投入できる見積もりとしては使わない。

## 構成

| 構成 | 内容 | 標準の GPU 時間 |
|---|---|---:|
| 最小 | 5 モデル × 1 seed。DF は batch 16 の 2 epoch | 411 h |
| 標準 | 5 モデル × 3 seed。同じ DF | 1,234 h |
| DF を epoch 4 まで | 標準に、測った batch 24 の 3 epoch × 3 seed | 1,237 h |
| 拡張 | 上に、未測定の DF 残りを仮定の 40 時間 × 3 seed で足し、やり直し 20% を見る | 約 1,630 h |

拡張の 20% と DF 残り 40 時間は測定ではない。負の SNR 評価セットの作成は CPU である。固定 ASR の WER は未実測なので、この時間に入っていない。

フル H100 や A100 への速度換算はしない。SEMamba++ の batch 8 が 1 forward で載る GPU では、蓄積 2 より短くなる可能性がある。その秒数は無い。

## 実測と外挿

100 step の wall clock が 888 秒から 496 秒になったことは、この 5 本のプロセスで測った事実である。公開 epoch 数への 186 時間は、その step 中央値と検証外挿を epoch 数倍した見込みである。Profiler 付きの step 時間はこの倍率に入っていない。

## ジョブ時間

上の時間は学習プロセスの計算の外挿である。TSUBAME がジョブに確保する時間、課金ポイント、対話ジョブの予約とは別である。30 分を超える本学習は、投入の前にそのジョブを含む明示の承認が要る。この文書は投入の承認ではない。

実験プロセス時間は [VERIFICATION.md](VERIFICATION.md)。上限 25,200 秒は課金ポイントではない。

## TF-GridNet の実測 1 epoch（2026-11-04）

chunk 48,000、batch 4、seed 17 の 24 step で、定常中央値は 0.1916 秒だった。上の 0.191 秒をこの条件で確認した。続けた 5,000 更新と検証 872 件のプロセス時間は 1,045.70 秒（0.290 時間）である。更新だけの 958 秒に、起動、検証、checkpoint が載っている。

同じ 1 epoch が 35 回続くと 10.17 時間になる。上の表の 9.9 時間との差は 0.27 時間で、1 seed の 5 モデル合計は約 412 時間、3 seed は約 1,235 時間になる。他モデルの step 時間は測り直していない。残り 34 epoch は約 9.88 時間なので、30 分を超える投入の前にそのジョブの承認が要る。この実測は、Trainer の epoch がデータ 1 周のままであることも含めて [VERIFICATION.md](VERIFICATION.md) に書いた。公式の 5,000 反復ごとに plateau を進める時間ではない。

2026-11-04 追加監査の GPU 使用は 0 秒（失敗を含む新規 GPU プロセスなし）。前記累計 15,889.17 秒、上限 25,200 秒、残り 9,310.83 秒のまま。予約時間はプロセス時間と別であり、今回新規ジョブの予約はない。
