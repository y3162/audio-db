# 学習コスト（2026-11-02、tensor 実測）

長時間学習はしていない。測ったのは公式幅の 1 更新（warmup を除いた定常 step）である。学習セットは VoiceBank 28 話者 train の 10,700 件。`steps_per_epoch = ceil(10700 / global_batch)`。`epoch_seconds = steps_per_epoch × measured_seconds_per_step`。この式にデータローダ、validation、checkpoint は入っていない。GPU-days は `epoch_hours × GPU数 / 24` で、経過時間とは別である。

## 実測条件

| 項目 | 値 |
|---|---|
| 日時 | 2026-11-02 |
| サーバー | tensor。物理 GPU 0。`CUDA_VISIBLE_DEVICES=0` なのでプロセス内は `cuda:0` |
| GPU | NVIDIA A100-PCIE-40GB。開始時の使用メモリは 928–932 MiB、使用率 0%。他の GPU は 1024 MiB 以上だったので使っていない |
| ソフトウェア | PyTorch 2.6.0+cu124。TF32 オフ。`CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Git | HEAD `2af91a3dfc0f95c0c18770bf94890b2faf2c3d77`。計測スクリプトはこの作業ツリーの未 commit 差分を含む |
| 入力 | VoiceBank の clean wav 1 本を公式長まで繰り返し、バッチ内は同じ波形の複製。noisy はその 0.6 倍に小さい乱数を足したもの |
| 更新 | 独立実装の `train_update`。forward、backward、勾配クリップ、optimizer step を含む。Forward だけではない |

結果は `data/results/training/cost-20261102/`。既存の結果ディレクトリは上書きしていない。

## tensor（実測）

| モデル | サーバー | GPU型番 | GPU数 | Global batch | 設定 | VRAM allocated | 秒/step | 時間/epoch | GPU-days/epoch | 10 epoch の経過時間 | 種別 |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| MP-SENet | tensor | A100-PCIE-40GB | 1 | 4 | micro 4、蓄積 1。segment 32,000、16 kHz | 31.33 GiB | 1.065 | 0.791 h | 0.0330 | 7.91 h | 実測 |
| SEMamba++ | tensor | A100-PCIE-40GB | 1 | 8 | micro 4、蓄積 2。segment 48,000、16 kHz、fused | 26.64 GiB | 3.294 | 1.224 h | 0.0510 | 12.24 h | 実測（蓄積あり） |
| TF-GridNet | tensor | A100-PCIE-40GB | 1 | 4 | micro 4、蓄積 1。ms-snsd、segment 48,000、16 kHz | 4.98 GiB | 0.216 | 0.161 h | 0.0067 | 1.61 h | 実測 |
| DeepFilterNet3 | tensor | A100-PCIE-40GB | 1 | 16 | micro 16、蓄積 1。48 kHz、3.0 s | 3.03 GiB | 1.659 | 0.308 h | 0.0128 | 初期 batch のままなら 3.08 h | 実測は初期 batch のみ |
| MossFormerGAN | tensor | A100-PCIE-40GB | 1 | 2 | micro 1、蓄積 2。segment 32,000、16 kHz | 16.96 GiB | 1.483 | 2.204 h | 0.0918 | 22.04 h | 実測 |

秒/step は warmup を除いた 3 または 4 step の中央値である。ばらつきは各 JSON の min/max にある。MP-SENet は 1.043–1.071 秒、TF-GridNet は 0.206–0.232 秒、DeepFilterNet3 は 1.659–1.680 秒、MossFormerGAN は 1.457–1.501 秒、SEMamba++ は 3.273–3.294 秒。

2 GPU と 4 GPU は、空いている GPU が 1 枚だったので今回は測っていない。少数 GPU からの速度外挿はしていない。

### モデルごとの注記

- MP-SENet の公式 `config.json` は batch 4、segment 32,000。その形のまま 40 GB に収まった。peak reserved は 33.6 GiB。warmup（CUDA の初回実行を含む）は 18.9 秒。checkpoint 保存は 0.64 秒、29.7 MB。batch 4 の validation 1 回は 0.90 秒。
- SEMamba++ の公式は batch 8 を 1 回の forward で計算する。同じ 40 GB で batch 8 は discriminator の途中で OOM になり、確保済みは 37.37 GiB、追加の 202 MiB を取れなかった（`semambapp-batch8` は失敗）。収まる形は micro 4、蓄積 2 で、global batch 8 の平均勾配に相当する。この step は generator を micro-batch ごとにやり直すので、batch 8 が載る GPU より遅くなる。warmup は 45.5 秒で、fused kernel の初回コンパイルを含む。定常 step とは分けてある。checkpoint は 2.51 秒、90.3 MB。
- TF-GridNet は ms-snsd、Adam、勾配クリップ 5、chunk 48,000、batch 4。checkpoint 1.06 秒、31.1 MB。validation 1 batch は 0.107 秒。
- DeepFilterNet3 の出荷 `batch_size` は 64 だが、`batch_size_scheduling` は `0/16,2/24,5/32,10/64,20/128,40/9999` である。epoch 0–1 は batch 16、epoch 2–4 は 24、epoch 5–9 は 32。今回測ったのは batch 16 の 1 step だけである。10 epoch を batch 16 のまま計算すると 3.08 時間だが、それは出荷スケジュールではない。24 以上の batch は未計測なので、出荷スケジュールの 10 epoch は未確定。checkpoint は蓄積 4 の試行で 1.20 秒、34.4 MB。batch 16 の validation 1 回は 1.57 秒。
- MossFormerGAN の公開 yaml は `batch_size: 1`、`accu_grad: 1`、`effec_batch_size: 2`、`max_length: 2` 秒。計測は micro 1 を 2 回 backward してから両 optimizer を 1 回進める。batch 2 を 1 回で更新した別計測は 0.891 秒、26.4 GiB で、公開の蓄積手順ではない。checkpoint は 3.35 秒、137 MB。

## 追加コスト（外挿）

validation 872 件を、測った 1 batch の時間で割って足すと、1 epoch あたりおおよそ次の時間になる。バッチが検証でも同じだという仮定であり、実測の全件検証ではない。

| モデル | 学習 1 epoch | validation 外挿 | 1 回の checkpoint |
|---|---:|---:|---:|
| MP-SENet | 0.791 h | 196 s | 0.64 s |
| SEMamba++ | 1.224 h | 204 s（micro 4 の時間から） | 2.51 s |
| TF-GridNet | 0.161 h | 23 s | 1.06 s |
| DeepFilterNet3 | 0.308 h（batch 16） | 87 s | 1.20 s |
| MossFormerGAN | 2.204 h | 104 s（micro 1 の時間から） | 3.35 s |

データ読み込みは step 時間に入っていない。1 ファイルの読み込みとリサンプルは MP-SENet で 1.59 秒、SEMamba++ で 1.02 秒だった。epoch 全体の I/O は未計測。

## 合計

上表の 1 GPU 設定で 5 モデルを順に 10 epoch 回すと、学習 step だけで経過 46.9 時間、消費 1.95 GPU-days である。DeepFilterNet3 は batch 16 を 10 epoch 続けた場合の数字で、出荷の batch 増加は含まない。5 枚が同時に空くなら経過時間は一番遅い MossFormerGAN の 22.0 時間で、GPU-days は同じ 1.95 である。今回その 5 枚は空いていなかった。

TF-GridNet（5 GiB）と DeepFilterNet3（3 GiB）は 1 枚に同居できる。MP-SENet（31 GiB）と SEMamba++（27 GiB）は 40 GB の 1 枚をほぼ使うので、他の学習と重ねない。

## shannon（条件付き推定。実測ではない）

shannon は今回使っていない。既知の仕様は A100 80GB が 8 枚（`docs/envs/SHINODA_LAB.md`）。SXM か PCIe か、メモリ帯域は、このリポジトリの記録では未確定である。tensor の A100-PCIE-40GB の秒数を shannon の実測として扱わない。

40 GB に収まった計算が 80 GB でも同程度の秒数になる、というシナリオを置く。不確実性は帯域と基板差で、目安を ±20% とする。これは推定であり、保証ではない。

| モデル | サーバー | GPU型番 | GPU数 | Global batch | VRAM | 秒/step | 時間/epoch | GPU-days/epoch | 10 epoch の経過時間 | 種別 |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-SENet | shannon | A100 80GB（基板未確定） | 1 | 4 | 31 GiB なら収まる | 1.065 を仮定 | 0.79 h | 0.033 | 7.9 h | 条件付き推定 |
| SEMamba++ | shannon | 同上 | 1 | 8 | 40 GB では batch 8 が OOM。80 GB なら載る可能性が高い | 未確定 | 未確定 | 未確定 | 未確定 | 未確定。上限の目安は tensor の蓄積 step 3.29 秒 |
| TF-GridNet | shannon | 同上 | 1 | 4 | 5 GiB なら収まる | 0.216 を仮定 | 0.16 h | 0.0067 | 1.6 h | 条件付き推定 |
| DeepFilterNet3 | shannon | 同上 | 1 | 16、その後 24/32/64/128 | batch 16 は 3 GiB。大きい batch は未計測 | batch 16 は 1.66 を仮定。以降は未確定 | 未確定 | 未確定 | 未確定 | 初期 epoch のみ条件付き推定 |
| MossFormerGAN | shannon | 同上 | 1 | 2（蓄積 2） | 17 GiB なら収まる | 1.483 を仮定 | 2.20 h | 0.092 | 22 h | 条件付き推定 |

80 GB では SEMamba++ の公式 batch 8 を蓄積なしで試せる。40 GB では 37.4 GiB まで使って OOM したので、80 GB に収まる見込みはある。その秒数は測っていない。蓄積 2 の 3.29 秒は、公式の 1 forward より長い側の目安である。

## 再現

```bash
# tensor。実行前に nvidia-smi で 1024 MiB 未満の GPU を選ぶ。
ssh tensor 'cd /net/spring/work/seki/exp/audio-db && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'
export CUDA_VISIBLE_DEVICES=<空き>
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venvs/lab/bin/python tests/training/reference/benchmark_training_cost.py \
  --model mp_senet --device cuda:0 \
  --output data/results/training/NEW/mp-senet.json
```

SEMamba++ は `.venvs/semamba/bin/python` と、sm_80 の `PYTHONPATH=.venvs/semamba-sm80` が要る。出力先が既にあるファイルはスクリプトが拒否する。
