# P3 検証記録

2026-11-06。固定ペアの Data Pipeline を MP-SENet と MossFormerGAN の既存学習タスクへ接続した。本学習と P4 以降は行っていない。

## 実行したもの

| 項目 | 結果 |
|---|---|
| `unittest discover -s tests/reproduction` | 39 件 OK（77.7 秒）。内訳は P1 が 13、P2 が 7、P3 が 19 |
| Legacy 黄金比較 | `tests/training/test_legacy_adapter.py`。同じ seed の順序・波形、Trainer の 2 step の重み、prefetch を捨てた再開が RecipeDataset 経路と一致 |
| 公式 ClearerVoice | ピン留め `dataloader.py` の `DistributedSampler` と `audio_norm` をテストから実行。rank 順は一致。`audio_norm` の最大差は 1e-6 未満 |
| 実スナップショット | 4 Dataset を `open_training_data` で開ける。official は `allow_test_selection: false` で拒否 |
| GPU スモーク | ホスト `r3n11`、見える GPU は H100 1 枚（使用メモリ 101 MiB から開始）。`tests/reproduction/p3_gpu_smoke.py` |

GPU スモークは各モデルを別プロセスで、更新 1 回のあと checkpoint し、同じプロセス内の再開で更新 2 まで進めた。validation は公式 test 全件ではなく、Pipeline の先頭 1 バッチだけを既存タスクの `validation_report` に渡した。

| モデル | world | 停止 | plan hash の再計算 | validation |
|---|---|---|---|---|
| MP-SENet official | 1 | CHECKPOINTED のあと COMPLETED、global step 2 | 一致 `05d5c1386d12…` | 全長 27,861、損失は有限 |
| MossFormerGAN official | 1 | 同じ | 一致 `fd433af74050…` | 長さ 32,000、time L1 は有限 |

計画は CPU 上の純関数である。GPU プロセスが保存した hash と、後から CPU で開き直した hash は一致した。2 GPU は見えないので、実 DDP はしていない。

## 公式との差（記録）

- crop の範囲は公式の `randint(0, L-window)` と同じ両端を含む一様分布。乱数列は DataLoader worker のものではなく、`mix_seed("crop", seed, epoch, case_id)`。同じ case と epoch は同じ開始位置になる。
- MP-SENet の 1 GPU 順は `random.Random(1234).shuffle` を全 epoch で再利用する。2 GPU 以上は `DistributedSampler(seed=0, drop_last=True)`。公式コンストラクタは `drop_last` を渡さない。N が world で割り切れるとき（11,572 と world 2 または 4）は pad も切り詰めも起きず、両者は同じである。
- MossFormerGAN の順序は world 1 でも `rank_mod_fixed_v1`（seed 0）。公式の非 DDP は `DataLoader(shuffle=True)` で、その RandomSampler 列は再現しない。DDP の独自 Sampler は、ピン留めクラスを実行して一致した。
- 短い音声は、公式コードと同じく先頭へ 1 回コピーし、残りを 0 にする。頭を繰り返して埋めない。
- `independent_audio_norm_v1` のピーク集合が空（定数信号）のとき、公式は NaN になる。Pipeline はその計算を補正せず `ReproductionError` にする。
- MossFormerGAN の発話ごと `c` は従来どおりタスク側。Pipeline は掛けない。
- 検証の全 824 / 872 件を 1 回のスモークでは回していない。

`data.virtual_world_size` と `espnet_chunk_v1` は指定すると拒否する。オンライン混合と事前生成の Renderer は無い。
