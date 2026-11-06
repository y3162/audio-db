# P5 検証（2026-11-06）

Chunk Iterator の複数 rank と virtual rank。本格学習と MS-SNSD の全量生成は含まない。

## 公式コードで確定したこと

ESPnet `0d136c5`（`src/se/official/espnet`）と `90eed8e`（chunk iterator、`abs_task.build_chunk_iter_factory`、`validate_one_epoch` を取得して比較）:

- shard は `batches[rank::world]`。`num_cache_chunks // world_size`。`num_samples_per_epoch = num_iters_per_epoch`。学習だけ shuffle する。両 commit でこの式は同じ。
- `batch_size` が world で割り切れないとき、公式は低い rank に 1 件多く渡す。公開レシピは 4/2 と 4/4 で割り切れる。audio-db は割り切れない構成を拒否する。
- `90eed8e` の chunk iterator は短い発話を常に捨てる。`0d136c5` は `discard_short_samples` の既定 True で同じ。幅の抽選は `choice(..., size=1)` で、乱数の消費は維持される。
- 学習も validation も `iterator_stop`。尽きた rank が出たら、他 rank の残りバッチは学習・評価に使わない。
- epoch 開始の `set_all_random_seed(seed + epoch)` は全プロセスで同じ引数。rank は足さない。TF-GridNet と TF-Locoformer DNS の dropout は 0。
- 基底の発話順は shape file。そのファイルは事前生成の前には無い。計画は呼び出し側の列を並べ替えない。

## 何を実行したか

| 区分 | 結果 | 内容 |
|---|---|---|
| world size 1 の黄金 | PASS | 以前の `epoch_chunk_batches` と同じ開始位置・長さ・順序 |
| 公式 factory の実行 | PASS | `.venvs/official` で `ChunkIterFactory` を world 1・2・4、学習と validation について実行。ID・開始・終了が一致。`.venvs/se` には typeguard が無く SKIP |
| Virtual rank | PASS | world 2・4 で同じ入力から同じ `plan_hash`。prefetch を捨てて再計算しても同じバッチ |
| 再開 | PASS | virtual world 2 の連続実行と、2 step で切った再開の `case_ids` が一致。world を 2 から 4 に変える再開は拒否 |
| CPU 2 process | PASS | gloo。両 rank の更新回数は 1。重み付き validation が全件の平均と一致。更新後の重みは rank 間で一致。来歴は rank 0 |
| 2 GPU DDP | PASS | ジョブ `9279643`、`node_h=1`、H100 が 2 枚、MIG 無効。TF-GridNet 縮小網、chunk 48,000、rank あたり batch 2。`plan_hash` `e88fb48b0f57402c72dd7ec67b09a100b0c6ae925399c0656c77ad4f59f7e180`。validation の weight は 8 |
| 4 GPU DDP | PASS（再実行）| ジョブ `9279853`、`node_f=1`、H100 が 4 枚、MIG 無効。TF-Locoformer 縮小網、chunk 64,000、rank あたり batch 1。`plan_hash` `cd88d921c8f106eff7b274e528af4770d8539e706e8d679c89951730ed15745c`。validation weight 8。再開後の rank 0 は `t4`。先行ジョブ `9279686` は scheduler の `threshold` 欠落で失敗し、同じ資源で再実行した |
| P1〜P4 | PASS（再現・評価の来歴）| `tests/reproduction` 49 件、`tests/evaluation/test_provenance` 15 件。`tests/training` は 220 件中、P5 以前から欠ける設定と `voicebank.duckdb` 不在で 3 FAIL と 1 ERROR。`tf_gridnet_chunk_seed17.yaml` と `tf_gridnet_fixed5000_seed17.yaml` は履歴から戻し、そのテストは PASS |

Virtual rank の逐次 backward は DDP の all-reduce とビット一致するとは言わない。スモークのネットワークは公開幅より小さい。

## GPU 時間

要求 `h_rt` の合計は 01:30:00。承認残は 00:30:00。実時間は各ジョブ約 1〜2 分。キュー待ちを含む。

| ジョブ | 資源 | 要求 | 結果 |
|---|---|---|---|
| 9279643 | `node_h=1`、H100 ×2 | 00:30:00 | PASS |
| 9279686 | `node_f=1`、H100 ×4 | 00:30:00 | FAIL（設定漏れ。4 GPU の代用ではない）|
| 9279853 | `node_f=1`、H100 ×4 | 00:30:00 | PASS |

## 判定

P5 の chunk 計画・virtual rank・2 rank・4 rank は完了。`valid.loss.ave_5best` は P7 のまま。shape file の実順は生成後まで未確定。
