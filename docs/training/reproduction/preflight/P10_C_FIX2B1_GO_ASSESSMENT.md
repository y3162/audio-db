# P10-C Fix.2-B.1 GO 判定

実装担当の自己検証である。第三者検証はしていない。本格の 121 epoch は始めていない。

## 必須条件

| # | 条件 | 結果 |
|---|---|---|
| 1 | 保存直前と復元直後が、保存対象の全 rank で整合する | 満たす。重み、BN、Adam、勾配、rank 別 RNG の読み込み直後の最大絶対差は 0 |
| 2 | rank 別 RNG が保存・復元される | 満たす。`rng_by_rank` が現在 device を戻す。欠ける複数プロセス checkpoint は拒否する |
| 3 | Optimizer と蓄積勾配が復元される | 満たす。読み込み直後は差 0。`accu_count=1` は窓を残し、epoch 境界では捨てる |
| 4 | BN 浮動差の箇所と原因 | 満たす。保存時の buffer は rank 間で一致する。以前の `1.45e-2` は rank 1 の CUDA RNG が seed へ戻った後段である |
| 5 | 連続実行と再開が数値基準を満たす | 満たす。同じ step 1 からの再開は、パラメータ、Adam、BN、勾配が事前 atol 以内である。再開どうしの差と同じ大きさで、再開固有の上乗せは無い |
| 6 | 両 rank の更新後パラメータが一致する | 満たす。測定した更新の rank 差と Adam 浮動 moment の差は 0 |
| 7 | step、case id、plan hash | 満たす。`global_step` 3、`official_step` 6、plan hash と case id 列は一致 |
| 8 | PESQ 失敗を含む更新と再開 | 満たす。片 rank 失敗、両 rank 失敗、非有限損失、失敗を挟む窓は終了した |
| 9 | SyncBatchNorm と Validation | 満たす。R5 の SyncBatchNorm forward と、824 件を 412 と 412 で終えた |
| 10 | CUDA の既知系列 | 満たす。合成 3、実 batch 3 が終了し、illegal memory access は 0。原因不明のまま `NOT_REPRODUCED` |
| 11 | Plateau の二重更新が無い | 満たす。`validation_pending` の再開は Validation 確定を 1 行だけ書き、`global_step` は 2 のまま |
| 12 | 関連回帰テスト | 満たす。列挙した CPU テストは PASS |
| 13 | 本格 YAML の Preflight | 満たす。`configs/training/reproduction/mossformergan_vb28_official_2gpu.yaml`、world size 2、`/tmp/p10c-fix2b1/preflight` で PASS。PASS は実行条件であり、公式再現の成功ではない |
| 14 | Fix.2-A の分散更新 | 満たす。増分平均、`broadcast_buffers=True`、`find_unused_parameters=True`、片 rank PESQ の全 rank スキップは変えていない |

## 判定

**GO**

実装担当の暫定 GO である。第三者の追試はまだ無い。

`deterministic: false` の同じ checkpoint 再開は、事前の atol を満たす。最初の forward は連続実行とハッシュが一致し、`rng_by_rank` も一致する。残る重み差は、一致した forward のあとで二つの再開どうしにも同程度出る cuDNN の差である。独立した連続 2 回は Generator `1.18e-3`、Discriminator `1.35e-3` で atol を超えるが、これは再開経路の上乗せではない。

`deterministic: true` は `adaptive_max_pool2d_backward_cuda` で停止した。`warn_only` で通してはいない。公式 pooling は変えていない。本格 YAML は `deterministic: false` のままである。決定的モードのビット一致は主張しない。

## 意図した公式差異

公式 2 GPU の 2 回目の Discriminator backward は rank の重みを割る。audio-db はその増分を平均するので、公式の割れた重みとは一致させない。これは Fix.2-A のままである。GO の数値は、audio-db の連続実行と audio-db の再開の比較である。

## 残存リスク

- CUDA illegal memory access の原因は不明である。既知の 6 回と、その前後の短い 2 GPU 実行では再現しなかった。再発しない証明ではない。
- `deterministic: false` では、別プロセスの連続実行どうしがパラメータ atol を超えることがある。同じ重みから続けた再開は超えていない。
- `rng` の未使用 device 一覧は、再開後に連続実行と一致しない。学習 RNG は `rng_by_rank` である。
- `rng_by_rank` の無い古い 2 GPU checkpoint は再開できない。
- `segment_steps` と `deadline` は world size 1 のままである。NCCL の timeout は渡しておらず、既定は 10 分である。
- 824 件 Validation は R5 の同じプロセスで 1 回である。121 epoch の Validation 頻度では走らせていない。

## Fix.2-C への引き継ぎ

MossFormerGAN の 2 GPU は、実装担当の自己検証では暫定 GO である。残りの採用モデルが同じ自己検証で暫定 GO になるまで、第三者検証は始めない。本格学習も始めない。TF-Locoformer の DB 登録には進まない。他モデルの更新式は、この RNG 修正では変えていない。2 GPU 以上の古い checkpoint を再開するときは、`rng_by_rank` が無い拒否を前提にする。
