# P10-C Preflight Issue（修正後）

監査一覧の ID を残し、2026-11-07 の修正後の状態を付けた。状態は RESOLVED、MITIGATED、ACCEPTED_DIFFERENCE、OPEN、DISPUTED、BLOCKED、実装済み・未検証 のいずれかである。実装済み・未検証は、コードと単体テストはあるが独立検証が終わっていない。RESOLVED にはしない。テストを足しただけのもの、研究用 Validation の新設を公式バグの修正と読んだものは、RESOLVED にしない。

独立検証の追記があれば、その文書を優先して読み直す。

## 開始を止めていたもの

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-002 | RESOLVED | 検証とプレビューは全長を Generator に入れる。長さ 1650 と切り詰め 1600 の magnitude 差は 1.5627。同じ重みの公式経路との差は 0。更新経路の長さ拒否は残る |
| P10P-A-006 | RESOLVED | 残す重みは G1 と D4。担当テストは公式 state_dict との最大絶対差 0。`7729.488281` は正解にしていない |
| P10P-A-007 / P10P-D-004 | MITIGATED | 1 GPU は失敗 micro-batch を backward せず、後続の成功で蓄積窓を埋める。`official_step` は失敗を含めて増え、`global_step` は Optimizer 更新だけ数える。データ epoch の末では更新回数がずれても検証し、plateau に損失を渡す。2 GPU は片 rank の失敗で全 rank がその micro-batch を backward しない。公式の片 rank スキップとは違うので RESOLVED ではない |
| P10P-A-008 / P10P-D-002 | MITIGATED | 824 件は world 2 で 412 と 412。公式 DistributedSampler の epoch 0 と一致した。CUDA の world 2 以上だけ SyncBatchNorm。`find_unused_parameters=True`。Discriminator の `broadcast_buffers` は True。Fix.2-A で、公式の 3 forward と 2 backward は Discriminator の rank を割る。audio-db は増分を平均するので公式 2 GPU の重みとは一致しない。2 GPU の本格開始は NO-GO |
| P10P-B-005 / P10P-F-001 | RESOLVED | 本格 YAML を新設した。p10b は上書きしていない。スモーク cap は本格 YAML に無い |
| P10P-F-002 | RESOLVED | 本格 YAML の `checkpoint_interval` は 0。p10b の 2 はスモークのまま |
| P10P-F-003 | RESOLVED | `required_world_size` が実プロセス数と違うときは `build_task` の前に拒否する |
| P10P-F-007 | MITIGATED | 2026-11-08 に研究用 1,024 件を登録し、snapshot `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024` を発行した。DB は親 snapshot からの再構成で、元ファイルとの同一性は未検証。親 36,000 件は変えていない。本格 YAML の Preflight は PASS。150 epoch は始めていない。PASS は公式再現の成功ではない |

## 開始を止めていなかったもの

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-003 / P10P-G-001 | RESOLVED | 公式損失は -1 を平均に残す。研究用集約は別名のまま欠損を外す。実 824 波形の失敗件数は NOT_TESTED。807 は失敗件数ではない |
| P10P-A-011 / P10P-G-004 | OPEN | 検証 crop の乱数源は変えていない |
| P10P-B-003 | OPEN | `cv_synthetic_small` は UNKNOWN のまま。1,024 件はその名前ではない |
| P10P-B-006 | OPEN | 標準偏差の扱いは変えていない |
| P10P-D-001 | MITIGATED | 成功 micro-batch ごとに、増分勾配を rank 間平均してから走っている和を clip する。同期済みの蓄積勾配を平均し直すことと、増分だけを平均することは、PyTorch 2.6 では同じ値になる。公式の 2 回目の Discriminator backward は all-reduce されない。その割れはコピーしない。1 GPU の合成 2 秒では clip 前の Generator norm が 326.47 と 16.93、別初期化の 2 GPU では 26.28 と 56.13 で、閾値 10 を超えた。最後に 1 回だけ clip する更新とは重みが違う |
| P10P-E-001 / P10P-E-003 | MITIGATED | `validation_pending` を検証の前に残す。検証成果物だけで best を確定しない。公開済み pickle と周辺ファイルの別更新は残る |
| P10P-F-004 | RESOLVED | CLI の `--resume` は `logging` 以外の差を拒否する。上限を外した再開はモデル構築の前に失敗する |
| P10P-F-006 | OPEN | `complete.json` の中身は変えていない。完了の唯一の証拠にはしない、という Preflight の注記だけがある |

## MEDIUM

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-001 / P10P-G-002 | ACCEPTED_DIFFERENCE | 保存時点の 1 更新差は残す |
| P10P-A-004 | ACCEPTED_DIFFERENCE | 現行再開は scheduler の state を戻す。公式の作り直しは再現しない |
| P10P-A-010 | OPEN | cuDNN フラグは変えていない |
| P10P-A-012 | OPEN | 検証後に乱数を戻す動作は残る。MossFormer の公式は戻さない |
| P10P-A-013 | ACCEPTED_DIFFERENCE | 研究 YAML は公式再現の入力にしない |
| P10P-A-015 | ACCEPTED_DIFFERENCE | epoch 番号の巻き戻しは公式の欠陥として残し、現行は単調のまま |
| P10P-B-001 | ACCEPTED_DIFFERENCE | 21 batch の破棄は公式の最短 rank と同じ手順として残す。全件検証とは呼ばない |
| P10P-B-002 / P10P-G-003 | ACCEPTED_DIFFERENCE | 公式の選択は `valid.loss`。`valid.si_snr.best.pt` は公式成果物にしない |
| P10P-B-004 | OPEN | 分割端の動かし方は変えていない |
| P10P-B-007 | OPEN | README hash と recipe の対応は UNKNOWN |
| P10P-C-001 | MITIGATED | 2026-11-08 の `reproduction.duckdb` は親 snapshot からの再構成である。親の論理 snapshot hash は再構成後も `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`。元 DB ファイルとの同一性は未検証。学習は snapshot を読む |
| P10P-D-003 | OPEN | world 1 の並べ方は変えていない |
| P10P-D-006 | OPEN | DeepFilterNet3 は今回の 4 モデルの修正対象ではない |
| P10P-E-002 / P10P-F-008 | RESOLVED | `validation_pending` がある再開は、その epoch の Optimizer 更新をやり直さない。検証は先頭から。途中 batch の再開機構は無い |
| P10P-E-004 / P10P-E-005 / P10P-E-006 / P10P-E-010 | MITIGATED | CLI の `--resume` は従来どおり照合する。`experiment_profile: full` は `execute(..., strict_resume=False)` でも logging 以外の差を拒否する。profile が full でない直接呼び出しは False のまま。修正後の全件回帰は再実行していない |
| P10P-F-005 / P10P-F-009 | OPEN | git dirty は足していない。保持後の容量見積は Preflight にある。150 個全部を残す見積ではない |

## LOW と範囲外

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-005 | OPEN | MaxPool の同値タイは変えていない |
| P10P-A-009 | ACCEPTED_DIFFERENCE | plateau が履歴最小を上書きするのは公式の実効動作のまま |
| P10P-A-014 | OPEN | 定数波形の `audio_norm` は変えていない |
| P10P-D-007 / P10P-D-008 | OPEN | ログが rank 0 の損失であることは残る |
| P10P-D-009 | MITIGATED | Generator と Discriminator の DDP は `find_unused_parameters=True`。実音声の 1 更新では Generator 1053、Discriminator 93 のすべてに勾配があり、欠けは 0。他の長さは NOT_TESTED |
| P10P-E-007 / P10P-E-008 / P10P-E-009 | MITIGATED | 保存失敗で前の checkpoint が残るテストは OK。epoch 途中の未検証は `validation_pending` で表す |
| P10P-C-002 | BLOCKED | 4 モデルの開始条件ではない。DFN3 オンラインは開始できない |
| P10P-D-005 | OPEN | PESQ 失敗は micro-batch ごとに全 rank で backward しない。検証の件数が world 未満のときは同じ短い列にし、forward の前に回数を揃える。Fix.2-B と Fix.2-B.1 は、公式幅・2 GPU・実 VoiceBank batch の同じ系列を合成 3 回と実データ 3 回行い、そのあと 824 件の検証も同じプロセスで終えた。CUDA illegal memory access はどちらも出なかった。原因は不明のままなので RESOLVED にしない。片 rank の SyncBatchNorm 例外は、検証用の 20 秒 timeout で相手 rank が 9–11 秒で終了した。本番の `init_process_group` は timeout を渡さず、NCCL 既定は 10 分である |

## 意図した差

- 1 GPU の蓄積窓は、失敗した micro-batch のあとへ伸ばす。`global_step` は公式の `self.step` には合わせない。
- 2 GPU の PESQ 失敗は全 rank でその micro-batch を落とす。公式の片 rank だけのスキップは採らない。
- Discriminator の DDP は、差し戻し R2 のあと `broadcast_buffers=True`。手動の追加 broadcast はしない。
- 2 GPU 以上の再開は `rng_by_rank` が必要である。rank 0 の全 device リストを他 rank の現在 device へコピーしない。Fix.2-B.1 で直した。
- TF-GridNet は固定 `mix_seed`。公式 WAV とのビット一致も、統計的同等の実証も、しない。
- TF-Locoformer の 1,024 件は研究用の選択集合であり、公式再現の修正ではない。
- MP-SENet と MossFormerGAN の `official_faithful` と `leakage_safe` は別 YAML、別 claim、別 checkpoint ディレクトリである。
