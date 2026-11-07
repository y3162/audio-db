# P10-C Preflight Issue（修正後）

監査一覧の ID を残し、2026-11-07 の修正後の状態を付けた。状態は RESOLVED、MITIGATED、ACCEPTED_DIFFERENCE、OPEN、DISPUTED、BLOCKED のいずれかである。テストを足しただけのもの、研究用 Validation の新設を公式バグの修正と読んだものは、RESOLVED にしない。

独立検証の追記があれば、その文書を優先して読み直す。

## 開始を止めていたもの

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-002 | RESOLVED | 検証とプレビューは全長を Generator に入れる。長さ 1650 と切り詰め 1600 の magnitude 差は 1.5627。同じ重みの公式経路との差は 0。更新経路の長さ拒否は残る |
| P10P-A-006 | RESOLVED | 残す重みは G1 と D4。担当テストは公式 state_dict との最大絶対差 0。`7729.488281` は正解にしていない |
| P10P-A-007 / P10P-D-004 | MITIGATED | 欠測 batch は更新しない。`global_step` も増えない。公式は失敗時も `self.step` を増やし、成功 batch で蓄積窓を埋める。検証の非有限損失はまだ例外 |
| P10P-A-008 / P10P-D-002 | MITIGATED | Discriminator だけ SyncBatchNorm。短い GPU forward は両 rank で終わった。検証 batch は全 rank で同一で、公式の rank 分割ではない。`broadcast_buffers=False`、`find_unused_parameters=False`。2 GPU の本格開始は NO-GO |
| P10P-B-005 / P10P-F-001 | RESOLVED | 本格 YAML を新設した。p10b は上書きしていない。スモーク cap は本格 YAML に無い |
| P10P-F-002 | RESOLVED | 本格 YAML の `checkpoint_interval` は 0。p10b の 2 はスモークのまま |
| P10P-F-003 | RESOLVED | `required_world_size` が実プロセス数と違うときは `build_task` の前に拒否する |
| P10P-F-007 | MITIGATED | 選択集合は 1,024 件に決めた。150 epoch の検証は外挿で 23.92 時間。再計測ではない。登録と snapshot は BLOCKED。この YAML では学習を始めない |

## 開始を止めていなかったもの

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-003 / P10P-G-001 | RESOLVED | 公式損失は -1 を平均に残す。研究用集約は別名のまま欠損を外す。実 824 波形の失敗件数は NOT_TESTED。807 は失敗件数ではない |
| P10P-A-011 / P10P-G-004 | OPEN | 検証 crop の乱数源は変えていない |
| P10P-B-003 | OPEN | `cv_synthetic_small` は UNKNOWN のまま。1,024 件はその名前ではない |
| P10P-B-006 | OPEN | 標準偏差の扱いは変えていない |
| P10P-D-001 | OPEN | clip を平均の後へ移していない。実勾配 norm は NOT_TESTED |
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
| P10P-C-001 | BLOCKED | この作業木に `reproduction.duckdb` が無い。学習は既存 snapshot を読む。DB と snapshot の同一性は未証明 |
| P10P-D-003 | OPEN | world 1 の並べ方は変えていない |
| P10P-D-006 | OPEN | DeepFilterNet3 は今回の 4 モデルの修正対象ではない |
| P10P-E-002 / P10P-F-008 | RESOLVED | `validation_pending` がある再開は、その epoch の Optimizer 更新をやり直さない。検証は先頭から。途中 batch の再開機構は無い |
| P10P-E-004 / P10P-E-005 / P10P-E-006 / P10P-E-010 | MITIGATED | CLI 再開は cap と dataset identity を見る。`execute(..., strict_resume=False)` の直接呼び出しは照合しない |
| P10P-F-005 / P10P-F-009 | OPEN | git dirty は足していない。保持後の容量見積は Preflight にある。150 個全部を残す見積ではない |

## LOW と範囲外

| ID | 状態 | 修正後 |
|---|---|---|
| P10P-A-005 | OPEN | MaxPool の同値タイは変えていない |
| P10P-A-009 | ACCEPTED_DIFFERENCE | plateau が履歴最小を上書きするのは公式の実効動作のまま |
| P10P-A-014 | OPEN | 定数波形の `audio_norm` は変えていない |
| P10P-D-007 / P10P-D-008 | OPEN | ログが rank 0 の損失であることは残る |
| P10P-D-009 | OPEN | `find_unused_parameters=False` のまま。今回の GPU forward はこの設定で完了した。実学習形状の未使用パラメータは NOT_TESTED |
| P10P-E-007 / P10P-E-008 / P10P-E-009 | MITIGATED | 保存失敗で前の checkpoint が残るテストは OK。epoch 途中の未検証は `validation_pending` で表す |
| P10P-C-002 | BLOCKED | 4 モデルの開始条件ではない。DFN3 オンラインは開始できない |
| P10P-D-005 | OPEN | 片 rank の例外で他 rank が collective に残る実測は、今回の PESQ スキップ以外ではしていない。PESQ スキップは全 rank で backward しない |

## 意図した差

- 公式 1 GPU の蓄積窓を、失敗した固定群のあとへ伸ばさない。
- SyncBatchNorm の DDP は `broadcast_buffers=False`。公式の既定は True。
- TF-GridNet は固定 `mix_seed`。公式 WAV とのビット一致も、統計的同等の実証も、しない。
- TF-Locoformer の 1,024 件は研究用の選択集合であり、公式再現の修正ではない。
- MP-SENet と MossFormerGAN の `official_faithful` と `leakage_safe` は別 YAML、別 claim、別 checkpoint ディレクトリである。
