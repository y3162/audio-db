# P10-C Fix.2-C 暫定 GO

日付: 2026-11-08。判定は **GO**。150 epoch は始めていない。第三者検証はしていない。

判定の対象は、TF-Locoformer 研究用 Protocol を本格 YAML で開始できるかである。公式 DNS の再現成功ではない。`cv_synthetic_small` でも `leakage_safe` でもない。

## 必須条件

| # | 条件 | 結果 |
|---|---|---|
| 1 | 登録元が確定する | 親 snapshot からの再構成。`database_origin=reconstructed_from_verified_snapshot`。元 DB との同一性は未検証 |
| 2 | Dataset が登録される | 済 |
| 3 | Dataset id が発行される | `e4a53415122aba24055f4ba7eef6367a7e41d062485d74de3d7ce94c07d7bdd0` |
| 4 | Snapshot hash が発行される | `e4495fa6ac0048c6346cc65bc66fd71114fd1523f95230537e799610c8092024`。再 export も同じ |
| 5 | Validation 1,024 件が一致する | 親 valid の部分集合。一覧 SHA-256 `73e8d44a…` |
| 6 | Train / test が親と一致する | case id 列が一致 |
| 7 | 親 snapshot が不変 | `manifest.json` は 2026-11-07 09:30:31 のまま |
| 8 | Claim と来歴 | `research_checkpoint_selection`。`selection_uses_test=false`。manifest に再構成の来歴 |
| 9 | Snapshot Reader | 子 snapshot を開けた |
| 10 | On-demand Render | train 1 件と valid 1 件を二回描画し、最大絶対誤差 0。全件ではない |
| 11 | World 4 の Data Plan | train 14,000 更新、valid は 1,024 件が欠落なく 4 rank に分かれる |
| 12 | 本格 YAML の Preflight | PASS |
| 13 | 短時間の学習と validation | GPU 4 枚で 3 更新。validation は rank あたり 1 batch。1,024 件の完走ではない |
| 14 | Checkpoint の保存と再開 | step 1 から step 3 まで再開し `INTERRUPTED`。ビット一致は未確認 |
| 15 | 関連テスト | 27 件 OK。`tests/reproduction` 全体は再実行していない |

## 残るもの

- 元の `reproduction.duckdb` は見つかっていない。今のファイルに VoiceBank、MS-SNSD、P7 recipe / smoke、run は無い
- `catalog_id` は再計算で、元の値とは比べていない
- asset SHA-256 は 32 / 130,950 件だけ再計算した
- 雑音ファイルは train と共有する。話者と録音 ID は UNKNOWN
- 定義 JSON の `registration_status` は Fix.1 の BLOCKED のままである。登録の記録は `data/staging/reproduction/research_valid_fixed_v1/registration_result.json`
- 定義が note を変えて計算した内容 id `8bd167b5…` は、発行した dataset id ではない
- `deterministic: false` のため、再開の重み一致は主張しない
- 5 件未満なので `ave_5best` の平均ファイルはこの smoke では作っていない。`ave_1best` は作った

## 公式との差

研究用 1,024 件は公式の `cv_synthetic_small` ではない。親の 36,000 件もその名前ではない。混合は seed 付きの受理計画で、公式の共有カウンタとのビット一致は主張しない。Preflight の PASS は実行条件の確認である。

## Fix.2-D へ

- この DB を、VoiceBank や MS-SNSD や P7 recipe が入った歴史上の `reproduction.duckdb` として読まない
- 研究用 Dataset を独立な汎化評価と呼ばない
- 本格 150 epoch は、この GO だけでは始めない
- 全モデル回帰で TF-Locoformer を見るときは、snapshot `e4495fa6…` と本格 YAML の cache 1,024 を使う。今回の GPU smoke の cache は 128 である
- 1,024 件の validation 完走はまだしていない。計画では rank あたり 3,584 更新、破棄 0
