# P10-C Preflight Fix.1 計画

本格学習は開始しない。公式ソース `src/se/official/` は変更しない。`resource.duckdb`、既存 snapshot、既存 checkpoint は変更しない。

## 担当

| 役割 | 状態 |
|---|---|
| Implementation | この作業。本番コード、単体テスト、この計画と実装記録を書く |
| Verification | 未実施。この計画のテスト結果を独立検証の合格にしない |

## 対象 Issue

| ID | この修正での扱い |
|---|---|
| P10P-A-007 / P10P-D-004 | 1 GPU の PESQ 失敗は蓄積窓を後続の成功 micro-batch へ伸ばす。カウンタは分ける |
| P10P-A-008 / P10P-D-002 | 検証を公式 DistributedSampler の rank 分割に合わせる。Discriminator の `broadcast_buffers=False` は残す |
| P10P-D-001 | clip は、その micro-batch の増分を rank 間平均したあと |
| P10P-D-009 | `find_unused_parameters=True` |
| P10P-D-005 | 検証の forward 回数を揃える。SyncBatchNorm 内部の片 rank 例外は未解決として残す |
| P10P-F-007 | 研究用 1,024 件の登録手順をコード化する。正本 DB が無い間は BLOCKED |
| P10P-E-004 系 | `experiment_profile: full` の再開は `strict_resume=False` でも照合する |

## 修正範囲

- MossFormerGAN の `distribute`、`train_update`、`validation_report`、`reduce_validation`
- 固定ペア Pipeline の検証順。`rank_mod_fixed_v1` は rank ごとの epoch 0
- Trainer の検証集約、forward 回数の揃え、epoch 境界の蓄積破棄、full 再開の照合
- `research_checkpoint_selection` を claim と CHECK に追加し、追記登録関数を置く
- 他モデルの `validation_requires_all_ranks` が偽のときは、rank 0 だけが検証する契約を維持する

## 公式との差異

公式 Solver は Generator と Discriminator の両方を `SyncBatchNorm.convert_sync_batchnorm` し、`find_unused_parameters=True`、`broadcast_buffers` 既定 True で DDP する。検証 sampler は `set_epoch` されず epoch 0 のままである。shuffle は `floor(N/world)` 個のストライドだけを並べ、余った末尾を捨て、`ceil(N/world)` まで前方を繰り返す。各 rank のサンプル数は同じである。DataLoader の `drop_last` は false なので、短い最終 batch は 1 batch として分母に残る。損失が None の batch は加算せず、分母 `(i+1)` には残す。そのあと rank 平均は mean of means である。

学習の `self.step` は失敗 batch でも増える。`accu_count` は成功 batch だけ増える。clip は各成功 backward のあと、蓄積された勾配に対して行う。

一致させるもの:

- epoch 0 の rank 分割、捨てる末尾、繰り返す前方、rank あたりのサンプル数
- 失敗 batch を分母に残し、PESQ 値や損失値としては 0 を置かない
- 1 GPU の蓄積窓を後続の成功へ伸ばす
- `find_unused_parameters=True`
- Generator に対する `convert_sync_batchnorm`。生きた BatchNorm が 0 なら no-op

残す差:

- Discriminator の `broadcast_buffers=False`。3 回の forward が 1 つのグラフを共有し、True は backward 用の BatchNorm 状態を壊し得る。公式は True のままその構造である。この差があるので 2 GPU の本格開始は NO-GO のまま
- 2 GPU では片 rank の PESQ 失敗で全 rank がその micro-batch を backward しない。公式の片 rank だけのスキップは collective が割れる
- 勾配平均は、その micro-batch の増分だけを平均する。公式の `no_sync` なしの再平均はコピーしない
- `global_step` は Optimizer 更新回数のまま。公式 `self.step` は別カウンタ `official_step`

## 受け入れ条件

- 単体テストが、公式 sampler の rank 列、失敗 batch の分母、DDP フラグ、1 GPU の窓の継続、full 再開の拒否、claim の追記と親行の不変をアサートする
- 実 GPU の勾配 norm、実学習形状の未使用パラメータ、2 GPU の本格 step は NOT_TESTED と書く
- `reproduction.duckdb` を新規作成しない。無いときは登録を BLOCKED のままにする
- 独立検証が終わるまで RESOLVED と書かない
- Preflight の PASS は公式再現の成功ではない。Dataset が BLOCKED の研究用 YAML を PASS にしない
