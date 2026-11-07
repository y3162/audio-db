# P10-C Preflight Fix.1 実装記録

Implementation の記録である。Verification は未実施。ここに書いた単体テストの成功を、独立検証の合格とは読まない。

## 修正前

開始時の HEAD は `c0dc3c6672e9275133a7b5a6dce878eee4068c62`、作業木はクリーンだった。実装前に公式 `solver.py`、`dataloader.py`、`loss.py` と現行の `mossformergan.py`、Trainer の検証経路を読み直した。

修正前の要点:

- world size が 2 以上でも、検証は全 rank が同じ batch 列だった。公式の DistributedSampler は rank ごとにスライスする
- PESQ 失敗は、その group 全体を backward も Optimizer 更新もせず、`global_step` も増やさなかった。公式の 1 GPU は失敗 batch でも `self.step` を増やし、成功 batch だけで蓄積窓を埋める
- Discriminator だけ SyncBatchNorm。`broadcast_buffers=False`、`find_unused_parameters=False`
- clip は、最後以外の micro-batch では rank 間平均の前だった
- `research_checkpoint_selection` は claim の CHECK に無かった
- `experiment_profile: full` でも `execute(..., strict_resume=False)` は再開照合をスキップした

## 変更内容

### MossFormerGAN

`distribute` は world size が 2 以上のとき、公式と同じく Generator と Discriminator の両方に `convert_sync_batchnorm` を呼ぶ。構築された Generator の BatchNorm は 0 なので、この呼び出しはモジュールを増やさない。公式 Generator も 0 だった。Discriminator には BatchNorm がある。

DDP は両方 `find_unused_parameters=True`。この節を書いた時点では Discriminator の `broadcast_buffers` を False に残していた。差し戻し R2 のあと True であり、手動 broadcast は無い。最終状態は末尾の R2 を読む。2 GPU の本格開始は、公式学習との数値一致が未実施なので NO-GO のままである。

`train_update` は micro-batch ごとに判定する。失敗は backward しない。`accu_count` は成功だけ増やす。窓が `accumulation` に達したときだけ両方の Optimizer を step し、勾配を捨てる。窓の残りは次の呼び出しへ持ち越す。epoch の先頭では、公式が epoch 開始時に `accu_count` を 0 に戻すのと同じく、途中の窓を捨てる。再開が epoch の途中なら、checkpoint に保存した窓と勾配を戻す。

カウンタは分けた。

| 名前 | 意味 |
|---|---|
| `micro_batches_processed` | 処理した micro-batch |
| `micro_batches_succeeded` | backward した micro-batch |
| `micro_batches_failed` | backward しなかった micro-batch |
| `optimizer_updates` | 両方の Optimizer を step した回数 |
| `official_step` | 公式 `self.step` に合わせ、失敗を含めて micro-batch ごとに増やす |
| Trainer `global_step` | Optimizer 更新の回数。`official_step` には合わせない |

2 GPU では、その micro-batch をいずれかの rank が落とすと、全 rank が backward しない。データカーソルは進め、同じ batch はやり直さない。Scheduler は epoch 境界のままなので、失敗そのものでは増えない。片 rank だけの公式スキップは、もう一方が DDP の collective に残るので採らない。

成功した backward のあと、その micro-batch で増えた勾配だけを SUM して world size で割り、走っている和へ足してから clip する。閾値は 10 のまま。実 MossFormerGAN で 10 を超えるかは GPU を使っていないので NOT_TESTED。CPU で言える順序の差は、平均の前に局所勾配を clip しないことである。公式が `no_sync` なしで、すでに蓄積した勾配まで平均し直す動作はコピーしていない。

検証の 1 batch は、PESQ 失敗や非有限のとき `generator` を `na` とし、値は null のままである。`reduce_validation` が、含まれた損失の和を batch 数で割る。失敗 batch は分母に残り、損失 0 としては保存しない。PESQ 指標自体も `na` のままである。rank ごとの平均を、さらに rank 数で平均する。これは公式の `_reduce_tensor` と同じ mean of means である。batch 数が rank 間で同じなら、全体の和を全体の batch 数で割った値と一致する。

### 検証の rank 分割

`rank_mod_fixed_v1` の検証は、epoch 0 のその rank のスライスだけを返す。公式の検証 sampler は `set_epoch` されない。

`N=10`、`world=3` では、rank あたり 4 サンプル、末尾 1 件を捨て、前方から 3 件を繰り返す。短い最終 batch は drop しない。batch 数が 1 として分母に入る。重複した発話はもう一度分母に入る。捨てた末尾は分母に入らない。

Trainer は、それでも batch 数が違うときは、空の rank を除いて前方の batch を繰り返し、追加分を `validation_alignment_pad` として forward する。この追加は公式のサンプルではなく、SyncBatchNorm の回数を揃えるためだけなので、分母から外す。空の rank は forward の前に全 rank で拒否する。SyncBatchNorm の内側で片 rank だけが例外になると、その collective はまだ deadlock し得る。

`validation_requires_all_ranks` が偽のモデルは、従来どおり rank 0 だけが検証する。

### 研究用 Dataset

`research_checkpoint_selection` を `CLAIM_LABELS` と `repro.datasets` の CHECK に追加した。DuckDB は CHECK の DROP が未実装なので、datasets とそれを参照する表をコピーして、現在の CREATE TABLE で作り直す。行の値はコピーするだけで、列順は変えない。

追記関数は `register_research_validation_dataset`。データベースファイルが無いときは作らない。親 Dataset の行は更新しない。子は新しい protocol、validation case set、dataset だけで、train と test の binding は親の id を指す。test と case id が交わる選択は拒否する。二回目の同じ内容は挿入 0 である。

この作業木では次を探した。広い `/net` の探索はしていない。

- `data/db/reproduction.duckdb`
- `data/db/backups/reproduction.duckdb`
- `data/staging/reproduction/reproduction.duckdb`

どれも無かった。`data/db/snapshots` の親 hash `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5` も無い。登録は BLOCKED。snapshot hash は発行していない。親 36,000 件は変更していない。一時 DuckDB のテストでは、合成の親から子を追記し、親の claim が `leakage_safe_comparison` のままであること、子の valid 件数、snapshot hash の再実行一致、Snapshot Reader で開けることを確認した。これは本番の 1,024 件 Dataset の登録ではない。

研究用 YAML の Preflight は data gap のままである。PASS にはしていない。

### 共通 Trainer

`experiment_profile: full` の再開は、`strict_resume=False` でも logging 以外の差を拒否する。profile が full でない直接呼び出しは、既存テストが使う `False` のままである。

Checkpoint の保持、`validation_pending`、原子的 replace の実装は、今回落ちているテストが無かったので作り直していない。保持と中断の既存テストは、この変更のあとに再実行した。

## ファイル

- `src/training/se/mossformergan.py`
- `src/training/core/trainer.py`
- `src/training/se/run.py`
- `src/reproduction/pipeline/order.py`
- `src/reproduction/pipeline/planner.py`
- `src/reproduction/pipeline/training_data.py`
- `src/reproduction/ids.py`
- `src/reproduction/db/schema.py`
- `src/reproduction/db/validate.py`
- `src/reproduction/db/registry.py`
- `src/reproduction/prepare/research_valid_fixed.py`
- `.agents/DECISIONS.md`
- `.agents/DATA.md`
- 対応する `tests/training` と `tests/reproduction`
- `docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX_ISSUES.md`
- この計画とこの記録

## 公式との差異

解消したもの:

- 検証の rank 分割、末尾の破棄、前方の繰り返し、rank あたりの件数
- 失敗 batch を分母に残し、0 として保存しないこと
- 1 GPU で蓄積窓を後続の成功へ伸ばすこと
- `find_unused_parameters=True`
- Generator への `convert_sync_batchnorm` の呼び出し

残したもの:

- 差し戻し前は Discriminator の `broadcast_buffers=False` を残していた。差し戻し R2 で True に変えた。詳細は末尾の差し戻し対応
- 2 GPU の PESQ 失敗を全 rank で落とすこと
- 増分だけを平均してから clip すること。公式の蓄積済み勾配の再平均は採らない
- `global_step` と `official_step` を一つの数にしないこと
- 実勾配 norm と、実形状の未使用パラメータ。NOT_TESTED
- 研究用 Dataset の本番登録。BLOCKED

## 単体テスト

失敗を skip にはしていない。`tests/training` と `tests/reproduction` の全件は走らせていない。Verification に残す。

```text
.venvs/se/bin/python -m pytest \
  tests/training/test_mossformer_sampler.py \
  tests/training/test_mossformer_validation.py \
  tests/training/test_mossformer_pesq_skip.py \
  tests/training/test_mossformer_official_init.py \
  tests/training/test_mossformer_plateau.py \
  tests/training/test_mossformer_discriminator.py \
  tests/training/test_mossformer_syncbn_collective.py \
  tests/training/test_checkpoint_retention.py \
  tests/training/test_validation_interrupt.py \
  tests/training/test_resume_identity.py \
  tests/training/test_preflight_full.py \
  tests/reproduction/test_research_claim.py \
  tests/reproduction/test_p3.py::OrderTests::test_rank_mod_matches_the_official_sampler_formula \
  tests/reproduction/test_p3.py::PlanAndRenderTests::test_validation_is_full_length_or_windowed_and_skips_held_out_test \
  -q --tb=line
```

結果: 46 passed, 0 failed, 0 error, 0 skipped。292.00 秒。warning は既存の `src/se/models/tf_locoformer/separator.py` の `torch.cuda.amp.autocast` である。GPU は使っていない。

この数字は Implementation の単体テストであり、独立検証ではない。

## 未解決

- Discriminator `broadcast_buffers=True` を、3 forward のグラフのままで安全にする方法は採っていない
- SyncBatchNorm の内側で片 rank だけが例外になる deadlock
- 実 MossFormerGAN の勾配 norm。GPU 未使用
- `reproduction.duckdb` がこの作業木に無いための研究用 Dataset 登録
- 独立検証

## 対象

HEAD は `c0dc3c6672e9275133a7b5a6dce878eee4068c62` のままである。commit はしていない。作業木は dirty である。

## 差し戻し対応

検証文書は変更していない。ここでの単体テスト成功は独立検証の合格ではない。

### R1

直した。`MossFormerGANTask.validate_on_data_epoch_end` を真にした。`_run_data_epochs` は、この旗があるとき `global_step % validation_interval` では検証も plateau もしない。データ epoch の最後の micro-batch を処理したあと、最後の group が Optimizer を進めなくても、検証を行い `step_schedulers` にその損失を渡す。他モデルはこの旗を持たないので、従来の interval 契約のままである。

カウンタは分けたままである。処理した micro-batch、成功、失敗、optimizer 更新、`official_step`、`global_step`。epoch 末に余った成功勾配は、次 epoch の `start_epoch` が捨てる。公式も epoch の先頭で `accu_count` を戻す。

単体テストは `tests/training/test_mossformer_epoch_plateau.py`。蓄積 2、1 epoch 4 micro-batch、`validation_interval` 2。失敗なし、先頭失敗、末尾失敗の 3 件とも、epoch 末に検証値 0.5 が scheduler へ渡る。

### R2

直した。Discriminator の `broadcast_buffers` は True。`find_unused_parameters=True` は維持する。DDP が forward ごとにバッファを配るので、手動 broadcast は外した。

確認: 物理 GPU 1 と 2、`CUDA_LAUNCH_BLOCKING=1`、合成 32,000 サンプル。両 rank とも `broadcast=True` で、成功 micro-batch、PESQ 失敗、継続更新、そのあとの `validation_report` が戻った。これは検証担当が測った「3 forward のあと 2 backward が True で終わる」と同じ実行の再測定ではない。勾配 norm の True/False 比較は、このセッションではやり直していない。

### R3

直した。SyncBatchNorm へ変換するのは CUDA かつ world size が 2 以上のときだけである。CPU の world 2 は BatchNorm のまま DDP で包み、forward が通る。`tests/training/test_mossformer_syncbn_collective.py` の CPU world 2 は、Discriminator forward が例外にならず、SyncBatchNorm 数が 0 である。

### R4

テストを直した。学習の `zero_grad` は戻していない。`test_official_width_step` は step 後の空勾配を要求しない。重みが変わったことを、その step の証拠にする。

### R5

直したとは書かない。検証担当が記録した「実 batch、PESQ 強制分岐、合成の継続更新のあと、同じプロセスの検証で CUDA illegal memory access。Python 行は未特定」は、このセッションでは再現していない。

このセッションで実行したのは、物理 GPU 1 と 2、`CUDA_LAUNCH_BLOCKING=1`、合成 `(1, 32000)` である。順は、蓄積 2 の成功 1 件、PESQ を None にした 1 件、成功で窓を埋める 1 件、同じプロセスの `validation_report` 1 件。両 rank とも検証が戻り、rank 0 の generator 損失は 1.1569470167160034、rank 1 は 1.1678670644760132 だった。illegal memory access は出ていない。落ちた Python 行は、この再実行では特定できない。検証担当の実 Dataset batch と Trainer 経由の検証入口は、この合成 `validation_report` と同じ実行ではない。

勾配の rank 間平均は、非連続な `.grad` をそのまま `all_reduce` しないよう連続コピーしてから行う。これは R5 の修正として確認したものではない。

## 差し戻し後のテスト

```text
.venvs/se/bin/python -m pytest \
  tests/training/test_mossformer_epoch_plateau.py \
  tests/training/test_model_tasks.py::MossFormerTests::test_official_width_step \
  tests/training/test_mossformer_syncbn_collective.py \
  tests/training/test_mossformer_validation.py \
  tests/training/test_mossformer_plateau.py \
  tests/training/test_mossformer_sampler.py \
  tests/training/test_mossformer_pesq_skip.py \
  tests/training/test_mossformer_discriminator.py \
  -q --tb=line
```

この一括の途中結果は 27 passed、1 failed。失敗は `test_ddp_rank_disagreement_and_both_ranks_skip` で、CPU world 2 が SyncBatchNorm を持たないのに件数が 0 より大きいことを要求していた。断言を 0 に直したあとの再実行は 1 passed。epoch plateau は別実行で 4 passed。`test_official_width_step` と CPU world 2 の SyncBatchNorm テストは、上の 27 passed に含まれる。`tests/training` と `tests/reproduction` の全件は再実行していない。

GPU プローブは pytest ではない。上の R2 と R5 に書いた。負荷ジョブは GPU 1 と 2 だけ止め、終了後に同じ 2 枚へ戻した。

## R6

直した。対象は recipe 検証の `_validate` である。`rank_mod_fixed_v1` は、検証件数が world size 以上のときだけ使う。824 件を world 2 で割った rank 0 の列は 412 件のまま、公式 sampler と一致する。

件数が world size 未満のときは、公式 DistributedSampler の shuffle が完全なストライドを作れず、公式自身の長さ assert に落ちる。その分岐を公式の 824 件分割とは呼ばない。参加する全 rank が `range(count)` の同じ短い列を見る。forward 回数は揃う。CUDA の SyncBatchNorm では、この短い列も全 rank で同じ長さなので、揃わない forward は始めない。CPU では SyncBatchNorm が無いので rank 0 だけが forward し、`reduce_validation` の all_gather は全 rank が参加するときだけ行う。片 rank だけが例外や barrier に残らない。

1 件の TensorBoard スモークは、この短い列で検証記録まで戻すためのもので、公式 824 件の分割の再現ではない。MossFormer の epoch 末 plateau と、他モデルの `validation_interval` は変えていない。

```text
.venvs/se/bin/python -m pytest \
  tests/training/test_mossformer_sampler.py \
  tests/training/test_mossformer_validation.py \
  tests/reproduction/test_p3.py::OrderTests \
  -q --tb=short
```

16 passed。824 件・world 2 の rank 0 は 412 件のまま、`rank_order("rank_mod_fixed_v1")` と一致する。1 件・world 2 は公式 sampler が拒否するままで、recipe 側だけが両 rank に同じ `[0]` を返す。

```text
CUDA_VISIBLE_DEVICES= AUDIO_DB_TB_DDP_MODELS=mossformergan AUDIO_DB_TB_DDP_DEVICE=cpu \
  .venvs/se/bin/python -m pytest \
  tests/training/test_tensorboard_integration.py::TensorBoardDDPIntegration::test_other_real_models_two_ranks_validation \
  -q --tb=short
```

1 passed（815.30s）。GPU は使っていない。全件 pytest は再検証に残す。
