# P10-C Preflight Fix.1 独立検証

日付: 2026-11-07。検証担当。実装記録と単体テストの自己申告は、この文書の合格には使っていない。本番コード、テスト、YAML、`.agents`、既存の Issue 文書は変更していない。公式ソース `src/se/official/` は読んでいない範囲を変更していない。`resource.duckdb`、既存 snapshot、既存 checkpoint、`data/corpora`、`data/derived`、`data/generated`、`data/results` は削除も上書きもしていない。空の `reproduction.duckdb` は作っていない。学習用 WAV と波形キャッシュは作っていない。本格学習と公式 epoch 数までの学習はしていない。commit と push はしていない。

Preflight の PASS は、そのモジュールが見る起動条件の一致である。公式再現の成功ではない。

## 作業木

検証中にリポジトリ内の既存ファイルは変わっていない。増えたのはこの文書だけである。

開始時:

- ブランチ `feature/reproduction`
- `git rev-parse HEAD`: `c0dc3c6672e9275133a7b5a6dce878eee4068c62`
- `git status --short` は 23 パス。変更 19、未追跡 4

```text
 M .agents/DATA.md
 M .agents/DECISIONS.md
 M docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX_ISSUES.md
 M src/reproduction/db/registry.py
 M src/reproduction/db/schema.py
 M src/reproduction/db/validate.py
 M src/reproduction/ids.py
 M src/reproduction/pipeline/order.py
 M src/reproduction/pipeline/planner.py
 M src/reproduction/pipeline/training_data.py
 M src/reproduction/prepare/research_valid_fixed.py
 M src/training/core/trainer.py
 M src/training/se/mossformergan.py
 M src/training/se/run.py
 M tests/training/test_mossformer_official_init.py
 M tests/training/test_mossformer_pesq_skip.py
 M tests/training/test_mossformer_syncbn_collective.py
 M tests/training/test_mossformer_validation.py
 M tests/training/test_resume_identity.py
?? docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX1_IMPLEMENTATION.md
?? docs/training/reproduction/preflight/P10_C_PREFLIGHT_FIX1_PLAN.md
?? tests/reproduction/test_research_claim.py
?? tests/training/test_mossformer_sampler.py
```

終了時:

- HEAD は `c0dc3c6672e9275133a7b5a6dce878eee4068c62` のまま
- 上の 23 パスは同じ
- 未追跡にこの文書が増えている

## 総合判定

差し戻しが要る。MossFormerGAN の公式 1 GPU と 2 GPU は NO-GO。研究用 TF-Locoformer も NO-GO。MP-SENet 公式と TF-GridNet は、この作業で新しい数値不一致を見つけていないので CONDITIONAL GO のままにする。Preflight PASS を開始承認にはしない。

根拠の HEAD は `c0dc3c6672e9275133a7b5a6dce878eee4068c62` と、上の未コミット差分である。

## 公式コード比較

公式の比較元は `src/se/official/clearervoice-studio/train/speech_enhancement/solver.py`、同ディレクトリの `dataloader/dataloader.py`、`losses/loss.py`、`train.py`、`networks.py`。ビット一致は主張しない。測った差と、計画が残すと書いた差を分けた。

### 検証の rank 分割

公式 `DistributedSampler` を、そのクラス定義だけ実行して `rank_order("rank_mod_fixed_v1")` と比較した。shuffle 既定は True、seed は 0。`get_dataloader` は `shuffle` を渡さないので、検証も学習もこの既定である。`Solver.train` が `set_epoch` するのは `train_sampler` だけで、検証 sampler の epoch は 0 のままである。DataLoader の `drop_last` 既定は False。

一致した組は `(size, world, epoch)` の 12 組。`(8,2,0)`、`(10,3,0)`、`(11,4,0)`、`(5,2,0)`、`(824,2,0)`、`(11572,2,0)`、`(11572,1,0)`、`(824,1,0)`、`(10,3,1)`、`(824,2,5)`、`(7,2,0)`、`(9,4,2)`。不一致は 0。`shuffle=False` の公式列は、同じ N=10、world=3、epoch=0 で True の列と違った。現行は False 側を使っていない。

N=10、world=3、epoch=0、shuffle=True の実列は次である。

| rank | indexes |
|---|---|
| 0 | 6, 0, 3, 7 |
| 1 | 1, 4, 8, 2 |
| 2 | 5, 6, 0, 3 |

末尾 index 9 はどの rank にも無い。重複は 0、3、6 の 3 件。rank あたり 4 件。`drop_last=False` かつ batch 3 なら rank あたり 2 batch で、最後の batch は 1 件のまま分母に残る。VoiceBank の 11,572 と 824 は world 2 で割り切れる。末尾破棄 0、前方の繰り返し 0、rank あたり 5,786 と 412。検証 batch は protocol も公式 YAML も 1 なので、2 GPU の検証分母は rank あたり 412 batch である。

実 snapshot `mossformergan.vb28.ondemand.official@1` で、GPU 上の `validation_refs` は rank あたり 412 件、公式 sampler の epoch 0 と case id 列が一致した。epoch 引数 5 でも case id 列は epoch 0 と同じだった。crop は `per_epoch_random` なので epoch 5 と epoch 0 で開始位置が変わる。公式の切り出しは `random.randint` で、この crop 差は P10P-A-011 のままである。

world 1 の公式は DistributedSampler を付けず、`DataLoader(shuffle=True)` である。現行の world 1 は replicas=1 の sampler 列である。この作業の差分では、world 1 の検証列は変更前の連結結果と同じ 1 本である。

### 初期化

全件 pytest の `tests/training/test_mossformer_official_init.py` は失敗一覧に無い。このテストは、公式 `train.py` が残す G1 と D4 との `state_dict` 最大絶対差を 0.0 とし、Generator の BatchNorm 数を 0 とする。checksum `7729.488281` は使っていない。GPU 上で数えた現行モジュールは、変換前の Generator が BatchNorm 0、InstanceNorm 16、Discriminator が BatchNorm1d 2 と BatchNorm2d 6 である。`distribute` のあと Generator の BatchNorm は 0 のまま、Discriminator は SyncBatchNorm 8 になった。

### PESQ、蓄積、カウンタ

公式 `loss_mossformergan_se_16k` は、NaN または PESQ が None のとき `None, None` を返す。`_run_one_epoch_mossformergan_se_16k` は、その batch を backward せず `accu_count` も増やさない。`self.step` はそのあとで失敗 batch も含めて増える。窓が `effec_batch_size / batch_size` に達したときだけ両方の optimizer を step する。epoch 開始で `accu_count` を 0 にし、両方を `zero_grad` する。検証は `loss_gen is not None` のときだけ和に足し、戻り値は `total_loss / (i+1)` である。失敗 batch は分母に残る。足さないことは、和に 0 を足すのと同じ商になる。バッチ報告の値を 0 として保存する処理とは別である。

現行 1 GPU の窓継続は、失敗一覧に無かった `test_later_micro_batch_pesq_failure_does_not_step_the_group` と `test_later_success_completes_the_window_after_a_pesq_failure` が全件実行の中で通っている。GPU の 2 rank では、片 rank の PESQ を None にした group で両 rank が backward せず、重みが変わらなかった。rank 0 の理由は `peer`、rank 1 の理由は `pesq`。カウンタ増分は両 rank とも processed 2、succeeded 0、failed 2、optimizer_updates 0、official_step 2。両 rank 失敗も同じ増分で、重みは変わらなかった。両 rank 成功は processed 2、succeeded 2、failed 0、optimizer_updates 1、official_step 2 で、重みは変わった。成功した実音声 2 micro-batch は processed 2、succeeded 2、optimizer_updates 1、official_step 2、`update_applied` 1 だった。

`global_step` は Trainer が `update_applied == 1` のときだけ増やす。`official_step` は micro-batch ごとである。Scheduler 名 `none` の実タスクでは `sched_g` と `sched_d` が None だった。torch の scheduler 更新回数は 0 のままである。

Plateau は別である。実 `Trainer.fit` に、MossFormer と同じ窓規則のタスクを載せて数えた。1 epoch は micro-batch 4、蓄積 2、計画 optimizer 更新 2。`validation_interval` は 2。`max_steps` は 3。失敗が無いとき、epoch 末の global_step 2 で検証が走り、`step_schedulers` は検証値 0.5 を受け取った。

micro-batch 1 だけ失敗したとき、成功 3 件から optimizer 更新は 1 回になり、余った 1 成功は次 epoch の `start_epoch` が捨てた。その epoch 末の `step_schedulers` の引数は None だった。次の検証は次 epoch の途中、global_step 2 で走り、そこでは `step_schedulers` が呼ばれなかった。次の epoch 末は global_step 3 で、再び引数 None だった。最後の micro-batch だけ失敗したときは、最後の group が `update_applied` 0 になり、epoch 末の `step_schedulers` 自体が呼ばれなかった。公式は epoch 末に必ず検証し、plateau をその損失で更新する。現行の MossFormer `step_schedulers` は validation が無いと plateau の前で戻る。失敗 1 件で、計画更新数と `validation_interval` の位相がずれ、以降の epoch 末が検証点に戻らない。

2 GPU の全 rank スキップは、失敗した micro-batch をやり直さない。成功側の rank もその micro-batch を backward しない。公式の片 rank だけスキップすると、もう一方が DDP の backward collective に残る。全 rank スキップはそれを避ける代わりに、成功側の勾配を捨て、両 rank の `accu_count` をその件数だけ遅らせる。epoch 末に窓が半端なら、公式と同じく次 epoch の開始でその勾配は捨てる。公式と違うのは、成功側の件数まで落ちることと、上の plateau が epoch 末の検証を見なくなることである。

### DDP、SyncBatchNorm、clip

公式 Solver は Generator と Discriminator の両方を `SyncBatchNorm.convert_sync_batchnorm` し、`find_unused_parameters=True` の DDP にする。`broadcast_buffers` は指定が無く、既定 True である。損失関数は Discriminator を 3 回 forward してから、generator 損失と discriminator 損失を順に backward する。

現行のラップを GPU で読んだ。Generator は `find_unused_parameters=True`、`broadcast_buffers=True`。Discriminator は `find_unused_parameters=True`、`broadcast_buffers=False`。フラグは属性に入っている。この実 step の forward では未使用パラメータは 0 で、DDP は `find_unused_parameters=True` だが未使用は見つからなかった、と警告した。Generator の勾配を持つパラメータは 1,053、Discriminator は 93 だった。

`broadcast_buffers=True` が保存済み BatchNorm 状態を壊すかは、コメントではなく実行した。小さな SyncBatchNorm ネットで 3 回 forward してから backward すると、True も False も backward が終わり、勾配 norm はどちらも 6.719967365264893、`running_mean[0]` はどちらも -0.04393921419978142 だった。実 Discriminator のコピーで、公式と同じく 1 回目の forward のあと 2 回目と 3 回目を行い、1 回目の損失を backward してから残りを backward した。True も False も例外は無かった。勾配 norm は True が 1.936649203300476、False が 1.8368158340454102 で違った。`running_mean` の先頭要素はどちらも -0.059167541563510895 だった。True で autograd が落ちることは、この入力では再現しなかった。False は公式既定と数値が違う。

clip は、成功 backward のあと増分を SUM して world size で割り、走っている和へ足してから `clip_grad_norm_` する。公式は `no_sync` が無く、蓄積済み勾配も含めて backward のたびに平均し直してから clip する。この再平均は現行がコピーしていない。実音声の 2 micro-batch で、clip 直前の norm は Generator 45.13755767847905 と 56.32420063585707、Discriminator 1.0868246749268693 と 1.9465334760124042 だった。Generator は閾値 10 を超える。この step では clip は恒等変換ではない。学習全体で何回超えるかは、この 2 micro-batch の外は測っていない。

### Optimizer、Scheduler、Checkpoint

公式に epoch 単位の torch scheduler は無く、検証損失の plateau が Generator の学習率を半分にする。現行 YAML の scheduler 名は `none`。上のとおり、失敗が無い epoch では plateau が epoch 末の検証を見る。失敗があると見ない。

Checkpoint の `global_step` は optimizer 更新回数である。`official_step` は task の `accumulation.counters` に分かれて保存される。`experiment_profile == "full"` の再開は、`execute(..., strict_resume=False)` でも logging 以外を照合する。CLI の `--resume` は `strict_resume=True` である。実 checkpoint の設定不一致再開は、このセッションでは実行していない。

## GPU-DDP

物理 GPU 1 と 2 を使った。公式 epoch は回していない。簡易 all_gather の CPU 代替は、SyncBatchNorm の PASS に使っていない。

| 項目 | 結果 |
|---|---|
| 公式幅の 2 GPU DDP 起動 | 起動した。visible device 上の local rank 0 と 1 |
| 実 Dataset の batch | `mossformergan.vb28.ondemand.official@1` から取得した。形状 `(1, 32000)` が 2 件。合成入力ではない |
| Generator と Discriminator の更新 | 実 batch で `update_applied` 1。両 rank の clip 前 norm は一致 |
| 正常 PESQ | 実 batch は PESQ 失敗にならず、optimizer が 1 回進んだ |
| 片 rank だけ PESQ 失敗 | 両 rank が backward しない。重み不変。deadlock にはならなかった |
| 両 rank の PESQ 失敗 | 両 rank が backward しない。重み不変 |
| rank 分割された検証 | 別プロセスで 1 batch ずつ。rank 0 は case `f58a4396...`、損失 1.2719812393188477。rank 1 は case `5ef5ec31...`、損失 1.244946837425232。分母は `[1, 1]`。412 件の case 列は公式 epoch 0 と一致 |
| SyncBatchNorm を含む forward | 検証 forward のあと、SyncBatchNorm `running_mean` の先頭 4 要素が両 rank で一致。値は 0.0003279687953181565、-0.0300550889223814、-0.14138595759868622、-0.05775193125009537 |
| Checkpoint 保存 | `/tmp` に task state を書いた。rank 0 のファイルは 136,460,302 バイト |
| 別プロセスで再開 | 2 プロセスを新しく起動し、保存 state を DDP の前に読み、同じ seed の合成 2 micro-batch を 1 回更新した。例外は無かった |
| 中断なしランとの比較 | スカラー損失は再開側も 0.5940971076488495 と 1.995571494102478 で、中断なし側の表示と一致した。Generator の最大絶対差は 0.00023778752074576914、パラメータは `blocks.0.attn_conv_K_3.2.beta`。`deterministic: false` での差であり、ビット一致ではない |

更新のあとに同じプロセスで検証へ進んだ実行は、2 回とも CUDA の illegal memory access でプロセスが落ちた。Python の例外行は残らず、NCCL watchdog が SIGABRT した。検証だけの新しいプロセスでは同じ 1 batch 検証が終わった。落ちた箇所は、更新列のあとの検証入口である。824 件の検証 forward はしていない。

公式 2 GPU 学習との損失一致、公式 step との勾配一致は NOT_TESTED。

## 全体回帰

```text
.venvs/se/bin/python -m pytest tests/reproduction tests/training -q --tb=line -p no:cacheprovider --basetemp=/tmp/pytest-p10c-fix1
```

1645.37 秒。結果は 389 passed、17 failed、1 error、12 skipped、39 subtests passed。skip は成功に数えていない。12 件の skip 名は、この実行の `-q` 出力には残っていない。

失敗の切り分け:

| テスト | 分類 |
|---|---|
| `test_p4.py` の provenance | 環境。`data/db/reproduction.duckdb` が無い |
| `test_p8_dfn3.py` 3 件 | 環境。`reproduction-p8.duckdb` と `dfn3.online.smoke@1` の snapshot が無い |
| DeepFilter の kernel、model task 3 件、TensorBoard の deepfilternet3 | 環境。`libdf_rfft.so` が `GLIBC_2.33` を要求し、このホストの libc が満たさない |
| `test_p1.py::test_staging_rejects_different_content` の teardown ERROR | 環境。期待する `data/db` の名前一覧に、16:48 からある空の `.reproduction.lock` が入っている。この検証が作ったファイルではない |
| `MossFormerTests::test_official_width_step` | この差分。`train_update` が窓完了後に `zero_grad` し、テストは戻ったあとの勾配を要求する。重みが変わらない、という手前の断言には達していない |
| `test_ddp_rank_disagreement_and_both_ranks_skip` と SyncBN collective 2 件 | この全件実行の spawn が `No module named 'tests.training'` で終了コード 1 になった。ロジック断言の前に落ちている。同じ PESQ 分岐は GPU で別途実行した |
| TensorBoard の mossformergan 2 rank | この差分。CPU の world 2 で Discriminator が SyncBatchNorm になり、forward が `SyncBatchNorm expected input tensor to be on GPU` で落ちた |
| `test_preview_failure_preserves_training_state` | 失敗はした。記録された例外は `LinearPairTask` に `_generate` が無い、である。`preview.py` はこの作業木の差分に無い。Fix.1 がメッセージを変えたとは断定しない |

`test_mossformer_official_init.py`、研究 claim、sampler、1 GPU の窓継続、resume identity の変更テストは失敗一覧に無い。

## Dataset 監査

正本として見たパスは次の 3 つだけである。どれもファイルが無い。

- `data/db/reproduction.duckdb`
- `data/db/backups/reproduction.duckdb`
- `data/staging/reproduction/reproduction.duckdb`

`data/staging/reproduction/INDEX.json` も無い。`register_research_validation_dataset` はデータベースファイルが無いとき、ファイルを作らず拒否する。登録は BLOCKED。子 snapshot は発行していない。

広い探索はしていない。`data/db/snapshots` の直下には、親の dataset id ディレクトリがある。

- 親 `tf_locoformer.dns.accept_recipe@1`
- dataset id `5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe`
- manifest の snapshot hash `3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5`
- 件数は train 324,000、valid 36,000、test 150。各集合の case id は一意。valid と train、valid と test、train と test の交差は 0
- `selection_uses_test` は false
- 注記は、36,000 件が `cv_synthetic_small` ではなく、その名前は UNKNOWN だと書いている

実装記録が「親 hash のディレクトリが無い」と書いたのは、snapshot hash をディレクトリ名として探した結果である。中身は dataset id ディレクトリの manifest にある。親 36,000 件は、この検証の開始時と終了時で `dataset.json` 4,651 バイト、`manifest.json` 1,029 バイト、`case_set_items.npz` 60,507,030 バイト、mtime は 2026-11-07 09:30 のままである。

研究用集合を、親 valid 36,000 件から `select_research_validation` で再計算した。規則 `research_valid_fixed_v1`、seed `20261107`、1,024 件、除外 0。公開リストと case id 列が一致した。SHA-256 は `73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4`。train と test への交差は 0。`leakage_safe` は false。`cv_synthetic_small` ではない。claim は `research_checkpoint_selection`。リスト JSON の blockers には、claim が CHECK に無い、という古い文が残っている。現行の `CLAIM_LABELS` と `repro.datasets` の CREATE TABLE にはこのラベルがある。JSON は再生成されていない。登録関数はその文を読まない。

一時 DB の親不変と冪等は、失敗一覧に無かった `tests/reproduction/test_research_claim.py` が全件実行の中で通っている。本番 1,024 件の登録ではない。本番 DB が無いので、本番行での冪等は BLOCKED のままである。

他の snapshot 件数は、train / valid / test の順に、MP-SENet official 11,572 / 824 / 824、MossFormerGAN official 11,572 / 824 / 824、TF-GridNet 7,345 / 1,835 / 305。safe の MP-SENet と MossFormerGAN、研究用子 Dataset の snapshot は 0 件だった。

## Checkpoint 監査

GPU の再開は、別プロセスが state を読み、1 更新した。読み込み例外は無かった。中断なし実行との重み差は上の 2.38e-4 である。`deterministic: false`。公式 checkpoint 形式との一致は見ていない。

保持、`validation_pending`、原子的 replace の既存テストは、失敗一覧にそれらの名前が無い。実在の `data/checkpoints` と既存 run は削除していない。検証の途中 batch から再開する機構は、コード上も無い。`validation_pending` の再開は検証を先頭からやり直す、という既存契約は、このセッションで GPU の中断検証としては実行していない。

`full` の `strict_resume=False` 照合は `src/training/se/run.py` にある。不一致の実 checkpoint 再開は NOT_TESTED。

## Preflight

`src.training.se.preflight.run_preflight` を、出力先 `/tmp` で実行した。学習は始めていない。PASS は公式再現の成功ではない。

| 対象 | world | 結果 | 内容 |
|---|---:|---|---|
| MP-SENet official | 2 | PASS | snapshot `265f256b...`、11,572 / 824 / 824 |
| MP-SENet leakage-safe | 2 | FAIL | snapshot 0。data gap |
| MossFormerGAN official 1 GPU | 1 | PASS | snapshot `ba0c77c9...`、11,572 / 824 / 824 |
| MossFormerGAN official 2 GPU | 2 | PASS | 同じ snapshot。DDP の差はこの検査に出ない |
| MossFormerGAN leakage-safe | 1 | FAIL | snapshot 0。data gap |
| TF-GridNet full | 2 | PASS | snapshot `a69a477a...`、7,345 / 1,835 / 305 |
| TF-Locoformer 研究用 full | 4 | FAIL | `research_valid_fixed@1` の snapshot 0。data gap。PASS にはしていない |

safe と研究用の FAIL は、コードが dataset を READY と誤認した失敗ではない。snapshot が無い。

## NOT_TESTED

- 公式 2 GPU 学習との損失、重み、勾配の数値一致
- 公式 epoch 数、121 epoch、400 epoch、150 epoch
- 実勾配 norm が 10 を超える頻度。測ったのは実音声 2 micro-batch で、Generator は超えた
- 別の入力長で未使用パラメータが出るか。この 32,000 サンプルの forward では 0
- VoiceBank 824 件の検証 forward と、その PESQ 失敗件数
- 824 件の PESQ 失敗が分母に残る実波形での集計
- 実 checkpoint の設定不一致再開
- 更新のあとに同じプロセスで検証する経路。2 回落ち、原因の行は特定していない
- 中断なし実行とのビット一致。差は 2.38e-4 まで測った
- `tests` の 12 skip の内訳
- 本番 `reproduction.duckdb` の行と、親 36,000 件の DB 上の不変。snapshot ファイルは比較した
- MP-SENet の公式 magnitude 比較の再実行。このセッションではしていない

## 差し戻し

修正はしていない。

### R1. PESQ 失敗で epoch 末の plateau が検証を見ない

- 再現: 蓄積 2、1 epoch 4 micro-batch、`validation_interval` 2 の `Trainer.fit` で、1 件目または 4 件目を失敗させる。失敗が無い対照では epoch 末に検証値 0.5 が scheduler へ渡る。失敗があると、epoch 末は None か、呼び出し自体が無い。次の検証は epoch の途中で、plateau は呼ばれない
- 対象: `src/training/core/trainer.py` の `_run_data_epochs`、`src/training/se/mossformergan.py` の `step_schedulers`
- 実際: optimizer 更新数が計画から 1 減ると、`global_step % validation_interval` が epoch 末とずれる。最後の group が step しないと、epoch 末処理ごと通らない。余った成功勾配は次 epoch の `start_epoch` が捨てる
- 期待: 公式はデータ epoch の末で必ず検証し、その損失で plateau を更新する。失敗 batch でも `self.step` は増える
- 根拠: `solver.py` の `train` と `_run_one_epoch_mossformergan_se_16k`
- 深刻度: HIGH
- 修正が要る理由: 1 件の PESQ 失敗で、以降の epoch 末が学習率半減の判定から外れる。1 GPU の公式再現にも効く

### R2. Discriminator の `broadcast_buffers=False` は、壊れることの実証になっていない

- 再現: 実 Discriminator を SyncBatchNorm にし、DDP の `broadcast_buffers` を True と False で、3 forward のあと 2 回 backward する
- 対象: `src/training/se/mossformergan.py` の `distribute` と `train_update`
- 実際: True も backward が終わる。勾配 norm は True 1.936649203300476、False 1.8368158340454102。小さな SyncBatchNorm では両方 6.719967365264893 で一致した
- 期待: 公式は既定 True で、同じ 3 forward のあと backward する。True が安全なら公式のフラグにする。False を残すなら、測った数値差として 2 GPU を GO にしない
- 根拠: `solver.py` の DDP 構築と `loss_mossformergan_se_16k`
- 深刻度: HIGH（2 GPU の数値）。1 GPU には出ない
- 修正が要る理由: 「True が保存済み状態を壊す」は、この実行では例外として現れなかった。False は公式と勾配が違う

### R3. CPU の world 2 で Discriminator forward が SyncBatchNorm に拒否される

- 再現: 全件 pytest の TensorBoard 2 rank、model `mossformergan`
- 対象: `src/training/se/mossformergan.py` の `distribute`
- 実際: world 2 で SyncBatchNorm に変換したあと、CPU では DDP で包む前に戻る。forward が `SyncBatchNorm expected input tensor to be on GPU or privateuseone` になる
- 期待: CPU の複数 rank テストが、変換前の BatchNorm か、forward できる形で終わる
- 根拠: torch 2.6 の `SyncBatchNorm.forward`。公式の学習プロセスは CUDA である
- 深刻度: MEDIUM
- 修正が要る理由: 既存の CPU DDP テストが、今回の変換で落ちる

### R4. 公式幅 1 step のテストが、step 後の勾配を要求する

- 再現: `tests/training/test_model_tasks.py::MossFormerTests::test_official_width_step`
- 対象: `src/training/se/mossformergan.py` の `train_update` と、このテストの `_updated`
- 実際: `AssertionError: generator gradients are missing or non-finite`。窓が埋まったあと `zero_grad` している
- 期待: テストが、公式と同じく step 後に勾配が空であることを受け入れるか、backward 直後を見る
- 根拠: 公式も step のあと `zero_grad` する
- 深刻度: LOW
- 修正が要る理由: 全件の失敗として残っている。学習手続きを戻す種類の失敗には見えない

### R5. 更新のあとに同じプロセスで検証へ進むと GPU プロセスが落ちた

- 再現: 物理 GPU 1 と 2 で、実 batch の更新、PESQ の強制分岐、合成の継続更新のあと、検証 1 batch へ進む。検証だけの新しいプロセスでは落ちない
- 対象: `src/training/se/mossformergan.py` の `train_update` と `validation_report`。落ちた Python 行は特定していない
- 実際: CUDA illegal memory access。NCCL watchdog が SIGABRT。検証 JSON は書かれない
- 期待: 学習のあとの検証 forward が戻る
- 根拠: 公式の epoch は学習のあと検証する。この落ち方は公式の数値差とは別の異常終了である
- 深刻度: MEDIUM
- 修正が要る理由: 2 回再現し、検証だけの実行では再現しなかった。原因を特定しないまま 2 GPU を GO にできない

## モデル別の開始判定

| モデル | 判定 | 根拠 |
|---|---|---|
| MP-SENet official | CONDITIONAL GO | この差分の失敗一覧に MP-SENet の断言は無い。Preflight は PASS。このセッションで公式 magnitude は再測定していない。PASS は公式再現の成功ではない |
| MP-SENet leakage-safe | NO-GO | snapshot が無く Preflight は FAIL |
| MossFormerGAN official 1 GPU | NO-GO | 初期化テストは通り、PESQ の窓とカウンタは公式の 1 GPU に合う。R1 の plateau ずれが、失敗 1 件で epoch 末の学習率判定を外す |
| MossFormerGAN official 2 GPU | NO-GO | Preflight は PASS。R1 に加え、`broadcast_buffers=False` の勾配差、全 rank スキップ、clip が実 step で 10 を超える。更新後の検証は同じプロセスで落ちた。公式学習との数値一致は NOT_TESTED |
| MossFormerGAN leakage-safe | NO-GO | snapshot が無く Preflight は FAIL |
| TF-GridNet full | CONDITIONAL GO | Preflight は PASS。件数は 7,345 / 1,835 / 305。固定 `mix_seed` は既知の差。このセッションで公式 WAV との一致は測っていない |
| TF-Locoformer 研究用 1,024 | NO-GO | 定義と再計算は一致した。`reproduction.duckdb` が無く登録は BLOCKED。Preflight は FAIL。`leakage_safe` でも `cv_synthetic_small` でもない |
| TF-Locoformer の 36,000 を毎 epoch | NO-GO | 親 snapshot の 36,000 件は残っている。選択方式としては始めない |

本格学習は開始しない。

## 再検証（差し戻し R1–R5）

日付: 2026-11-07。実装報告は合格にしていない。本番コード、テスト、YAML、`.agents` は変更していない。追記したのはこの節だけである。本格学習はしていない。空の `reproduction.duckdb` は作っていない。

開始時と終了時の HEAD はどちらも `c0dc3c6672e9275133a7b5a6dce878eee4068c62`。`git status --short` は開始時と終了時で同じである。開始時点の 23 パスに加え、この文書、`tests/training/test_model_tasks.py` の変更、未追跡の `tests/training/test_mossformer_epoch_plateau.py` がある。検証中にそれらの中身は変えていない。

### R1. 解消

`MossFormerGANTask.validate_on_data_epoch_end` は True だけである。`Trainer._run_data_epochs` はこの旗があり、かつ `validation_interval` が 0 でないときだけ、データ epoch の末で `_validate_at_data_epoch_end` を呼ぶ。その関数が検証のあと `step_schedulers` に検証損失を渡す。最後の group が optimizer を進めなくても、`data_epoch_end` なら `continue` せずこの経路に入る。

`_run_epochs`、chunk 経路、旗の無い `_run_data_epochs` の else は、従来の `global_step % validation_interval` のままである。旗を立てているタスクは MossFormerGAN だけである。

`tests/training/test_mossformer_epoch_plateau.py` は 4 件とも passed。このテストのタスクは最初の scheduler 呼び出しで `stop_training` を立てるので、次の epoch は見ていない。別の実行で、同じ Trainer に旗を付け、失敗を epoch 0 だけに置き、`stop_training` を立てずに 3 回の epoch 末まで進めた。先頭 micro-batch の失敗でも、末尾 micro-batch の失敗でも、scheduler が受け取った値は `[0.5, 0.5, 0.5]` で、検証 epoch は 0、1、2 だった。次の epoch 末は外れていない。旗を False にした対照は、失敗があると epoch 末の scheduler 引数が None のままである。他モデルの interval 契約は変えていない。

公式 YAML の `validation_interval` は 0 ではない。0 にするとこの経路は無効になり、旧い interval へ戻る。現行の公式 YAML はその値ではない。

### R2. 解消

`distribute` は Generator と Discriminator の両方を `broadcast_buffers=True`、`find_unused_parameters=True` で `wrap_module` する。`_broadcast_discriminator_buffers` はソースに無い。

物理 GPU 1 と 2、`CUDA_LAUNCH_BLOCKING=1` で、公式幅の Discriminator を SyncBatchNorm 8 の DDP にした。属性は両 rank とも `broadcast_buffers=True`、`find_unused_parameters=True`。Generator の SyncBatchNorm は 0。ランダムな `(1, 1, 201, 40)` で 3 forward のあと 2 回 backward し、両 rank とも戻った。勾配 norm は rank 0 が 1.8556146621704102、rank 1 が 1.8556159734725952。公式の DDP 既定 True と、このフラグは一致する。

残る公式差はフラグではない。`no_sync` の増分平均、片 rank 失敗時の全 rank スキップ、`global_step` と公式 `self.step` の分離である。公式 2 GPU 学習との数値一致は NOT_TESTED のままである。

### R3. 解消

SyncBatchNorm への変換は `world_size > 1` かつ device が CUDA のときだけである。CPU は BatchNorm のまま DDP で包む。

`test_both_ranks_run_syncbn_forward` は passed。CPU world 2 で Discriminator の forward が戻り、SyncBatchNorm 数は 0、`broadcast_buffers` と `find_unused_parameters` は True だった。CUDA world 2 の再実行では Discriminator の SyncBatchNorm が 8 のままである。

既存の TensorBoard mossformergan 2 rank は通らなかった。理由は R3 ではない。下の R6。

### R4. 解消

`tests/training/test_model_tasks.py::MossFormerTests::test_official_width_step` は passed。`train_update` は窓が埋まったあと `zero_grad` を残している。このテストだけ `require_gradients=False` で、step 後の空勾配を失敗にしていない。他モデルの `_updated` は勾配要求の既定 True のままである。

### R5. NOT_REPRODUCED

直していない、という実装の記述を修正済みにはしない。前回落ちたのは、実 Dataset の batch で更新したあと同じプロセスで検証へ進んだときである。

同じ形を `CUDA_LAUNCH_BLOCKING=1`、物理 GPU 1 と 2、公式幅で再実行した。Dataset は `mossformergan.vb28.ondemand.official@1`。rank 0 の学習 batch は `(1, 32000)` が 2 件、case `1664c3fe473bbb2b5914685c9d47c10fffd044ee41117d1426271761bb3ad109` と `4e0814eb6a79ca84b6e339c1771cebb3a119617a50460479eabbdf5c2a17da76`。`update_applied` は 1。続けて同じプロセスで検証 1 batch、形状 `(1, 32000)`、case `f58a4396f4500ef62c7fe2668d7f5b4fd29ee8c05da8001fdc82de0867d47223`。`validation_report` は両 rank とも戻り、generator は `ok`、値は rank 0 で 0.6692999005317688、分母は `[1, 1]`。illegal memory access は再現しなかった。824 件の検証はしていない。この 1 回で 2 GPU を GO にはしない。

### 実行したテスト

関連 16 件は 321.21 秒で 16 passed、0 failed。

- `test_mossformer_epoch_plateau.py` 4
- `test_official_width_step` 1
- `test_mossformer_pesq_skip.py` 5
- `test_mossformer_syncbn_collective.py` 3
- `test_mossformer_validation.py` 3

TensorBoard の mossformergan 2 rank は別に 607.87 秒で SUBFAILED。全件 pytest は再実行していない。

### R6. 差し戻し。検証 1 件を world 2 の rank 分割に入れる

- 再現: `AUDIO_DB_TB_DDP_MODELS=mossformergan`、`AUDIO_DB_TB_DDP_DEVICE=cpu` で `TensorBoardDDPIntegration::test_other_real_models_two_ranks_validation`。検証セットは 1 件、world は 2
- 対象: `src/training/core/trainer.py` の `_validate`。`reduce_validation` があり `validation_order` が `rank_mod_fixed_v1` で world が 2 以上のとき、`rank_order` を呼ぶ
- 実際: `rank_mod_fixed_v1 cannot assign 1 samples across world_size 2`。rank 0 が落ち、rank 1 は barrier で接続が切れる。CPU の SyncBatchNorm 拒否ではない
- 期待: この既存の 2 rank スモークが、学習 2 step と検証記録まで戻る
- 公式との根拠: 公式 DistributedSampler も `floor(N/world)` が 0 の shuffle では空列になり、`len(indices) == total_size` の assert に達する。公式の VoiceBank 検証 824 件はこの分岐に入らない
- 深刻度: LOW
- 修正が要る理由: R3 を外したあとに、既存テストが別の例外で止まっている。本番の 824 件検証とは別の経路である

### 再検証後の開始判定

| モデル | 判定 | 根拠 |
|---|---|---|
| MP-SENet official | CONDITIONAL GO | この再検証の失敗一覧に MP-SENet の断言は無い。公式 magnitude は再測定していない |
| MP-SENet leakage-safe | NO-GO | snapshot 不在は変えていない |
| MossFormerGAN official 1 GPU | CONDITIONAL GO | R1 と R4 は解消した。crop の乱数源と、`global_step` を公式 `self.step` と別にする差は残る。Preflight PASS は再実行していない |
| MossFormerGAN official 2 GPU | NO-GO | R2 と R3 のフラグは公式の CUDA DDP に揃えた。公式 2 GPU 学習との数値一致は NOT_TESTED。全 rank スキップと増分平均は残る。R5 は 1 回再現しなかっただけで、2 GPU の GO にはしない |
| MossFormerGAN leakage-safe | NO-GO | snapshot 不在は変えていない |
| TF-GridNet full | CONDITIONAL GO | この再検証では新しい数値不一致を見ていない。固定 `mix_seed` は既知の差 |
| TF-Locoformer 研究用 1,024 | NO-GO | 登録は BLOCKED のまま。空の `reproduction.duckdb` は作っていない |
| TF-Locoformer の 36,000 を毎 epoch | NO-GO | 選択方式としては始めない |

本格学習は開始しない。

物理 GPU 1 と 2 を使った。負荷ジョブは再開した。GPU 6 と 7 は止めていない。

## 再検証（R6）

日付: 2026-11-08。実装報告の 16 passed と TensorBoard の 1 passed は、この節の合格にしていない。本番コード、テスト、YAML、`.agents` は変更していない。GPU は使っていない。負荷ジョブは止めていない。本格学習はしていない。

開始時と終了時の HEAD はどちらも `c0dc3c6672e9275133a7b5a6dce878eee4068c62`。`git status --short` は開始時と終了時で同じである。上の再検証節に書いたパスに加え、未追跡のこの文書だけが検証担当の追記先である。

### R6. 解消

824 件を world 2 に分ける経路は `rank_mod_fixed_v1` のままである。公式 `DistributedSampler`（shuffle 既定 True、epoch 0、seed 0）を実行し、`recipe_validation_positions(824, 2, rank)` と比べた。rank 0 も rank 1 も 412 件で、公式の列と一致した。末尾破棄 0、前方の繰り返し 0、評価に入る枠は 824、重複 0 である。`distributed_sampler_coverage` も同じ件数である。1 件・world 2 の `rank_order("rank_mod_fixed_v1")` は、従来どおり `cannot assign` で拒否する。recipe 側の `recipe_validation_positions` だけが両 rank に同じ `[0]` と `undersized_same_list` を返す。

1 件・world 2 の collective はコードで読んだ。TensorBoard の 13 分テストは再実行していない。

CPU では `validation_requires_all_ranks` が偽なので、`_validate` に入って forward するのは rank 0 だけである。`reduce_validation(..., collective=False)` は all_gather しない。rank 1 は forward も all_gather もせず、関数末尾の `barrier` で rank 0 と揃う。rank 0 はこの 1 件で例外を出さない。

CUDA かつ SyncBatchNorm では `validation_requires_all_ranks` が真なので、両 rank が `recipe_validation_positions` に入る。件数が world 未満のときは同じ短い列になり、forward の前に `_match_validation_forwards` が件数を all_gather する。件数が揃っていればそのまま、0 件の rank があれば forward の前に全 rank で例外にする。揃わない非ゼロの差は、分母に入れない padding forward で回数を揃えてから forward する。そのあと `reduce_validation(..., collective=True)` が全 rank で all_gather する。件数が world 未満でも、回数が揃う前の forward は始めない。

`_validate_at_data_epoch_end` は残っている。データ epoch の末で検証し、`step_schedulers` に検証損失を渡す分岐は外していない。`validate_on_data_epoch_end` は MossFormerGAN だけが True である。この確認はコード読みで、plateau テストは再実行していない。

実装記録と `recipe_validation_positions` の注記は、1 件の短い列を公式 824 件の DistributedSampler 分割とは書いていない。検証文書の誤記もない。

再実行したテストは 6 passed、4.44 秒である。`tests/training/test_mossformer_sampler.py` の 5 件と、`tests/reproduction/test_p3.py::OrderTests::test_rank_mod_matches_the_official_sampler_formula` の 1 件。validation テストと TensorBoard は NOT_REEXECUTED。

さらなる差し戻しは無い。モデル別判定は変えない。MossFormerGAN の公式 2 GPU は NO-GO のままである。公式 2 GPU 学習との数値一致は NOT_TESTED。研究用 Dataset の登録は BLOCKED のままである。

GPU は未使用。負荷ジョブは止めていない。確認した時点で PID 2418233、2418234、662476、4109908 は生きていた。
