# P10-C Preflight Agent D: DDP・Sampler・Validation

発見者は Agent D。結論は 2026-11-07 に読んだ現行コードと、CPU 上の純関数・gloo 再現だけに基づく。過去監査、`DECISIONS.md`、他エージェントの preflight 文書は根拠にしていない。NCCL は起動していない。実 GPU ジョブ、本格学習、DB、checkpoint、本番コードの変更はしていない。

実行は `.venvs/official` の Python（torch、typeguard あり）。スクリプトは `/tmp/p10p_agent_d.py`、`/tmp/p10p_clip.py`、`/tmp/p10p_nosync.py`、`/tmp/p10p_syncbn.py`。gloo は world size 2、127.0.0.1 の短時間プロセス。SyncBatchNorm の数値比較は、CPU の DDP が `SyncBatchNorm layers only work with GPU modules` で拒否したため NOT_TESTED。

## 読んだ関数

現行:

- `src/training/core/distributed.py`: `setup_process`, `wrap_module`, `barrier`
- `src/training/core/batch.py`: `plan_batches`
- `src/training/data/sampler.py`: `epoch_indices`
- `src/training/data/chunk_plan.py`: `epoch_chunk_batches`, `select_utterances`
- `src/reproduction/pipeline/chunking.py`: `plan_chunk_epoch`, `iterate_chunks`, `select_source_ids`
- `src/reproduction/pipeline/order.py`: `rank_order` と各 policy
- `src/reproduction/pipeline/planner.py`: `plan_fixed_pair`, `validation_refs`
- `src/reproduction/pipeline/training_data.py`: `epoch`, `validation`
- `src/reproduction/pipeline/online_dfn3.py`: `plan_online_epoch`, `_shard`, `validation`
- `src/reproduction/pipeline/online_semamba.py`: `validation`（学習側の重なりは `per_rank_independent_v1`）
- `src/training/core/trainer.py`: `accumulation_scope`, `fit`, `_run_epochs`, `_run_data_epochs`, `_run_chunk_epochs`, `_run_aligned_chunk_epochs`, `_validate`, `_validate_training_data`, `_validate_aligned_chunks`, `_validate_chunks`, `_reduce_weighted_reports`, `_aggregate_validation`, `_combine_metric`, `_scheduler_validation`, `_chunk_should_stop`, `_broadcast_plan_hash`, `_save`
- `src/training/se/run.py`: `execute` の `plan_world`（`virtual_world_size` を batch plan の world に使う）
- `src/training/se/mp_senet.py`: `distribute`, `_official`, `validation_report`
- `src/training/se/mossformergan.py`: `distribute`, `train_update`, `eval_mode`, `validation_report`, `state_dict`
- `src/training/se/tf_gridnet.py`, `tf_locoformer.py`: `train_update`, `validation_report`
- `src/training/se/deepfilternet.py`: `distribute`（`find_unused_parameters=True`）, `train_update`, `validation_report`
- `src/training/se/semambapp.py`: `distribute`, `train_update`
- `src/training/se/mossformer_discriminator/__init__.py`: `AdaptCNN` の `BatchNorm2d` 6 個
- `src/se/models/mossformergan/`: Generator は LayerNorm 系で running stats を持たない
- `src/se/models/deepfilternet3/model.py`: `df_fc_a` は定義のみで forward から呼ばれない

公式:

- `src/se/official/espnet/espnet2/iterators/chunk_iter_factory.py`: `build_iter`, `_generate_mini_batches`
- `src/se/official/espnet/espnet2/iterators/sequence_iter_factory.py`: `build_iter` の `num_iters_per_epoch`
- `src/se/official/espnet/espnet2/tasks/abs_task.py`: `build_chunk_iter_factory`（`batches[rank::world_size]`、cache の `// world_size`、余り batch は低い rank）
- `src/se/official/espnet/espnet2/train/trainer.py`: `train_one_epoch` / `validate_one_epoch` の `iterator_stop`
- `src/se/official/espnet/espnet2/enh/espnet_model.py`: loss weight は `speech_ref[0].shape[0]`
- `src/se/official/clearervoice-studio/train/speech_enhancement/solver.py`: SyncBatchNorm、DDP、`model.eval()` のみ、accumulation 中の clip、検証分母 `(i+1)`
- 同 `losses/loss.py`: `loss_mossformergan_se_16k`（PESQ 失敗は `None, None`）
- 同 `dataloader/dataloader.py`: `DistributedSampler.__iter__`、非分散時は `shuffle=True`

## 実行した確認

| 確認 | 結果 | 状態 |
|---|---|---|
| `epoch_indices` world 1/2/4、size 1/7/8/10/11 | 重複 0。余りは全 rank で同数だけ欠ける。epoch で欠ける index は変わる | CONFIRMED |
| `plan_batches` | `micro * world * accum == global`。割り切れない global は拒否 | CONFIRMED |
| `rank_mod_fixed_v1` と公式 Sampler の写経 | size 5/8/11、world 1/2/4、epoch 0/1/3 で一致 | CONFIRMED |
| `plan_chunk_epoch` と公式 `ChunkIterFactory` | world 1/2/4、`num_samples` None/5/20、shuffle 両側、epoch 1/2/4。126 比較、不一致 0 | CONFIRMED |
| `select_utterances` と `SequenceIterFactory` | n=3/7/10/20、epoch 1/2/3/5。32 比較、不一致 0 | CONFIRMED |
| `no_sync` を forward だけに掛けてから backward | allreduce されない（rank 0 grad 10、rank 1 grad 1）。`no_sync` 無しの遅延 backward は平均 5.5 | CONFIRMED |
| clip してから平均、対、平均してから clip | 純計算の最大差 0.020。gloo Linear の最大差 0.018 | CONFIRMED（実 MossFormer の grad norm は NOT_TESTED） |
| `BatchNorm2d` / `AdaptCNN.bn1` を `train()` + `no_grad` | running_mean と `num_batches_tracked` が変わる。`eval()` では変わらない | CONFIRMED |
| gloo、`broadcast_buffers=False` | forward 後に rank 間の running_mean は不一致。rank 0 からの broadcast で rank 0 に揃う。その後 rank 0 だけ追加 forward すると再び分かれる。保存対象は rank 0 | CONFIRMED |
| `_reduce_weighted_reports` を gloo で 2 rank | generator は重み付き平均 1.5。rank 1 の `unavailable` が si_snr 全体を unavailable にし、reason は None | CONFIRMED |
| SyncBatchNorm の出力差 | CPU DDP が拒否。NOT_TESTED | NOT_TESTED |
| NCCL、実モデルの DDP step | 実行していない | NOT_TESTED |

## world size 別の動作

global batch は `plan_batches` が exact に保つ。chunk 経路は `accumulation_steps == 1` かつ `micro == global / world` を要求する。`data.virtual_world_size` はプロセス world が 1 のとき、batch plan の world をその値にする（`run.py`）。1 プロセスが rank バッチを順に足し、loss を rank 数で割る。これは DDP allreduce ではない。

### レシピ経路（`epoch_indices` + `_run_epochs`）

| world | サンプルの割り当て | 端数 | 勾配 | Validation |
|---|---|---|---|---|
| 1 | seed と epoch の全順列 | micro と accumulation に満たない末尾だけ捨てる | DDP なし。`no_sync` は無い | rank 0 が全件。分母はバッチ平均の非加重平均 |
| 2 | 同一順列の `[:N//2*2][rank::2]`。重複なし、件数同一 | 世界サイズの余りに加え、rank 内の micro / accumulation 余りも捨てる。捨てる件数は rank 間で同じ | 最終 micro だけ DDP が平均。それ以外は `no_sync` | 同上。rank 1 は forward しない |
| 4 | 同様に 4 分割。size 10 なら各 rank 2 件、2 件はどの rank にも入らない | 同上 | 同上 | 同上 |

例: size 7、world 2、epoch 0 は各 rank 3 件、index 2 が欠ける。epoch 1 では欠ける index が変わる。size が world 未満なら開始前に `TrainingError`。

optimizer 1 step は accumulation 個の micro-batch。loss は accumulation で割ってから backward するので、clip が無いときの勾配は rank 内平均を world で平均したものになる。epoch 末尾の半端な accumulation は次 epoch に持ち越さない。再開位置は accumulation 境界だけを許す。

### 固定ペア（MP-SENet / MossFormerGAN）

MP-SENet は world 1 が `fixed_single_shuffle_v1`、world > 1 が `torch_distributed_sampler_v1`。後者は余りを捨て、重複しない。検証は policy が `rank_mod_fixed_v1` でないので index 順の全件を rank 0 だけが採点する。batch 1 なので分母は発話数。

MossFormerGAN は全 world で `rank_mod_fixed_v1`。公式 `DistributedSampler` と一致した。`N % world != 0` のとき、index `>= (N // world) * world` は毎 epoch 欠け、先頭側が pad で重複する。size 5、world 2 は各 rank 3 件、欠けるのは 4、重複 2。size 11、world 4 は 8 と 9 と 10 が欠ける。検証は epoch 0 のこの列を rank 順に連結し、rank 0 だけが batch 1 で採点する。batch 1 かつ各 rank の件数が同じなので、連結の平均は公式の「rank 平均を world で平均」と同じ集合になる。重複と欠落はその集合に入る。

world 1 の公式ローダは `DistributedSampler` を使わず `DataLoader(shuffle=True)` である。現行の world 1 は replicas=1 の Sampler 順列であり、公式の 1 GPU 順とは別物。world 2/4 の列は公式 Sampler と一致する。

### DeepFilterNet3 オンライン

world 1 は pad しない。末尾の短い batch は捨てずに 1 update にする。world > 1 は件数を world の倍数まで先頭の繰り返しで pad し、各 rank の update 数を揃える。重複は loss に入る。短い末尾も 1 update として optimizer が踏む。検証は world 1 の計画で全 valid を rank 0 だけが見る。検証 forward は `eval()` なので BatchNorm の running stats は更新しない（汎用 BN で再現）。

### SEMamba++ オンライン

`per_rank_independent_v1` を実行した。world 2、size 10 で unique 8、重複 2、欠落 2。world 4、size 10 で unique 3、重複 5。rank 間で同じ発話を学び、別の発話を両方とも見ないことがある。公式 Sampler との照合は NOT_TESTED。検証は index 順の全件を rank 0 だけが読む。学習の重なりは検証分母にそのままは出ない。

### TF-GridNet / TF-Locoformer の chunk

各 rank は `samples[rank::world]` を持ち、発話は重ならない。`num_samples_per_epoch` は shard 後の各 rank に同じ回数だけ渡す。これは公式 `build_chunk_iter_factory` が shard 後の factory に同じ `num_iters_per_epoch` を渡すのと一致し、factory 比較でも一致した。

`iterator_stop` は公式 trainer が「最短 rank の終了を allreduce し、それ以降の batch を処理しない」動きである。現行は事前に `updates = min(raw_batch_counts)` とし、`rank_batches` をその prefix にする。処理される batch 集合は公式と同じ prefix である。余分な allreduce は毎 step では出さない。

捨てるもの:

- 長さが chunk 以下の発話。公式も `discard_short_samples` で捨てる。
- cache 末尾の batch 未満の chunk（`discarded_tail_chunks`）。公式の最後の `_generate_mini_batches` も出さない。
- 長い rank の、`updates` 以降の batch。公式も reporter に登録しない。

いずれも train の `train_update` にも、aligned validation の `validation_report` にも渡らない。loss の分子にも分母にも入らない。world 2 の合成例では raw batch が 14 と 26、updates 14、rank 1 の 24 chunk が捨てられた。world 4 でも kept chunk 数は rank 間で同じだった。

検証の集約は `_reduce_weighted_reports`。重みは `len(case_id)`。公式 enh の weight は batch の time 軸長ではなく参照波形の batch 次元であり、chunk 1 本が 1 サンプルなら一致する。全 rank が同じ 5 個の allreduce（sum、count、bad、present、weight）を同じ順で呼ぶ。`validation_max_updates` があるときだけ、各 rank の prefix をさらに切る。切った分も分母に入らない。`configs/training/tf_gridnet.yaml` の本線にはこの上限は無く、`p10b` の smoke だけが 1 である。

次 epoch の発話窓は `num_samples * epoch` で進む。捨てた chunk batch は次 epoch の入力にはならない。`SequenceIterFactory` も chunk の残骸を知らない。epoch 1/2/4 の比較で一致した。

global batch が world で割り切れない計画は拒否する。公式は余りを低い rank の batch に足す。公開設定の global 4 は world 1/2/4 で割り切れる。

仮想 world は 1 プロセスが全 rank の kept batch を 1 step で平均する。捨てた batch は同じく分母に入らない。Dropout や BN の逐次更新は実 DDP と違う。TF-Locoformer は Dropout を持つ。仮想 world との数値差は NOT_TESTED。

### 勾配同期と複数 Optimizer

| モデル | DDP の包み方 | accumulation | collective のコード上の順（実 NCCL は NOT_TESTED） |
|---|---|---|---|
| MP-SENet official | Generator と Discriminator を別 DDP。`find_unused` は False | 最終 micro だけ Discriminator が DDP 平均。Generator は常に `no_sync` のあと、grad があるパラメータを順に SUM して world で割る | Discriminator backward の allreduce、その後 Generator の手動 allreduce。遅延 backward が `no_sync` を守ることは gloo で確認した。二重平均にはならない |
| MossFormerGAN | Generator は `broadcast_buffers=True`。Discriminator は False。micro の前に buffer を rank 0 から broadcast | DDP かつ accumulation > 1 のとき、非最終 micro は平均前に clip する。Issue P10P-D-001 | buffer broadcast、最終 micro の Generator allreduce、Discriminator の手動 allreduce、Trainer の barrier |
| TF-GridNet / TF-Locoformer | Generator のみ。chunk は accumulation 1 | 1 backward で DDP 平均のあと、1 回 clip | forward の buffer broadcast、backward の allreduce、epoch 末の plan hash、検証の 5 allreduce、scheduler と early stop の broadcast、barrier |
| DeepFilterNet3 | `find_unused_parameters=True`。`df_fc_a` は forward に出ない | 最終 micro で DDP 平均、その後 1 回 clip | 未使用パラメータ用の探索 collective のあと、通常の allreduce |
| SEMamba++ | Generator、MRD、band の 3 DDP。1 optimizer が二つの Discriminator を持つ | 単一 micro は forward 時に sync を予約し、Discriminator step のあと Generator を backward。複数 micro は最終だけ sync。clip は蓄積の後に 1 回 | 同一 backward 内で二つの Discriminator reducer が走る。グラフが rank 間で同形なら順序は揃う。実機は NOT_TESTED |

学習ログの loss は rank 0 の局所平均である。chunk の検証 loss だけが rank 間で合算される。公式 ESPnet は `recursive_average` で学習 stats も平均する。

### Rank 0 だけが行う処理

checkpoint、`metrics.jsonl`、`status.json`、`complete.json`、TensorBoard、provenance の書き込み、レシピと TrainingData の検証 forward、chunk の best ファイル書き込み、SIGTERM ハンドラの設置。早期停止フラグと scheduler 用の検証値は rank 0 が計算し、DDP なら broadcast する。`batch_schedule` の早期停止は world 1 以外を開始時に拒否するので、broadcast 欠如では止まらない。

### Exception

`fit` の `except` は rank 0 だけ `FAILED` を書こうとして再送出する。他 rank への通知 collective は無い。失敗した rank が barrier や allreduce に到達しないと、生存側は待つ。

## Issue

### P10P-D-001

- ID: P10P-D-001
- 発見者: Agent D
- 対象: MossFormerGAN の DDP と gradient accumulation
- 種別: OFFICIAL_DIFF
- 深刻度: 高
- 確度: 高（clip が実際に縮むとき）。実ネットワークの grad norm が 10 を超えるかは NOT_TESTED
- 場所: `src/training/se/mossformergan.py` の `train_update`（約 229–257 行）。公式は `solver.py` の `_run_one_epoch_mossformergan_se_16k`（約 277–286 行）
- 根拠: 公式は micro-batch ごとに DDP backward（平均）したあと `clip_grad_norm_` する。`accu_grad: 1` は真なので accumulation 枝に入り、`effec_batch_size / batch_size` は 2。現行の 2 GPU 設定（global 4、micro 1、world 2）も accumulation 2。現行は非最終 micro を `no_sync` にし、平均前の局所勾配を clip する。最終 micro の allreduce はその clip 済み合計を平均し、もう一度 clip する。clip が恒等なら両辺は `mean(g1)+mean(g2)` で一致する。clip が縮むと `clip(mean(g))` と `mean(clip(g_local))` は一致しない。異なる 2 micro の純計算で最大絶対差 0.020、gloo の Linear で 0.018。同一 micro を 2 回繰り返した最初の gloo では差が 1e-8 未満だった。
- 再現手順: `/tmp/p10p_clip.py`。world 2、gloo、CPU。max_norm 0.05、rank ごとに違う 2 入力。
- 影響: world 1 は DDP が無く、公式の単プロセスと同じく局所勾配を clip するのでこの差は出ない。world 4 で global 4、micro 1 なら accumulation は 1 になり、平均後の 1 回 clip になる。公開 2 GPU（accumulation 2、`grad_clip` 10）で部分勾配の norm が 10 を超えると、Generator と Discriminator の更新が公式とずれる。
- 対応方針: 公式に合わせるなら、micro ごとに allreduce してから clip する。現行の `no_sync` を保つなら、clip は最終 allreduce の後に 1 回だけにする。後者は公式の数値とは、clip が途中で掛かる点でまだ違う。
- P10-C影響: 2 GPU の MossFormerGAN 再現で、clip が効いた step の重みが公式と一致しない。
- 検証状態: CONFIRMED

### P10P-D-002

- ID: P10P-D-002
- 発見者: Agent D
- 対象: MossFormerGAN の Validation と BatchNorm、checkpoint
- 種別: OFFICIAL_DIFF
- 深刻度: 高
- 確度: 高
- 場所: `mossformergan.py` の `distribute`（112–117 行）、`eval_mode`（123–127 行）、`validation_report`（277–280 行）、`state_dict`（180–185 行）。Trainer の `_validate_training_data`（1262–1324 行）は rank 0 だけが forward する。公式 `solver.py` 29–34 行と 197–201 行
- 根拠: Discriminator の `AdaptCNN` は `BatchNorm2d` を 6 個持つ。Generator は LayerNorm で running stats は無い。公式 DDP は Generator と Discriminator を `SyncBatchNorm.convert_sync_batchnorm` してから DDP し、`find_unused_parameters=True`。現行 Discriminator は `broadcast_buffers=False` で、micro-batch の前に buffer を rank 0 から 1 回 broadcast する。train 中の正規化は batch stats なので、公式は rank をまたいだ batch stats、現行は micro-batch 1 の局所 stats を使う。2 GPU・batch 1 では公式の BN 実効バッチは 2、現行は 1。検証は公式が全 rank で `model.eval()` のみ（Discriminator は train のまま、`torch.no_grad()`）。現行も Discriminator を `train()` のまま `no_grad` で forward するが、呼ぶのは rank 0 だけである。`AdaptCNN.bn1` はこの条件で running_mean が変わり、`num_batches_tracked` が 0 から 1 になった。`eval()` では変わらなかった。gloo では forward 後に rank の running_mean は不一致、rank 0 broadcast で rank 0 に揃い、rank 0 だけの追加 forward で再び不一致になった。保存される `state_dict` は rank 0 の unwrap である。検証の直後に epoch 末 checkpoint を書くので、ファイルの BN buffer は検証で更新された rank 0 のもの。rank 1 のメモリは次の `train_update` の broadcast まで古い。再開時は全 rank が同じファイルを DDP 化の前に読む。現行ループは Discriminator を `eval()` しないので、running stats は正規化の出力そのものには使われない。学習と検証の出力を公式からずらすのは、batch stats が局所であることと、検証 batch を rank 0 が全件逐次で見ることである。
- 再現手順: `/tmp/p10p_agent_d.py` の AdaptCNN 部と gloo bn。SyncBN との出力差は CPU では起動できず NOT_TESTED。
- 影響: 検証 GAN 項と学習時の Discriminator 特徴が公式の SyncBN と違う。checkpoint の Discriminator running stats は全 GPU の検証バッチを同期したものではない。Generator の推論重みは LayerNorm なので、この buffer 差は Generator のパラメータには乗らない。
- 対応方針: 公式に合わせるなら Discriminator を SyncBatchNorm にし、検証 forward を全 rank で行う。複数 forward が同一グラフの buffer を壊す現行コメントがあるので、SyncBN の in-place 更新が 3 回の Discriminator forward と衝突しないかを別途確認する。
- P10-C影響: 2 GPU MossFormerGAN の検証 loss と、再開に使う Discriminator buffer が公式 DDP と一致しない。
- 検証状態: CONFIRMED

### P10P-D-003

- ID: P10P-D-003
- 発見者: Agent D
- 対象: MossFormerGAN の world size 1 のサンプル順
- 種別: OFFICIAL_DIFF
- 深刻度: 中
- 確度: 高
- 場所: `src/reproduction/pipeline/order.py` の `_rank_mod_fixed`。公式 `dataloader.py` の `get_dataloader`（225–242 行）
- 根拠: world 2/4 の `rank_mod_fixed_v1` は公式 `DistributedSampler.__iter__` の写経と一致した。公式は `args.distributed` が偽のとき Sampler を付けず `shuffle=True` にする。現行の world 1 は replicas=1 の Sampler 順列（seed+epoch）である。1 プロセス用の `mossformergan_vb28_official.yaml` もこの policy を使う。
- 再現手順: `/tmp/p10p_agent_d.py` の rank_mod 節。公式 DataLoader 自体は起動していない。分岐は公式ソースの条件で足りる。
- 影響: world 1 の学習順と検証順は公式 1 GPU と一致しない。world 2/4 の列は公式 Sampler と一致する。`N % world != 0` の欠落と重複は公式と同じであり、現行だけの欠落ではない。
- 対応方針: 1 GPU 公式を再現するなら、非分散の DataLoader shuffle を別 policy にする。2 GPU 公式が対象なら world 1 の設定を公式比較に使わない。
- P10-C影響: 1 プロセス実行を公式 1 GPU と比較するとデータ順が違う。2 GPU 比較の列はこの Issue の対象外。
- 検証状態: CONFIRMED

### P10P-D-004

- ID: P10P-D-004
- 発見者: Agent D
- 対象: MossFormerGAN の PESQ 失敗時の学習 step
- 種別: OFFICIAL_DIFF
- 深刻度: 高
- 確度: 高
- 場所: `mossformergan.py` の `train_update`（235–240 行）。公式 `loss.py` 72–75 行と `solver.py` 326–329 行
- 根拠: 公式の学習は PESQ が None の batch を backward せずに次へ進む。検証は loss を足さず（分子 0）、分母 `(i+1)` には残す。現行の学習は `metric_targets` が None（バッチ内のどれか 1 本でも PESQ 失敗）なら `TrainingError` で step を中断する。検証の `validation_report` は値 0、status ok で分母に残す。これは公式の検証分母と一致する。付随して、ri / magnitude / time / gan は失敗 batch を `na` にして平均から外す。plateau が見るのは generator なので、判定分母は 0 を残す側である。
- 再現手順: コード対応。PESQ 失敗の実音声では起動していない。
- 影響: 無音などで PESQ が落ちると公式は学習を続け、現行は実験が落ちる。失敗が一部の rank だけだと、進んだ rank の allreduce に失敗 rank が来ない。停止自体は P10P-D-005。
- 対応方針: 公式に合わせるなら、その micro-batch を backward しない。その場合も rank ごとに backward の有無が違うと allreduce が揃わないので、skip フラグを allreduce してから全 rank が同じ枝に入る必要がある。
- P10-C影響: 長い 2 GPU 学習で PESQ 失敗が 1 回でも出ると、公式なら継続する地点で現行は停止し得る。
- 検証状態: CONFIRMED

### P10P-D-005

- ID: P10P-D-005
- 発見者: Agent D
- 対象: DDP の停止と例外
- 種別: RISK
- 深刻度: 高
- 確度: 中（分岐はコードで確定。待ちは起こし得るが、意図的には再現していない）
- 場所: `trainer.py` の `fit`（182–184 行、267–274 行、377 行付近の barrier）。`_stop_requested`。`mossformergan.py` の PESQ 例外。`mp_senet.py` 298–303 行の「grad が None なら allreduce しない」
- 根拠: SIGTERM ハンドラは rank 0 だけが設置する。`timed_stop` はプロセスローカルなので rank 0 だけが内側ループを抜ける。他 rank は次の `train_update` の allreduce に入り、rank 0 は epoch 末の barrier にいる。managed stop（deadline、segment_steps）は world 1 以外を拒否するが、SIGTERM は拒否しない。いずれか 1 rank が collective の前に例外を出すと、`fit` は他 rank を解放せず再送出する。rank 0 以外の FAILED 書き込みは `_write_status` が即 return する。PESQ 例外、非有限 loss、検証中の例外がこの形になる。MP-SENet と MossFormerGAN の手動 allreduce は `parameter.grad is not None` のときだけ呼ぶ。静的な計算グラフなら全 rank で同じ回数だが、rank だけ grad が欠けると回数がずれて待つ。
- 再現手順: 実行していない。待ちを起こすと既存の学習プロセスを巻き込む恐れがあるため。
- 影響: 正常終了の経路では全 rank が同じ回数の collective に到達する。SIGTERM、rank 局所の例外、rank 局所の未使用勾配で、プロセスが collective 上で待ち続けることがある。
- 対応方針: 停止フラグと失敗フラグを、次の collective の前に全 rank で broadcast する。手動 allreduce は grad の有無を bitmap で揃えてから呼ぶ。
- P10-C影響: 正常な step の正しさとは別に、2 GPU 以上のジョブを SIGTERM や PESQ 失敗で止めようとすると回収できない可能性がある。
- 検証状態: SUSPECTED

### P10P-D-006

- ID: P10P-D-006
- 発見者: Agent D
- 対象: DeepFilterNet3 の DDP と BatchNorm
- 種別: BUG
- 深刻度: 中
- 確度: 高（機構）。公式 DeepFilter の複数 GPU 実装はこの checkout に無く、公式差としては NOT_TESTED
- 場所: `deepfilternet.py` の `distribute`（144–146 行）と `validation_report`（246–258 行）。`src/se/models/deepfilternet3/model.py` の `BatchNorm2d`。`wrap_module` の既定 `broadcast_buffers=True`
- 根拠: `find_unused_parameters=True` は `df_fc_a` が forward に無いことと整合する。未使用パラメータで DDP が止まる問題はここだけ塞いである。running stats は通常の BatchNorm なので、各 forward は rank 0 の buffer を配ってから局所バッチで更新する。checkpoint は rank 0 の更新後の stats である。検証は `eval()` かつ rank 0 だけなので、検証そのものは running stats を変えない。変えるのは学習 forward である。`validation_report` は「eval は running stats を使う」と書いており、保存された stats は rank 0 の学習バッチだけを反映する。gloo の汎用 BN で、broadcast 後の保存値が rank 0 の局所更新になることを確認した。実 DFN forward は回していない。
- 再現手順: `/tmp/p10p_agent_d.py` の gloo bn は同じ broadcast 規則の縮小版。
- 影響: 複数 GPU で保存した DFN を eval すると、BN は rank 0 のデータ分布の running stats を使う。重みの勾配は全 rank の平均である。
- 対応方針: eval 品質に running stats を使うなら SyncBatchNorm か、全 rank の stats を明示的に平均してから保存する。
- P10-C影響: DFN を world 1 で学習するならこの差は無い。world 2/4 の checkpoint を公式の単一 GPU 統計と比較すると BN の母集団が違う。
- 検証状態: CONFIRMED

### P10P-D-007

- ID: P10P-D-007
- 発見者: Agent D
- 対象: 学習 loss の rank 間集約
- 種別: OFFICIAL_DIFF
- 深刻度: 低
- 確度: 高
- 場所: 各 `train_update` の戻り値。`trainer.py` の `_log_train`（1410–1435 行）。公式 `trainer.py` 682–694 行の `recursive_average`
- 根拠: chunk の DDP は勾配を平均する。ログに書く loss は rank 0 の batch 平均で、allreduce しない。aligned validation は重み付きで allreduce する。仮想 world は 1 プロセスが全 rank の loss を平均してログする。実 DDP のログとは母集団が違う。
- 再現手順: コード対応。学習ログの数値比較はしていない。
- 影響: 勾配の更新対象は全 rank の kept batch である。TensorBoard と `metrics.jsonl` の train loss は rank 0 の batch であり、公式ログの全体平均ではない。early stopping は検証側を見るので、このログ差だけでは停止判定は動かない。
- 対応方針: 比較に train loss を使うなら、ログの直前に allreduce する。勾配側のスケールは変えない。
- P10-C影響: 公式ログの train loss との数値比較に rank 0 の値を使うと母集団が違う。検証 loss の chunk 集約は別経路。
- 検証状態: CONFIRMED

### P10P-D-008

- ID: P10P-D-008
- 発見者: Agent D
- 対象: aligned validation の `unavailable`
- 種別: BUG
- 深刻度: 低
- 確度: 高
- 場所: `trainer.py` の `_reduce_weighted_reports`（1627–1689 行）
- 根拠: 1 件でも status が `unavailable` だと、その rank の bad フラグが 1 になる。allreduce の和が正なら、他 rank の有限値も捨てて指標全体を unavailable にする。reason は allreduce せず、rank 0 の局所 reason が残る。gloo で rank 0 の si_snr 10（重み 2）と rank 1 の unavailable（重み 2）を入れ、結果は unavailable、reason None、count 0 だった。generator の重み付き平均 1.5 は同時に正しく残った。現行の TF-GridNet / TF-Locoformer の `validation_report` は pesq と stoi を `na` にし、si_snr は `score_si_sdr` の ok / na / error であり、`unavailable` は返さない。この経路は今の chunk 検証では発火しない。MP-SENet の PESQ `unavailable` は rank 0 の `_combine_metric` であり、reason は残る。
- 再現手順: `/tmp/p10p_agent_d.py` の gloo reduce。
- 影響: 将来または別モデルが chunk 検証で `unavailable` を返すと、他 rank の有効スコアが分母から消え、理由も消える。0 としては入らない。
- 対応方針: サンプル単位で unavailable を分母から外すか、全 rank が unavailable のときだけ指標を unavailable にする。reason も gather する。
- P10-C影響: 現行 TF の chunk 検証の数値は変えない。集約関数としては欠陥がある。
- 検証状態: CONFIRMED

### P10P-D-009

- ID: P10P-D-009
- 発見者: Agent D
- 対象: MossFormerGAN の `find_unused_parameters`
- 種別: RISK
- 深刻度: 中
- 確度: 低（フラグの差は確定。未使用パラメータの有無は未測定）
- 場所: `mossformergan.py` の `wrap_module` 呼び出し。公式 `solver.py` 31–34 行は両方 True
- 根拠: 現行は既定 False。公式は True。未使用パラメータが 1 つでもあると、False の DDP は backward の allreduce で待つ。Generator の手動経路は MossFormer には無く、通常の DDP backward である。CPU で全パラメータの grad 被覆は見ていない。
- 再現手順: 実行していない。
- 影響: 全パラメータが毎 step 使われていれば数値は変わらない。使われていなければ 2 GPU の最初の backward で停止し得る。
- 対応方針: 小さい入力で grad が None のパラメータが無いかを CPU で数える。あれば True にする。
- P10-C影響: 未使用が無ければ P10-C の数値には影響しない。ある場合は 2 GPU 学習が開始直後に止まる。
- 検証状態: SUSPECTED

## 反証して問題にならなかった仮説

- chunk の `iterator_stop` が公式と違い、捨てた chunk が loss 分母に入る。公式 factory と 126 件一致し、Trainer は `plan.updates` の prefix だけを `train_update` と aligned validation に渡す。尾と短い発話も batch に入らない。
- `num_samples_per_epoch` を全体に掛けてから shard している。shard 後の各 rank に同じ回数を渡しており、`SequenceIterFactory` と 32 件一致した。epoch をまたいでも一致した。
- world 間で `epoch_indices` が同じサンプルを二重に学ぶ、または rank ごとに件数が違う。重複は 0、件数は揃う。欠けるのは余りだけである。
- `plan_batches` が global batch を壊す。試した割り切れる組は積が global と一致し、割り切れない組は拒否した。
- `no_sync` を抜けたあとの backward が DDP allreduce をやり直す。gloo では allreduce されなかった。MP-SENet の手動 SUM / world は二重平均にならない。
- MossFormer の検証分母が PESQ 失敗を 0 件として落とす。generator は値 0 の ok として平均に残る。公式の `(i+1)` と同じ扱いである。
- 検証の早期停止や scheduler が rank 0 だけで分岐し、他 rank が別の collective に入る。`_chunk_should_stop` と `_scheduler_validation` は broadcast する。`batch_schedule` は world > 1 を拒否する。
- plan hash の不一致で allgather の途中に片方だけ落ちる。全 rank が allgather したあと、hash が違えば全 rank が `TrainingError` を出す。
- DeepFilterNet の `df_fc_a` が `find_unused_parameters=False` のままである。`distribute` は True を渡している。
- 仮想 world が batch plan の world 1 と矛盾して必ず起動できない。`run.py` は plan の world を仮想 world にする。global 4、micro 上限 2、仮想 world 2 なら accumulation 1 で chunk の検査を通り、step 内で rank 数分の平均を取る。
- rank_mod の欠落が現行だけのバグである。公式 Sampler の写経と一致する公式と同じ欠落である。

## 未監査

- NCCL、実 GPU、実モデルの 1 step。collective の順序はソース上の順序だけである。
- SyncBatchNorm と局所 BN の出力差。CPU DDP は SyncBN を拒否した。
- MossFormer 実勾配の norm が 10 を超えるか。未使用パラメータの被覆。
- 公式 DeepFilter の複数 GPU。checkout の Python に DDP は無かった。
- SEMamba 公式 Sampler と `per_rank_independent_v1` の一致。重なりは現行関数では確認した。
- TF-Locoformer の Dropout を仮想 world と実 DDP で比べること。
- MP-SENet 公式リポジトリの DataLoader と `fixed_single_shuffle_v1` の一致。現行の分割規則だけを実行した。
- `validation_max_updates` や `max_updates` を付けた smoke が、フル検証集合の代わりに使われたときの運用判断。コード上は prefix だけが分母に入り、ログに `validation_scope` が残る。
- 既存プロセスや本番データは触っていない。
