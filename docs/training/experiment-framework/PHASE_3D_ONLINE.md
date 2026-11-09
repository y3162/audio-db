# Phase 3-D Online Data Pipeline

DeepFilterNet3 の `online.dfn3` と SEMamba++ の `online.semambapp_gsr` を、Phase 3-B / 3-C と同じ `SnapshotDataModule` から供給する。新しい DataModule は作らない。SE モデル、公式損失、公式 Trainer は移していない。world size 2 の Data System 正式受け入れは Phase 3-E のままである。

測定は 2026-11-09、`.venvs/se`、`CUDA_VISIBLE_DEVICES=""` である。GPU 試験は行っていない。

## 1. 変更ファイル一覧

追加:

- `src/experiments/data/online.py`。`Dfn3IndexSampler`、`Dfn3Protocol`、`SemambaProtocol`。
- `src/reproduction/pipeline/clips.py`。`Dfn3Clip` と `SemambaClip`。
- `tests/experiments/test_online_data.py`。
- この記録。

変更:

- `src/experiments/data/__init__.py`。上の 3 クラスを公開する。
- `src/experiments/data/sample.py`。`SampleRef` に `split` と `plan_epoch` を足した。plan hash の payload には入れない。
- `src/experiments/data/fixed_pair.py`。`OrderSampler` が SEMamba の索引を返す。`MicroBatcher` に `keep_partial` を足した。検証の詰め方を online と chunk で分けた。
- `src/reproduction/pipeline/online_dfn3.py` と `online_semamba.py`。各 `_AssetClip` を上の clip 型の別名にした。混合と劣化の式は変えていない。

`src/training/` は変えていない。既存の `OnlineDFN3TrainingData`、`OnlineSemambaTrainingData`、`ReproductionTrainingData` は残している。DB、snapshot、素材は再構築していない。

## 2. Dfn3IndexSampler

公開クラスは `src.experiments.data.Dfn3IndexSampler` である。コンストラクタは source、`seed`、任意の `global_batch` である。policy は `dfn3_online_ceil_v1` をそのまま返す。`resolve_order_policy` は呼ばない。この名前は `ORDER_POLICIES` に無い。

`sequence` と `all_ranks` は `plan_online_epoch` の micro-batch を平坦な `SampleRef` 列にする。`SampleRef.index` は計画上の sample index であり、発話 index ではない。world size はプロセスの world size のまま渡す。world size 1 では関数が短い最終 micro-batch を残す。Sampler はその尾を捨てない。

発話数は `speech_key_positions(speech_count, sampling_factor, epoch=0)` の長さである。件数は epoch に依存しない。並べ替えだけが epoch に依存する。`sampling_factor` は rule の `online.sampling_factor` で、正の数でなければ `DataError` である。

`group_size` が 1 でないときは `DataError`。`bind_batch` は `MicroBatcher.keep_partial` を True にする。省略した実効 batch は `micro_batch * world_size` である。kwargs の `global_batch` がその積と違うときは `DataError`。rule の `global_batch`（出荷上限）と違うときは、rank 0 が 1 行だけ `warning: global_batch_size {effective} differs from the shipped cap {shipped}` を出し、指定値を使う。差だけでは `DataError` にしない。

## 3. Dfn3Protocol

公開クラスは `src.experiments.data.Dfn3Protocol` である。`prepare` は 48 kHz、窓 144,000、教師が null または `official_mixed_target` であることを見る。分布は `DFN3OnlineSpec.shipped()` である。合成 snapshot の rule には `sampling_factor` だけを書き、混合分布は出荷仕様のままである。

`render` は `render_mixture` を呼ぶ。発話 index は `speech_key_positions(...)[sample.index]` である。種は `SampleRef` の realization から読む。波形の epoch key は train だけが epoch で、valid と test は 0 である。rank と worker は種に入れない。NumPy のグローバル乱数は使わず、`PCG64(sample_seed(...))` をその呼び出しだけで作る。教師は `official_mixed_target`。case があるとき、文書の `teacher_kind` がこれと違うと `DataError` である。

RIR と雑音は汎用 Transform に分けない。読み出しは `Dfn3Clip` で、soundfile の span 読みである。RIR の slot が無いときは空列、speech と noise が無いときは `DataError` である。欠けるファイルの `LibsndfileError` も `DataError` にする。

## 4. OrderSampler のオンライン対応

`OrderSampler` は固定ペアと SEMamba++ の両方に使う。固定ペアは従来どおり `protocol.locate` を呼ぶ。`online.semambapp_gsr` のときだけ `locate` を呼ばず、`rank_order("per_rank_independent_v1", ...)` の発話 index を `SemambaProtocol.reference` で `SampleRef` にする。この列は drop 前の全件である。端数を捨てるのは `MicroBatcher` である。

`order_parameters` は、固定ペアのとき既存の `order_params(policy, seed)` だけを返す。追加キーは付けない。SEMamba のときだけ `dataset_id`、`micro_batch`、`group_size` をその辞書へ足す。`bind_batch` はどちらの供給でも world size を記録する。固定ペアの plan hash の形はこれで Phase 3-B のままである。

SEMamba の `group_size` は 1 以上で、Trainer の `accumulation` と DataModule が一致を見る。kwargs の `global_batch` が `micro_batch * world_size * group_size` と違うときは `DataError`。rule の `global_batch` と違うときは rank 0 が 1 行 `warning: global_batch_size {effective} differs from the shipped batch {shipped}` を出し、指定値を使う。旧供給クラスが等式を要求していた条件と、この警告は別である。

`validation_ranks` は SEMamba だけが持つ。固定ペアの検証は、このメソッドが付いたことでは切り替わらない。`SnapshotDataModule.validation_epoch` は、供給が `online_dfn3` または `online_semamba` のときだけ online の詰め方を使い、`chunk` のときだけ chunk の `validation_ranks` を使う。

## 5. SemambaProtocol

公開クラスは `src.experiments.data.SemambaProtocol` である。`prepare` は 16 kHz、窓 48,000、教師が null または `dry_clean` であることを見る。`render` は `render_example` を呼ぶ。`speech_index` は `SampleRef.index` そのもので、DFN3 のような発話キー変換はしない。順序は reverb、noise、clipping、bandwidth のままである。公式の 2 回抽選と 1 回目の破棄、独立 peak、crop / pad は `render_example` の中である。教師は `dry_clean`。

読み出しは `SemambaClip` で、`librosa.load(..., sr=16000, mono=True)` のままである。DFN3 の soundfile span 読みとは別の型にした。式は移していない。

種は `OrderSampler.bind_batch` が protocol へ渡す。realization の epoch key は train が epoch、valid と test が 0 である。rank は入れない。

## 6. MicroBatcher の変更点

`MicroBatcher` に `keep_partial: bool = False` を足した。既定は False で、端数の sample を捨て、そのあと `group_size` に満たない更新群も捨てる。空、完全な micro-batch が 0、完全な更新群が 0 は、これまでどおり `DataError` である。

`keep_partial` が True のときだけ、幅に満たない最後の micro-batch を残す。`group_size` の端数は、そのときも捨てる。DFN3 は `group_size` を 1 に固定してから True にするので、短い最終 micro-batch が残る。固定ペア、chunk、SEMamba は False のままである。chunk は Phase 3-C と同じく、計画関数が切った batch を平坦化して同じ幅で区切り直す。

検証は学習用の `MicroBatcher` 幅を使わない。`validation_batch()` の幅、`keep_partial=True`、`group_size=1` で詰める。検証の最後の短い batch は捨てない。

## 7. Data Identity と Plan Hash

DataModule の公開メソッドは変えていない。`EpochView.batch(i)` は長さ 1 の `Batch` リストである。`state_dict` のキーは `epoch`、`update_index`、`plan_hash`、`world_size`、`identity`、`weights` である。`update_index` は次の micro-batch 添字である。`weights` は null である。`initialize_from` は data 状態を戻さない。

identity のキーは Phase 3-B のままである。再開比較は `dataset_name` と `dataset_version` だけを外す。`dataset_id` は外さない。

online で case id が無い sample の `sample_id` は `{protocol_id}:{split}:{index}` である。valid と test に case id があるときは、その case id を `sample_id` にする。`realization_id` は常に `{protocol_id}:{snapshot_hash}:{seed}:{split}:{epoch_key}:{index}` である。train の `epoch_key` は epoch、valid と test は 0 である。rank は入れない。case 文書は実現した SNR を持たないので、realization を case id にはしない。

plan hash のトップレベルキーは Phase 3-B / 3-C のままである。`dataset_id` を新しいトップレベルキーにはしていない。それを足すと固定ペアと chunk の hash が変わる。dataset、protocol、snapshot、seed、epoch、world size、順序方針、sampling factor、micro-batch、group size、実際の sample 順は、既存の `order_params` と `ranks` の `SampleRef` 列で表す。DFN3 の `order_parameters` には `sampling_factor`（小数 6 桁の文字列）、`speech_key_epoch`（整数因子は `zero`、非整数は `epoch`）、`count`、`dataset_id`、`micro_batch`、`world_size` を入れる。worker 数、prefetch、device、消費済み cursor は入れない。hash の再計算は波形を描かない。

`SampleRef.split` と `plan_epoch` は描画が realization を読むために持つ。`_ref_payload` には入らない。

## 8. 数値比較

fixture は一時ディレクトリの合成 snapshot である。公式素材は開いていない。DFN3 の整数因子は speech 5、`sampling_factor` 1、micro-batch 2、rule の `global_batch` 2、valid batch 2 である。非整数は speech 4、因子 1.5、発話キー長 6 である。SEMamba は speech 5、micro-batch 2、rule の `global_batch` 2、valid batch 2 である。seed は計画比較が 7、world size は 1 である。

DFN3 の学習列は epoch 0 と 1 で `plan_online_epoch` と一致した。sample index、micro-batch 境界、短い最終 batch を比べた。整数因子は 5 件を幅 2 で切り、最後は 1 件である。因子 1.5 は 6 件を幅 4 で切り、最後は 2 件である。どちらも最後が幅より短く、0 から count-1 を 1 回ずつ含む。因子 1.5 の `speech_key_positions` は epoch 0 と 1 で違う。整数因子の `speech_key_epoch` は `zero`、1.5 は `epoch` である。描画した 1 件は、sample index と発話キーが違う位置を選んだ。`render_mixture` との noisy と clean の最大絶対誤差は 0 で、長さは 144,000、教師は `official_mixed_target` である。

SEMamba の `sequence` は epoch 0 と 1、`group_size` 1 と 2 で、`rank_order` の全件と一致した。`MicroBatcher` のあと、つまり端数を捨てた index 列は `plan_semamba_epoch` と一致した。`len(micro-batches) // group_size` は `plan.updates` と一致した。残した sample 数は 5 未満で、幅 2 の端数 1 件は捨てている。`group_size` 2 でも、捨てたあとの index 列は計画関数と同じである。train の `sample_id` は `{protocol_id}:train:{index}` なので、index 列が同じなら sample id も同じである。描画した 1 件は `render_example` との degraded と clean の最大絶対誤差が 0 で、長さは 48,000、教師は `dry_clean`、train の epoch key は 1 である。

`.venvs/se` には `resampy` が無い。`kaiser_best` と `kaiser_fast` の bandwidth は、既存の `render_example` が librosa 経由で `resampy` を読む。式は変えていない。再開試験の SEMamba は、epoch 0 の先頭 2 件がこの分岐に入らない seed を選んだ。選んだ抽選でも `render_example` を呼ぶ。Kaiser の一致は、これまでどおり `.venvs/asr` の再現試験の範囲である。

## 9. Validation

DFN3 の検証列は `plan_online_epoch` を world size 1、件数全体、検証 batch 幅で作る。rank 0 だけが返し、他 rank は空の iterator である。並びは epoch に依存してよい。波形の epoch key は 0 である。整数因子の合成 snapshot では、epoch 0 と 1 の列がどちらも index 0 から 4 の順列で、同じ index の noisy と clean の最大絶対誤差は 0 だった。並びが epoch で変わることと、波形が index で一致することは別に確認した。同じ batch 位置の比較では足りないので、index で対応させた。

非整数の `sampling_factor` では、既存の `speech_key_positions` が検証 epoch でも発話キーを入れ替える。波形の種の epoch key は 0 のままである。Phase 3-D はこの既存関数に従った。検証波形の epoch 間一致は、出荷値でもある整数因子で測った。非整数のキーが epoch で変わることは、学習計画の因子 1.5 で記録した。黙って epoch 0 のキーへ揃えてはいない。

SEMamba の検証は学習 Sampler の並べ替えを使わない。rank 0 が素材順 `range(count)` を返し、他 rank は空である。epoch key は 0 である。epoch 0 と 1 の index 列はどちらも 0, 1, 2, 3, 4 で、同じ index の波形の最大絶対誤差は 0 だった。valid の `sample_id` は保存された case id で、`realization_id` は `:valid:0:` を含む。検証 batch 幅は rule の `validation.batch` で、学習の drop は掛けない。`group_size` 2 の学習モジュールでも、検証は 5 件すべてを返した。

## 10. Checkpoint と Resume

形式は `audio-db-experiment-v1` のままである。両方のアルゴリズムで、CPU / FP32、world size 1、logical step 2、1 step で中断し、同じ設定の別プロセスから再開した。連続実行の metrics 上の sample id と、再開後の sample id は一致した。中断ファイルを別プロセスの `restore` で読んだあと、次の micro-batch の `sample_id`、`realization_id`、input、teacher の最大絶対誤差は 0 だった。epoch は 0、`update_index` は 1、data identity と plan hash は未消費の実行と一致した。prefetch は中断側で有効にしたが、未消費の先読みは学習位置に含まれない。`load_state_dict` は先に prefetch を捨てる。

改ざんした `plan_hash` の checkpoint は `CheckpointError` である。同じプロセスの `load_state_dict` で `plan_hash` または `dataset_id` を変えると `StateError` である。

## 11. 異常系

次は `DataError` である。`group_size` 2 の DFN3。kwargs の `global_batch` が積と違う DFN3 と SEMamba。範囲外の sample index。case の `teacher_kind` が DFN3 の `official_mixed_target` と違う検証 batch。`declared_status` が blocked の snapshot。noise 素材が無い SEMamba の最初の学習 batch。speech が空、case 件数が online 列と違う、index が発話キー長の外、という条件も同じ例外である。

rule の出荷 batch と実効 batch の差は警告であり、それだけでは拒否しない。因子 1.5 の snapshot で micro-batch 2、出荷上限 4 のとき、ログに `shipped cap` が出ることを確認した。

## 12. 回帰テスト

変更前、2026-11-09、`CUDA_VISIBLE_DEVICES=""`:

- `tests/experiments`: 63 件、失敗 0、skip 1、exit 0。Phase 3-C の基準と同じである。
- `tests/reproduction`: 135 件、error 1、skip 2、exit 1。error は `test_p8_dfn3.TrainerSmokeTests.test_one_deepfilternet_step_is_finite` の `GLIBC_2.33`（`libdf_rfft.so`）である。skip は保存済み VoiceBank 派生物が無いことと、P2 snapshot がこの機械に無いことである。

変更後、同じコマンド:

- `tests/experiments`: 71 件、失敗 0、skip 1、exit 0。増分 8 件は `test_online_data.py` である。skip は変更前と同じ `test_cuda_amp` の `NOT_RUN: CUDA is not visible` で、`CUDA_VISIBLE_DEVICES=""` による既存の skip である。
- `tests/reproduction`: 135 件、error 1、skip 2、exit 1。error は変更前と同じ `test_one_deepfilternet_step_is_finite` の `GLIBC_2.33`（`libdf_rfft.so`）である。新しい fail、error、skip は無い。

新規の `tests/experiments/test_online_data.py` は 8 件、失敗 0、38.915 秒、exit 0 である。

## 13. Phase 3-E への残課題

- world size 2 の Data System 正式受け入れ。今回のコードは `plan_online_epoch` と `rank_order` にプロセスの world size を渡すが、world size 2 の一致は受け入れていない。SEMamba の world size 2 で既知の差がある並べ替えは直していない。
- CUDA / NCCL のオンライン学習、prefetch の worker プール、動的重み、公式素材での学習、DFN3 と SEMamba のモデル接続、公式損失、GAN Trainer、FSDP、sharded checkpoint。
- `.venvs/se` に `resampy` が無いため、Kaiser bandwidth の描画はこの環境では動かない。式の変更では解消しない。
