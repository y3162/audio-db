# Phase 3-B 固定ペア DataModule

固定ペアの学習供給を `src/experiments/data/` の公開 API にした。MP-SENet の `noisy_rms_unit_v1` と MossFormerGAN の `independent_audio_norm_v1` を、world size 1 の合成 snapshot で既存の純関数と比較した。公式 SE の学習再現、公式 16 kHz 変換器との一致、実在の VoiceBank snapshot、world size 2 の Data System 正式受け入れは、この結果に含まれない。

測定は 2026-11-09、`.venvs/se`、`CUDA_VISIBLE_DEVICES=""` である。GPU 試験は行っていない。

## 1. 変更ファイル

実装:

- `src/experiments/data.py` を削除し、`src/experiments/data/` にした。中身は `__init__.py`、`sample.py`、`tensor.py`、`identity.py`、`fixed_pair.py`。
- `src/experiments/errors.py` に `DataError`。
- `src/experiments/trainers.py` は、forward 前の device 移動と `group_size` 検査だけを足した。
- `src/experiments/checkpoint.py` は、表示名を外した identity 比較と、epoch ごとの `plan_hash_for_state` を使う。
- `src/experiments/builder.py` は、構築途中のレジストリを `components` に残す。
- `src/experiments/api.py` の `close` は、完成グラフか、失敗時の部分レジストリの `role: data` を閉じる。
- `src/reproduction/pipeline/planner.py` に `fixed_pair_crop`、`fixed_pair_case_length`、`fixed_pair_pad_name`、`fixed_pair_plan_index` を足した。中身は既存の `_crop`、`_pair_length`、`_pad_name`、`_plan_index` を呼ぶ。

試験と記録:

- `tests/experiments/test_fixed_pair_data.py`
- `tests/experiments/support.py` の `IdentityWave` と `BoomLoss`
- この文書と、設計書の現状文、`.agents/PROJECT.md`、`.agents/DECISIONS.md` の実装済み 52

`src/training/` は変えていない。`src/reproduction` の既存供給クラスは変えていない。checkpoint 形式は `audio-db-experiment-v1` のままである。

## 2. 実装した公開 API

`from src.experiments.data import TensorPairData, Batch, EpochView` と、YAML の `src.experiments.data.TensorPairData` は残る。追加の公開名は `SnapshotSource`、`OrderSampler`、`FixedPairProtocol`、`MicroBatcher`、`SnapshotDataModule`、`SampleRef`、`Crop`、`ChunkSpan` である。`ChunkSpan` は型だけで、chunk の供給は作っていない。

`SnapshotDataModule` の呼び出しは `setup`、`train_epoch`、`validation_epoch`、`state_dict`、`load_state_dict`、`identity`、`lineage`、`close` である。`train_epoch` は cursor を進めない。`EpochView.batch(i)` は長さ 1 のリストで、要素は 1 micro-batch の `Batch` である。`Batch` は従来の 4 位置引数に、`lengths`、`sample_ids`、`realization_ids`、`teacher_kind` を足している。`TensorPairData` は `group_size` を持たない。

Source のコンストラクタは資産を読まない。`open` が `load_snapshot` と `AssetStore.from_snapshot` を呼ぶ。`close` は二重に呼んでも例外にしない。

## 3. Phase 3-A 設計書との相違

3 点を、実装の前に決めた。

`identity()` は表示用の `dataset_name` と `dataset_version` を含む。再開の一致は `comparable_identity` が、`dataset_id` を持つ辞書からこの 2 キーだけを外して比べる。`dataset_id` と `snapshot_hash` は残る。`TensorPairData` は `dataset_id` を持たないので、辞書全体を比べる。CheckpointManager も同じ関数を使う。表示名だけの変更で再開を拒否しない。

`close` の所有者は `Experiment.close()` である。Phase 3-A の文面にあった「Trainer の finally」は採用しない。`StandardTrainer` は DataModule を閉じない。

`plan_hash` は epoch ごとに変わる。checkpoint は `load_state_dict` より前に hash を比べるので、`plan_hash_for_state` が保存された epoch の hash を再計算する。これは YAML の公開 API ではない。`TensorPairData` はこのメソッドを持たず、従来どおり属性の `plan_hash` を比べる。

prefetch は、次の 1 micro-batch を CPU で先に描く daemon thread で、既定は off である。worker プールの本実装ではない。cache は `state_dict` に入らない。`close` と `load_state_dict` で捨てる。

検証は固定ペアの 2 行を実装した。Phase 3-A の「Phase 3-B は MP-SENet の検証だけ」は、chunk とオンラインを後段へ残す意味に読み、MP-SENet と MossFormerGAN の両方をこの段の対象にした。

## 4. Identity と Close

`identity()` が返すキーは `algorithm`、`claim_label`、`class`、`dataset_id`、`dataset_name`、`dataset_version`、`group_size`、`micro_batch`、`normalize`、`order_policy`、`profile`、`protocol_id`、`seed`、`selection_uses_test`、`snapshot_hash`、`split`、`window` である。`class` は `src.experiments.data.SnapshotDataModule`。`selection_uses_test` は snapshot の値のままである。DataModule は profile を推測して書き換えない。

`require_profile` が snapshot の profile と違うときは `setup` が `DataError` を出す。Builder はそれを `BuildError` で包む。`delivery` が `on_demand` 以外、`declared_status` が `ready` 以外、algorithm が `fixed_pair.` で始まらないときも `DataError` である。`leakage_safe` で valid と test の case id が重なるときも `DataError` である。

`state_dict` は `epoch`、`identity`、`plan_hash`、`update_index`、`weights`、`world_size` を持つ。`weights` は null である。null 以外は `StateError`。`load_state_dict` は comparable identity、world size、再計算した plan hash、cursor が `0..updates` に入ることを要求する。`update_index == updates` は、Trainer が epoch を進める直前の位置として許す。

`Experiment.build` が完成グラフの前に失敗したとき、Builder のレジストリを `_partial_components` に残す。`close` は完成グラフがあればそれを、無ければ部分レジストリの `role: data` を閉じる。二重 close は何もしない。`setup` の検証失敗では Source を閉じず、Experiment の `close` が閉じる。

`initialize_from` は `initialize.allow` の module だけを読む。data の cursor は 0 のままである。

## 5. 固定ペアの数値比較

試験の snapshot は一時ディレクトリの 16 kHz PCM_16 である。case は 12 件で、train 8、valid 2、test 2。case 0 は 4 sample、他は 16 sample。window は 8。`resample` キーは置かない。seed は 7、micro-batch は 2、`group_size` は 1。`delivery` は `on_demand`。`plan_fixed_pair` は試験の比較にだけ使い、DataModule の本番経路では呼ばない。

`test_waveforms_match_plan_fixed_pair_for_both_normalizers` は、MP-SENet の `noisy_rms_unit_v1` と MossFormerGAN の `independent_audio_norm_v1` を、epoch 0 と 1 で `plan_fixed_pair` と `render_sample` と比べた。noisy と clean の最大絶対誤差は 0 だった。`ReproductionTrainingData` の生成回数は 0 だった。

順序は `rank_order` と一致した。MP-SENet の world size 1 は `fixed_single_shuffle_v1` で、epoch 0 と 1 の case 列は同じだった。MossFormerGAN は `rank_mod_fixed_v1` だった。

MP-SENet の検証は rank 0 が valid を全文、`crop-full` で返す。epoch 0 と 1 の入力の最大絶対誤差は 0 だった。rank 1 の iterator は空だった。MossFormerGAN の epoch 1 の検証は、`validation_refs` と `render_sample` に対する noisy と clean の最大絶対誤差が 0 だった。

world size 2 は偽の runtime で、rank 0 と rank 1 の `plan_hash_for_state({"epoch": 0})` が一致することを見た。ProcessRuntime と Gloo は使っていない。これは Phase 3-E の正式受け入れではない。

`plan_hash` の payload は、epoch、world size、protocol id、snapshot hash、seed、order policy、order params、micro-batch、group size、`weights_sha256="none"`、全 rank の残した SampleRef である。micro-batch と group size の端数で落とした尾は hash に入らない。worker 数、prefetch、device、消費済み cursor は入らない。旧 `hash_plan` との一致は受け入れに含めていない。

## 6. Factory と Builder

試験が書く `experiment.yaml` が、読み込める一式である。モデルは `tests.experiments.support.IdentityWave`、損失は `torch.nn.L1Loss` か `BoomLoss`、runtime は CPU FP32 である。SE の `build_model` は呼ばない。source、sampler、protocol、batcher は `role: other` で、StateRegistry には入らない。DataModule だけが `role: data` である。依存は YAML の inject で、モデル名の分岐は無い。

`test_factory_builds_components_without_duckdb_or_training_data` は、`duckdb.connect` と `ReproductionTrainingData` の呼び出しが 0 であること、data が StateRegistry に入ること、other の 4 つが入らないこと、`close` のあと `dataset()` が `DataError` になることを確認した。`TensorPairData`、`Batch`、`EpochView` の import も確認した。

`lineage()` は `snapshot_provenance_parts` の戻りを返す。provenance ファイルは書かない。

## 7. Resume

中断側は `stop_after_logical_steps: 2`、prefetch on、最大 4 step の CLI である。`step_00000002.pt` があり、`complete.json` は無かった。そのあと同じ出力ディレクトリへ resume 用 YAML を書き、別プロセスで再開した。連続実行は別ディレクトリで prefetch off である。metrics の index 2 の sample id は、再開と連続実行で一致した。

同じ checkpoint をプロセス内で戻すと、`update_index` は 2、epoch は 0 で、`batch(2)` の波形は prefetch off の新しい DataModule と最大絶対誤差 0 だった。prefetch on と off の `batch(0)` も最大絶対誤差 0 だった。保存ファイルの `plan_hash` を `"f" * 64` にすると、復元は `CheckpointError` になった。

`initialize_from` は、保存時の `update_index` が 2 でも、新しい実行の data は `update_index` 0、epoch 0 のままだった。許可した module は `model` だけである。

## 8. 例外と異常系

測ったものは次である。

- 存在しない selector `missing.dataset@1` の `open` は `DataError`。
- 範囲外の添字は `StateError`。
- 表示名だけの identity 変更は `load_state_dict` が受ける。`snapshot_hash`、`dataset_id`、`plan_hash` の改ざんは `StateError`。
- `official_faithful` の snapshot に `require_profile: leakage_safe` を付けると `BuildError` で、原因は `DataError`。部分レジストリの data は閉じ、Source の `dataset()` は `DataError` になる。profile を要求しないときは `selection_uses_test` が true のまま setup できる。
- preflight のあと `close` すると Source は閉じる。1 step の正常終了でも閉じる。`BoomLoss` の `RuntimeError` のあとでも閉じる。`accumulation` 2 と `group_size` 1 は `RuntimeFailure` で、そのあと Source は閉じる。
- すでに CPU にある batch は、`_batch_on_device` が同じ tensor オブジェクトを返す。

## 9. 回帰テスト

変更前:

| 対象 | 結果 |
|---|---|
| `tests/experiments` | Ran 43 tests in 113.914s、OK (skipped=1)、EXIT 0。ログ `/tmp/phase3b-experiments-before.txt` |
| `tests/reproduction` | Ran 135 tests in 463.342s、FAILED (errors=1, skipped=2)、EXIT 1。ログ `/tmp/phase3b-reproduction-before.txt` |

事前の reproduction の 1 error は `test_p8_dfn3.TrainerSmokeTests.test_one_deepfilternet_step_is_finite` で、`libdf_rfft.so` が GLIBC_2.33 を要求して読めない。この段の差分より前からある。planner の公開委譲は式を変えていない。

変更後の `tests/experiments` は、Ran 54 tests in 122.489s、OK (skipped=1)、EXIT 0。ログ `/tmp/phase3b-experiments-after.txt`。増分 11 件は `tests/experiments/test_fixed_pair_data.py` で、失敗は 0、新規 skip は 0、NOT_RUN は 0 である。既存の skip 1 は変更前からある。

変更後の `tests/reproduction` は、Ran 135 tests in 463.928s、FAILED (errors=1, skipped=2)、EXIT 1。ログ `/tmp/phase3b-reproduction-after.txt`。error は事前と同じ `test_one_deepfilternet_step_is_finite` で、`libdf_rfft.so` が読めない。errors と skipped は事前から増えていない。失敗の新規追加は 0、skip の新規追加は 0、NOT_RUN は 0 である。

GPU 試験は必須ではない。実行していない。

## 10. Phase 3-C への残課題

- chunk の Sampler と Protocol。
- DFN3 のオンライン混合と、SEMamba++ のオンライン劣化。
- 動的な重み付き Sampler。
- prefetch の worker プール。
- world size 2 の Data System 正式受け入れ。今回の偽 runtime の hash 一致は、その受け入れではない。
- 公式 16 kHz 変換器と、実在の VoiceBank snapshot に対する一致。今回の波形一致は、一時ディレクトリの 16 kHz ファイルと `render_sample` の間である。
- SE モデル、GAN Trainer、公式複合損失、長い GPU 学習、FSDP、sharded checkpoint。

## 11. 判定

**GO。**

公開 API で `SnapshotDataModule` を構築し、固定ペア 2 種の crop、順序、波形が純関数と最大絶対誤差 0 で一致した。identity、plan hash、micro-batch cursor の拒否と受理を確認した。CPU の別プロセス再開で、次の sample id と波形が連続実行と一致した。`tests/experiments` は 54 件 OK、skip 1 で、失敗は 0 である。`tests/reproduction` の error は事前と同じ 1 件のままである。`close` は正常終了、例外、preflight、group size 不一致、構築失敗のあとで Source を解放した。

公式 SE の学習再現完了ではない。公式 16 kHz 変換器、実在 snapshot、world size 2 の Data System 正式受け入れは測っていない。
