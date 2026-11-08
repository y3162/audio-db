# P10-C Fix.2-E 失敗分析

日付: 2026-11-08。ホスト `tensor`。修正前の HEAD `cc6ae84a0f017da01e7f23c96061a0359df6cff4`、ブランチ `feature/reproduction`。作業木はクリーンだった。

再現は、Fix.2-D の最終ログを根拠に直さず、同じ 11 件をこの作業木で実行してから分類した。ログは `/tmp/fix2e-repro.log`。終了コード 1。20 分 17 秒。11 failed、3 passed、13 subtests passed。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m pytest \
  tests/training/test_df_rfft_kernel.py::RealFftKernelTest::test_unnormalized_fft_matches_sum_and_nyquist \
  tests/training/test_model_tasks.py::DeepFilterTests::test_deepfilter_cosine_matches_the_shipped_first_step \
  tests/training/test_model_tasks.py::DeepFilterTests::test_official_width_step_reloads \
  tests/training/test_model_tasks.py::DeepFilterTests::test_valid_reduction_trains \
  tests/training/test_tensorboard_integration.py::TensorBoardModelIntegration::test_models \
  tests/training/test_tensorboard_integration.py::TensorBoardDDPIntegration::test_other_real_models_two_ranks_validation \
  tests/training/test_tensorboard_scalars.py::OtherModelPreviewTest::test_small_models_forward \
  tests/reproduction/test_p4.py::RealSnapshotSmokeTests::test_four_datasets_round_trip_provenance \
  tests/reproduction/test_p8_dfn3.py::SnapshotTests::test_smoke_snapshot_is_repeatable_and_official_stays_blocked \
  tests/reproduction/test_p8_dfn3.py::TrainerSmokeTests::test_schedule_validation_checkpoint_and_resume \
  tests/reproduction/test_p8_dfn3.py::TrainerSmokeTests::test_one_deepfilternet_step_is_finite \
  -q --tb=line -p no:cacheprovider \
  --basetemp=/tmp/pytest-fix2e-repro
```

## 11 件

| 試験 | 例外 | 分類 |
|---|---|---|
| `test_df_rfft_kernel.py::RealFftKernelTest::test_unnormalized_fft_matches_sum_and_nyquist` | `OSError: GLIBC_2.33 not found` のあと `RuntimeError: failed to load .../libdf_rfft.so` | 環境。実装 BUG ではない |
| `test_model_tasks.py::DeepFilterTests` の 3 件 | 同じ `RuntimeError` | 環境 |
| TensorBoard の `deepfilternet3` subtest 3 件 | 同じ。DDP は両 rank が同じ `OSError` で落ち、親は `distributed workers failed` | 環境 |
| `test_p4.py::RealSnapshotSmokeTests::test_four_datasets_round_trip_provenance` | `AssertionError: 'eval' unexpectedly found in {'pg_catalog', 'information_schema', 'main', 'eval', 'corpora'}` | 試験の前提が現行 DB と違う。製品の学習アルゴリズムではない |
| `test_p8_dfn3.py` の snapshot 1 件 | `IOException: Cannot open .../reproduction-p8.duckdb ... database does not exist` | データ欠落 |
| `test_p8_dfn3.py` の Trainer 2 件 | `ReproductionError: found 0 snapshots for dfn3.online.smoke@1` | データ欠落 |

4 モデルの学習経路はこの 11 件に入っていない。

## DeepFilterNet3

追跡ファイル `src/se/models/deepfilternet3/libdf_rfft.so` を `ldd` すると、このホストの glibc 2.31 に対して `GLIBC_2.33` と `GLIBC_2.34` が無い。`objdump -T` の要求もそこまで上がる。`DF_RFFT_LIBRARY` は未設定で、`rfft_library_path()` はこのファイルを読む。`ctypes.CDLL` が失敗し、`dsp.py` が `RuntimeError` にする。欠落時の fallback には入らない。ファイルは存在するが ABI がホストと合わない。

同じソースの `make df-rfft` 出力 `build/df-rfft/release/libdf_rfft.so` は、このホストで `ldd` が成功する。要求は `GLIBC_2.28` まで。BuildID は `d86ed921dbd4bf01e4673429e2b5971df229560d` で、`build/df-rfft-shannon-20261105/release/libdf_rfft.so` と同じである。ABI は 1。入力 `[1, 2, 3, 4]` の非正規化 RFFT は `(10, 0)`、`(-2, 2)`、`(-2, 0)` で、既存の数値試験と一致した。追跡ファイルは置き換えていない。

## P4

`assertNotIn("eval", schemas)` は、2026-11-06 に `resource.duckdb` が `eval` を持たなかったときの前提である。現行ファイルはレシピスキーマで、`corpora` と `eval` がある。来歴判定そのものは `judge_independence(document, None)` で、評価材料が無いので `comparison` は `NOT_TESTED` になる。スキーマの不在がこの判定を作っているわけではない。試験が続けて見ているのは、4 Dataset の来歴、identity、snapshot hash、`selection_uses_test`、claim、profile、分割の独立性、および DB ファイルの不変である。

履歴の 16 kHz 名 `mp_senet.vb28.official@1` などは、この snapshot root に無い。学習入口のオンデマンド official 2 件はある。safe 2 件は無かった。

## P8

`data/db/reproduction-p8.duckdb` と `data/staging/reproduction-p8/` は無く、`dfn3.online.smoke@1` の snapshot も無かった。準備関数は `src/reproduction/prepare/dfn3_online.py` の `register_p8` である。`reproduction.duckdb` と `resource.duckdb` は開かない。公式 Dataset は `blocked` のまま登録する。

## 対応方針

- DeepFilterNet3 は、追跡 `.so` を残し、`make df-rfft` の出力を `DF_RFFT_LIBRARY` で試験に渡す。
- P4 は、`eval` が無いことを要求せず、試験前後でスキーマ一覧とファイルが同じであることを確認する。来歴の検査は残す。読む Dataset は、現在の snapshot root にあるオンデマンド固定ペアにする。
- P8 は `register_p8` で欠落している DB と snapshot を作る。既存の snapshot ディレクトリは置き換えない。
