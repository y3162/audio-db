# P10-C Fix.2-E 回帰

仮想環境 `.venvs/se`。GPU は使っていない。`CUDA_VISIBLE_DEVICES` は空。

## 修正前

対象 11 件。ログ `/tmp/fix2e-repro.log`。終了コード 1。1217.66 秒。

| 結果 | 件数 |
|---|---:|
| passed | 3 |
| failed | 11 |
| subtests passed | 13 |

失敗の内訳は [失敗分析](P10_C_FIX2E_FAILURE_ANALYSIS.md)。Fix.2-D の全件は 411 passed、11 failed、14 skipped、40 subtests passed、終了コード 1、2494.63 秒だった。今回の修正前再実行は、その 11 件だけを同じ例外で再現している。

## 個別

| 範囲 | 結果 | 時間 | ログ |
|---|---|---:|---|
| P8 の 3 件 | 3 passed | 26.12 秒の実行に含まれる | `/tmp/fix2e-p4p8.log` |
| P4 の実スナップショット試験 | 最初は lineage の `derived` が 0 で失敗。オンデマンドの `source` に合わせて再実行し 1 passed | 9.65 秒 | `/tmp/fix2e-p4.log` |
| DeepFilterNet3 の kernel、モデル 4 件、TensorBoard 3 試験 | 10 passed、16 subtests passed、終了コード 0 | 1247.88 秒 | `/tmp/fix2e-dfn.log` |

P4 の最初の失敗は、オンデマンド素材が `corpora/` の source lineage であるためである。`derived > 0` は履歴の 16 kHz 派生素材の期待で、この snapshot には無い。`unknown == 0` と `source > 0` に変えたあとの単体は PASS である。

## 最終全件

終了コード 0。pytest の計測は 2472.09 秒（41 分 12 秒）。壁時計は 2475 秒。ログは `/tmp/fix2e-final.log`。

```bash
env CUDA_VISIBLE_DEVICES= .venvs/se/bin/python -m pytest \
  tests/training tests/reproduction \
  -q --tb=line -rs -p no:cacheprovider \
  --basetemp=/tmp/pytest-fix2e-final
```

| 結果 | 件数 |
|---|---:|
| passed | 419 |
| failed | 0 |
| error | 0 |
| skipped | 14 |
| subtests passed | 43 |

411 に、通常試験の失敗 8 件が戻った分を足すと 419 である。残る 3 件は TensorBoard の `deepfilternet3` subtest で、subtests passed が 40 から 43 になっている。新しい FAIL と ERROR は無い。

この実行の前後で、`resource.duckdb`、`reproduction.duckdb`、`reproduction-p8.duckdb` の SHA-256 は同じである。snapshot ディレクトリの集合も同じである。

## 14 SKIP

SKIP へ変更した試験は無い。Fix.2-D と同じ 14 件である。

| 試験 | 理由 | 区分 | 4 モデルの開始 |
|---|---|---|---|
| `test_chunk_official.py` の 4 件 | pinned ESPnet source が無い | 公式 checkout 不足 | TF-GridNet の公式 iterator 照合は未実施。このスイートの学習試験とは別 |
| `test_dataset.py` の 1 件 | rebuild database が無い | データ不足 | 開始条件ではない |
| `test_df_libdf_frontend.py` の 2 件 | `libdf` が import できない | 任意依存の不足 | DeepFilterNet3 の公式波形照合。4 モデルの外 |
| `test_df_reference_spectral.py` の 1 件 | CUDA が要る | GPU 必須 | 4 モデルの外 |
| `test_moss_probe_accumulation.py` の 1 件 | CUDA が要る | GPU 必須 | MossFormerGAN の CUDA 蓄積プローブはこの CPU 実行に入っていない |
| `test_shared_fork.py` の 2 件 | `mamba_ssm` が無い | 任意依存の不足 | SEMamba++ の公式照合。4 モデルの外 |
| `test_tf_locoformer_best.py` の 1 件 | `torch_complex` が無く公式損失を import できない | 公式 checkout 不足 | 独立実装の試験は PASS。公式損失との照合は未実施 |
| `test_p2.py` の 1 件 | 保存済み VoiceBank 派生素材が登録されていない | データ不足 | オンデマンド入口とは別 |
| `test_p3.py` の 1 件 | 履歴の P2 snapshot がこの機械に無い | データ不足 | オンデマンド official の snapshot は別にある |

SKIP は PASS ではない。4 モデルの本格 epoch は、この回帰では始めていない。
