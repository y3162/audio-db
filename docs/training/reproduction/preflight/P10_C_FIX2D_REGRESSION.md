# P10-C Fix.2-D 回帰

日付: 2026-11-08。仮想環境 `.venvs/se`。`CUDA_VISIBLE_DEVICES` は空。GPU は使っていない。

## 収集

指定のコマンドを、準備スクリプトを直す前に一度実行した。

```bash
.venvs/se/bin/python -m pytest tests/training tests/reproduction -q --tb=line
```

終了コード 3。`tests/training/reference/prepare_voicebank_test.py` が import 時に `SystemExit` し、収集が落ちた。このファイルはテストではなく、既存の staging があると上書きを拒む準備スクリプトである。本体を `main` に入れ、`__name__ == '__main__'` のときだけ実行するようにした。スクリプトとして実行したときの拒否は残している。

## 最初の全件

修正後の同じ範囲。終了コード 1。39 分。

| 結果 | 件数 |
|---|---:|
| passed | 403 |
| failed | 19 |
| skipped | 14 |
| subtests passed | 40 |

失敗の内訳は次のとおりである。

| 分類 | 件数 | 内容 |
|---|---:|---|
| DeepFilterNet3 の共有ライブラリ | 7 | `libdf_rfft.so` が `GLIBC_2.33` を要求し、このホストでは読めない。モデル試験 4 件と TensorBoard の deepfilternet3 subtest 3 件 |
| MossFormer の spawn | 7 | 子プロセスが `No module named 'tests.training'`。単体で再実行すると 8 件は PASS |
| TensorBoard プレビュー文 | 1 | 学習状態の一致のあとに、例外文が `spec` であることを要求していた。実際は mock に `_generate` が無い |
| `resource.duckdb` の `eval` | 1 | `tests/reproduction/test_p4.py` が `eval` スキーマの不在を期待し、現行 DB にはある |
| P8 の欠落 | 3 | `data/db/reproduction-p8.duckdb` が無く、`dfn3.online.smoke@1` の snapshot も無い |

14 件の SKIP は PASS に数えない。公式 ESPnet の checkout が無い試験を含む。SKIP へ変更していない。

## 修正

- プレビュー試験の期待文を、MP-SENet 経路が呼ぶ `_generate` の欠落に合わせた。状態一致の検査は残している。単体は PASS。
- spawn の直前にリポジトリルートを `sys.path` の先頭へ置く `pin_repo_root_for_spawn` を足した。子プロセスは親の `sys.path` をコピーするため、先に走った試験が checkout を残すと `tests` が別物になる。

DeepFilterNet3 の `.so` は再ビルドしていない。P8 の DB と snapshot は作っていない。`resource.duckdb` は変更していない。これらを 4 モデルのコード欠陥としては扱わない。

## 最終の全件

子プロセスが、`tests/__init__.py` を持つ checkout を親の `sys.path` から外してから `tests.training` を import する。親の `sys.path` からは消さない。消すと、後続の SEMamba++ 試験が `resampy` を見失う。

正本は次の実行である。終了コード 1。41 分 34 秒。ログは `/tmp/fix2d-pytest-final5.log`。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m pytest \
  tests/training tests/reproduction \
  -q --tb=line -p no:cacheprovider \
  --basetemp=/tmp/pytest-fix2d-final5
```

| 結果 | 件数 |
|---|---:|
| passed | 411 |
| failed | 11 |
| skipped | 14 |
| subtests passed | 40 |

失敗は次だけである。

| 試験 | 原因 |
|---|---|
| `test_df_rfft_kernel.py` の 1 件、`DeepFilterTests` の 3 件、TensorBoard の deepfilternet3 subtest 3 件 | `libdf_rfft.so` が `GLIBC_2.33` を要求する。このホストでは読めない |
| `test_p4.py::test_four_datasets_round_trip_provenance` | 現行 `resource.duckdb` に `eval` がある。試験は無いことを期待する |
| `test_p8_dfn3.py` の 3 件 | `reproduction-p8.duckdb` と `dfn3.online.smoke@1` の snapshot が無い |

MossFormer の spawn 7 件、TensorBoard のプレビュー文、P9 の `resampy` は、この実行では失敗していない。SKIP は PASS に数えない。DeepFilterNet3 と P8 の欠落は、4 モデルの学習経路の FAIL ではない。

## 共通 Trainer

最終スイートに含まれる範囲で、失敗一覧に出ていなければ PASS として扱う。対象は次のとおりである。

- 原子的 checkpoint と、保存失敗時の前ファイル保護
- latest 3、best、n-best、他 run の保護
- resume の dataset、snapshot、config、world size、batch plan
- `rng_by_rank` の world 1 と、欠落した複数プロセス checkpoint の拒否
- `validation_pending` と、optimizer を進めない再検証
- TensorBoard の scalar と、プレビュー失敗が学習状態を変えないこと

GPU 上の world 2 と world 4 の `rng_by_rank` は [GPU 検証](P10_C_FIX2D_GPU_VERIFICATION.md)。`ave_5best` の数値は同じ文書の fixture。

Preflight は 4 本格 YAML が PASS。記録は [検証マトリクス](P10_C_FIX2D_TEST_MATRIX.md)。
