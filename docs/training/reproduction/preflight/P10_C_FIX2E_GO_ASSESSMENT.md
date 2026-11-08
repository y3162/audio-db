# P10-C Fix.2-E GO

日付: 2026-11-08。ホスト `tensor`。HEAD `cc6ae84a0f017da01e7f23c96061a0359df6cff4`。総合判定は **GO**。本格学習は始めていない。第三者検証は始めていない。

この GO は、`tests/training` と `tests/reproduction` が終了コード 0 で終わるテスト環境の判定である。4 モデルの公式再現の成功でも、収束後の性能でもない。

## 必須条件

| # | 条件 | 結果 |
|---|---|---|
| 1 | 11 件の原因 | 特定した。glibc、`eval` の前提、P8 の欠落 |
| 2 | 解決できる FAIL | 11 件とも修正後に PASS |
| 3 | 試験の意味 | 来歴、identity、hash、claim、profile、選択、leakage 判定、DB の不変を残した。SKIP と xfail にはしていない |
| 4 | P4 の DB | `eval` の不在ではなく、スキーマ一覧とファイルの不変 |
| 5 | P8 | `make reproduction-register-p8` で再構築でき、記録済み hash と一致する |
| 6 | DeepFilterNet3 | この glibc 2.31 で `build/df-rfft/release/libdf_rfft.so` が読める。追跡ファイルは残した |
| 7 | 元の 11 件 | PASS |
| 8 | 全件の終了コード | 0 |
| 9 | 新しい FAIL / ERROR | 無い。419 passed、14 skipped、43 subtests passed |
| 10 | 4 モデルの関連試験 | このスイートで PASS。下の SKIP は公式照合か CUDA で、失敗に変えていない |
| 11 | 学習アルゴリズム | `src/` は無変更 |
| 12 | 既存資産 | `resource.duckdb` と `reproduction.duckdb` の SHA-256 は開始時と同じ。既存 snapshot は置き換えていない |

## 残るもの

- safe の leak_check は、現行 `resource.duckdb` を渡すと `eval_sets` で FAIL する。今回の snapshot は resource DB を開いていないので UNKNOWN である。PASS とは呼ばない。
- 履歴の 16 kHz Dataset の snapshot は復元していない。P4 が読むのはオンデマンド 4 Dataset である。
- `unittest discover` は `DF_RFFT_LIBRARY` を自動では設定しない。
- TSUBAME と、このホスト以外は NOT_TESTED である。
- 14 SKIP のうち、TF-GridNet の公式 chunk 照合、TF-Locoformer の公式損失 import、MossFormerGAN の CUDA 蓄積プローブは、この CPU 実行では未実施である。

## 4 モデル

MP-SENet、MossFormerGAN、TF-GridNet、TF-Locoformer の学習コードは変えていない。DeepFilterNet3 をこの 4 モデルに加えていない。本格 400 / 121 / 35 / 150 epoch は始めていない。

## 第三者検証

今回は起動していない。再実行は [環境](P10_C_FIX2E_ENVIRONMENT.md) の `make df-rfft`、`make reproduction-register-p8`、および最終の pytest である。確認するのは、追跡 `.so` を上書きしていないこと、`reproduction.duckdb` を変えていないこと、P8 の公式 Dataset が `blocked` のままであること、safe の leak_check を PASS と呼ばないことである。
