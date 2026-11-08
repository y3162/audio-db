# P10-C Fix.2-E 実装

`src/` は変更していない。4 モデルの学習アルゴリズム、損失、Optimizer、checkpoint 選択は変えていない。公式ソースも変えていない。

## 変更ファイル

| ファイル | 内容 |
|---|---|
| `tests/conftest.py` | 追跡 `libdf_rfft.so` が読めず `DF_RFFT_LIBRARY` が空のときだけ、`build/df-rfft/release/libdf_rfft.so` をその変数へ入れる |
| `tests/reproduction/test_p4.py` | `eval` の不在の代わりに、スキーマ一覧とファイルの不変を確認する。Dataset 名をオンデマンド固定ペアにする |
| `Makefile` | `reproduction-register-p8` を足す |
| `.agents/DEVELOPMENT.md` | pytest のライブラリ選択、P8 の make、safe の leak_check |
| `.agents/DECISIONS.md` | 現行 `resource.duckdb` に `eval` があるという前提 |
| `.agents/DATA.md` | オンデマンド再構築先と、safe の leak_check が UNKNOWN であること |

## DeepFilterNet3

`pytest_configure` が追跡ファイルを `ctypes.CDLL` で試し、失敗したときだけリポジトリ相対の `build/df-rfft/release/libdf_rfft.so` を読む。既に `DF_RFFT_LIBRARY` があるときは触らない。ビルドが無いときは変数を足さず、試験は失敗したままである。SKIP にはしていない。子プロセスは親の環境を継承する。

`make df-rfft` は、追跡ファイルと内容が違うとき追跡ファイルを置き換えない。今回も置き換えていない。

`unittest discover` は `conftest.py` を読まない。その経路では、同じ相対パスを `DF_RFFT_LIBRARY` に置く。

## P4

`RealSnapshotSmokeTests` は、試験前の `(size, mtime_ns)` と `information_schema.tables` の `(table_schema, table_name)` を取り、来歴を書いたあと同じであることを要求する。`resource.duckdb` は読み取り専用のままである。`reproduction.duckdb` は stat だけを見る。

来歴、Dataset 名、snapshot hash、`selection_uses_test`、profile、claim、`split_independent`、`independent == UNKNOWN`、`comparison == NOT_TESTED` は残している。評価材料は `None` のままなので、`NOT_TESTED` は「材料を渡していない」の意味である。

履歴の 16 kHz Dataset 名は snapshot root に無い。試験が読むのは次の 4 件である。

- `mp_senet.vb28.ondemand.official@1`
- `mp_senet.vb28.ondemand.safe@1`
- `mossformergan.vb28.ondemand.official@1`
- `mossformergan.vb28.ondemand.safe@1`

official の profile、claim、`selection_uses_test = true` は、履歴名に対する期待と同じである。safe も `leakage_safe`、`leakage_safe_comparison`、`selection_uses_test = false` のままである。オンデマンドの素材は `corpora/` の 48 kHz で、snapshot の lineage は `source` である。`unknown == 0`、`source > 0`、`derived == 0` を確認する。履歴の 16 kHz 派生素材に対する `derived > 0` は、この Dataset には当てはまらない。

## P8

`make reproduction-register-p8` は `register_p8()` を呼ぶ。登録先は `data/db/reproduction-p8.duckdb`、staging は `data/staging/reproduction-p8/`、snapshot は `data/db/snapshots/<dataset_id>/` である。`resource_db=None` なので `resource.duckdb` は開かない。`reproduction.duckdb` も開かない。

再実行の挿入件数は 0 で、snapshot hash は同じだった。

## VoiceBank の safe snapshot

`make reproduction-prepare` は現行 `resource.duckdb` を読む。safe の `leak_check` は、test 素材の SHA-256 が `eval.cases` と重なるため `eval_sets` が FAIL になり、登録前に止まる。製品の判定は変えていない。

試験用の 4 Dataset は、`resource_db` を渡さない `PreparePaths` で `data/db/reproduction-vb-ondemand.duckdb` と `data/staging/reproduction-vb-ondemand/` に登録した。この経路の leak_check は UNKNOWN である。PASS ではない。official の snapshot hash は既存ディレクトリと一致したので、そのディレクトリはコピーしていない。safe の 2 ディレクトリだけを、無かった ID として `data/db/snapshots/` に追加した。

## 既存資産

試験の前後で、次の SHA-256 は同じだった。

| ファイル | SHA-256 |
|---|---|
| `data/db/resource.duckdb` | `3998f5ae86b3d1b7a3809bdff20f2564a8e90859b0e237f2191150db9afacb6a` |
| `data/db/reproduction.duckdb` | `c4f8f786c67d0ade73ef8be3eb46d490c12f688ab970feaa01c46745e4d32ca0` |

全件回帰の前後でも、この 2 ファイルと `reproduction-p8.duckdb` の SHA-256、および snapshot ディレクトリの集合は変わっていない。既存の official snapshot は置き換えていない。checkpoint と既存 run には書いていない。
