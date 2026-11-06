# P4 検証記録

2026-11-06。学習・validation・checkpoint 選択に使ったデータの来歴を `data_provenance.json` に残し、評価結果が独立したテストかどうかを true / false / `UNKNOWN` で分ける。P5 以降は行っていない。本学習も行っていない。

## 実行したもの

| 項目 | 結果 |
|---|---|
| `unittest discover -s tests/reproduction` | 49 件 OK（93.7 秒）。P1〜P3 に P4 の単体・統合・実スナップショットスモークが含まれる |
| `tests/evaluation/test_provenance.py` と `tests/evaluation/test_pipeline.py` | 判定 15 件と評価本線は OK。`discover -s tests/evaluation` は 105 件中、失敗 2・エラー 2・skip 1 |
| `tests/training/test_protocol.py`、`test_logical_epoch.py`、`test_legacy_adapter.py`、`test_checkpoint_atomic.py`、`test_mock_trainer.py` | OK。再開と Legacy 黄金比較を含む |
| `tests/training/test_dataset.py` | 1 件 ERROR。`configs/training/tf_gridnet_chunk_seed17.yaml` が無い。P4 以前からの欠落で、今回は直していない |

評価スイートの失敗 4 件は、`configs/evaluation/smoke.yaml` と `configs/evaluation/example.yaml` がリポジトリに無いことによる。来歴フィールドの追加とは別である。

`data/db/resource.duckdb` と `data/db/reproduction.duckdb` は読み取りだけ確認し、サイズと mtime はスモークの前後で変わっていない。`repro.runs` は本番 DB では 0 件のままである。登録試験は一時 DB だけを使う。

## 実スナップショット

4 Dataset を開き、stamp を読んで来歴 JSON を一時ディレクトリへ書いた。学習ステップは回していない。16 kHz 派生素材は、stamp の `output_sha256` がスナップショットのファイルハッシュと一致し、`source_sha256` を得られた。`lineage: unknown` は 0 件である。

件数は role ごとの素材行である。official は test の 824 ペアが selection と test の両方に入るので、行数はユニークファイル数より大きい。

| Dataset | profile | selection_uses_test | 素材行（train / selection / test） | snapshot_hash | 外部評価との `independent` |
|---|---|---|---|---|---|
| `mp_senet.vb28.official@1` | `official_faithful` | true | 23144 / 1648 / 1648 | `ebbe60d377dd93c8f43e641b25bb1ed57d7090f7a64f3ed192066b877953d3bb` | `UNKNOWN`（`NOT_TESTED`） |
| `mp_senet.vb28.safe@1` | `leakage_safe` | false | 21400 / 1744 / 1648 | `ff6e05243a2ad20b3e772e907f049072112a921f986a69a75a47c5281f366087` | `UNKNOWN`（`NOT_TESTED`） |
| `mossformergan.vb28.official@1` | `official_faithful` | true | 23144 / 1648 / 1648 | `23af02cefb602107dee5affc324158865e19189869178d4f888eeca71dde9263` | `UNKNOWN`（`NOT_TESTED`） |
| `mossformergan.vb28.safe@1` | `leakage_safe` | false | 21400 / 1744 / 1648 | `4a0cec401a5a2bbba6a22eac4207883c4dac567c7ca275d4d58092b377f8def6` | `UNKNOWN`（`NOT_TESTED`） |

safe の train / selection / test は内容ハッシュで互いに重ならず、`split_independent` は true である。これはその Dataset の分割の話で、外部の評価セットとの独立ではない。official は `selection_uses_test` が true なので `split_independent` は false である。

`resource.duckdb` のスキーマは `corpora` だけで、`eval` は無い（読み取り専用、2026-11-06）。実在の名前付き評価セットとの突合は `NOT_TESTED` である。仮の評価セットは作っていない。

## 合成データで確認した判定

| ケース | 結果 |
|---|---|
| 選択に使った SHA-256 と同じ素材 | `independent = false`。評価セット名と case set id、ハッシュを記録 |
| 内容と派生元が両方とも attested で、どちらも重ならない | `independent = true` |
| 集合名が同じでハッシュが違う | `independent = true`。名前では重なりにしない |
| 別ファイルでも派生元 SHA-256 が同じ | `independent = false` |
| 評価側の lineage が `unspecified`、または学習側の stamp が無い | `independent = UNKNOWN`。空の重なりリストにはしない |
| 来歴 JSON が無い、壊れている、Legacy | `UNKNOWN` |

`UNKNOWN` のとき `confirmed_independent` は false である。

## 実装の置き場所

来歴 JSON の書き込みは `src/training/provenance.py` で、Trainer が rank 0 から呼ぶ。`save_atomic` の検証は従来どおり checkpoint 本体だけを見る。評価の読み取りと判定は `src/evaluation/provenance.py` にあり、`src.reproduction` を import しない。`repro.runs` は `register_training_run` が学習後に書く。

## 未解決のまま残したもの

P3 から継続する。今回は直していない。

1. MossFormerGAN の `validation_report` は time L1 のままである。公式の GAN・PESQ を含む検証損失ではない。checkpoint 選択の指標が公式と違うことは、来歴 JSON の「どの集合を見たか」とは別問題である。本格学習の前に直す必要がある。
2. 2 GPU の実 DDP は未実施。来歴ファイルは rank 0 だけが書く。CPU の 2 process 試験は、重みの一致までで、来歴ファイルの競合は rank 0 以外が書かないことと排他作成の単体テストで見ている。
3. `configs/training/tf_gridnet_chunk_seed17.yaml` と `tf_gridnet_fixed5000_seed17.yaml` が無く、`tests/training/test_dataset.py` の 1 件が落ちる。
4. 公式 Sampler の乱数列との完全一致は未確認。
5. 実評価セットとの独立性は `NOT_TESTED`。`eval` スキーマが無い。
6. Official Evaluation は [仕様](../../evaluation/OFFICIAL_EVALUATION.md) のみ。未実装。
