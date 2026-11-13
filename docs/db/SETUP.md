# データベース

評価用は `data/db/resource.duckdb`、再現用は `data/db/reproduction.duckdb` である。学習プロセスは `reproduction.duckdb` を開かず、`data/db/snapshots/<dataset_id>/` を読む。

- レシピ: [RECIPE.md](RECIPE.md)
- コーパス: [CORPORA.md](CORPORA.md)
- データセットの説明: [DATASETS.md](DATASETS.md)
- 学習スモークセット: [TRAINING_SETS.md](TRAINING_SETS.md)
- `corpora.audio_files`: [tables/AUDIO_FILES.md](tables/AUDIO_FILES.md)
- 再現用 DB の仕様: [REPRODUCTION_DB_DESIGN.md](REPRODUCTION_DB_DESIGN.md)
- ファイルの配置と構築記録: [INVENTORY.md](INVENTORY.md)

`make db-ingest` は `corpora.audio_files` を削除して作り直す。実行前に `resource.duckdb` を退避する。ユーザーが明示したときだけ実行する。

## 操作

### コーパスのシンボリックリンク

```bash
export AUDIO_DB_DATASET_ROOT=/path/to/datasets
python -m src.data.link_corpora
```

---

### コーパスの投入

転送と完全性の確認後のみ実行する。`make db-ingest` は `corpora.audio_files` を削除して作り直すため、既存 DB のバックアップと明示的な実行指示が必要です。評価セットの初期化からは呼びません。

```bash
make db-ingest
```

---

### 評価セットの初期化

`make db-init` は既存の `corpora.audio_files` を変更しません。空の DB、またはレシピスキーマの DB に対して `eval` スキーマを作り、`configs/db/init.yaml` に書いた Manifest、コーパス選択、CHiME3 の公式アノテーション、MP-SENet の VoiceBank-DEMAND リストを登録します。同じコマンドを重ねて実行しても行は増えません。音声や Manifest が無い項目は、`required: false` なら理由を出してスキップします。内容が違う再登録はエラーになり、一部だけは残りません。

`data/db/resource.duckdb` はレシピスキーマです。`recipe_json` の無い `eval.cases` に対する `make db-init` は、スキーマを検知して拒否し、中身は変えません。

切り替え前の固定列ファイルへ戻すときは、開いている接続が無いことを確認してから、そのバックアップを `data/db/resource.duckdb` へコピーします。ファイルのパス、SHA-256、いま在るかは [配置と記録](INVENTORY.md) です。

```bash
make db-init
```

追加するテーブルは次のとおりです。

| テーブル | 役割 |
|---|---|
| `eval.cases` | `case_id`、種類、データセット、レシピ JSON、内容ハッシュ、metadata |
| `eval.sets` | 評価セットの名前、説明、内容ハッシュ |
| `eval.set_items` | セットに含まれる case と順序 |
| `eval.runs` | 評価ランの設定。一次記録は `data/results/` のファイル |
| `eval.results` | 発話ごとの結果。`record_id` が主キー |

パスはデータベースファイルからの相対パスで、レシピの中に入ります。生成条件の形は [レシピ](RECIPE.md)、評価の取り出し方は [評価パイプライン](../evaluation/PIPELINE.md) です。

## ステージング

コーパス索引は元の sample rate を保存する。モデルが別の sample rate を使うときは、学習 Dataset のロード時に変換する。レシピの `case_id` は変えない。

`make db-bootstrap` は空の別ファイルへカタログの全コーパスを索引し、`configs/db/init.yaml` の評価セットを登録する。出力先が既にあるときは `--rebuild` と別の未使用 `--backup` パスが必要である。候補は相対音声パスが変わらないよう出力先と同じディレクトリに空から作り、検証後に退避して切り替える。本番ファイルへの反映は `--replace` で、新しい DB の検証、開いているプロセスの確認、バックアップ、同一ファイルシステム上の `os.replace` の順で行う。実装は `src/data/bootstrap.py` である。

`--backup` は排他作成で、既存のバックアップは上書きしない。候補は `.build-*` に作り、本番へ置く前に `.incoming-*` へコピーして fsync し、もう一度検証する。検証やコピーの失敗では本番を置き換えず、通常の失敗では incoming を消す。プロセス強制終了では `.incoming-*` や `.build-*` が残り得る。他の作業が使っていないことを確認してから処理する。置換のあと、親ディレクトリを fsync する。`/proc` の open 検査は、他ホストのプロセスや、検査のあとの新規 open を排除できない。共有ファイルシステムでは、全利用者の停止を別に確保する。この検証は、候補と staging 全体のバイト比較や、transcript の完全性までは見ない。

現状の bootstrap は索引と評価セットを DB に直接書き、対応する永続 staging を自動生成しない。staging を先に保存する本リポジトリの要件を満たす再構築手順としては未完成であり、本番再構築には使わない。

どのディレクトリが現行の再構築元かは [配置と記録](INVENTORY.md) にあります。

`data/staging/` 以下は DuckDB の再構築元です。登録が終わっても消しません。Git 管理外なので、DB と一緒にバックアップします。評価用 staging の形は `src/data/staging.py` が書く次の並びです。`data/staging/` 直下が、いまこの並びであるとは限りません。

```text
data/staging/
├── metadata.json
├── corpora/audio_files.parquet
├── evaluation/
│   ├── cases.parquet
│   ├── sets.parquet
│   └── set_items.parquet
└── results/
    ├── runs.parquet
    └── results.parquet
```

`src.data.staging.register_staging` がファイルを検証してから、1 トランザクションで登録します。同じ内容は増えず、同じ ID で内容が違うとエラーになり、そのバッチは残りません。`rebuild_database` は新しいファイルを作り、既存ファイルは上書きしません。


## 再現用 DB とスナップショット

`reproduction.duckdb` は評価用 DB とは別ファイルである。同じスキーマへの `make reproduction-init-db` は何もしない。違うスキーマや未完了のファイルは変更せず拒否する。

```bash
make reproduction-doctor
make reproduction-init-db
make reproduction-verify DATASET=name@version HASH=1
make reproduction-snapshot DATASET=name@version
make reproduction-rebuild-db OUT=data/db/reproduction-rebuilt.duckdb
make reproduction-register-p8
make reproduction-register-p9
```

`reproduction-register-p8` は `register_p8()`、`reproduction-register-p9` は `register_p9()` です。既定の出力は `data/staging/reproduction-p8` と `data/db/reproduction-p8.duckdb`、および `reproduction-p9` の対です。ready な Dataset のスナップショットを `data/db/snapshots/` に書きます。`reproduction.duckdb` と `resource.duckdb` には入れません。仕様と、いまの中身は [再現用 DB](REPRODUCTION_DB_DESIGN.md) と [配置と記録](INVENTORY.md) です。

`DATASET` が無い snapshot、`OUT` が無い rebuild、`DATASET` と `BUILD` のどちらも無い verify は失敗する。再構築は `OUT` が既存なら拒否する。leakage 監査が `UNKNOWN` の Dataset を書き出すときだけ `ALLOW_UNKNOWN_LEAK_CHECK=1` を付ける。このフラグはスナップショットの来歴に残り、ハッシュには入らない。

Dataset と Protocol の定義は `configs/data/` に残す。リストパスは Protocol 文書と登録済み ID に結び付いている。

`reproduction-rebuild-db` は `OUT` と同じ名前の `.partial` に作り、参照 DB と全表のハッシュが一致したときだけ置き換えます。失敗時は `.partial` を消します。強制終了で `.partial` が残ることがあります。
