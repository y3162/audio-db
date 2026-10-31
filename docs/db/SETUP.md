### コーパスのシンボリックリンク

```bash
ln -s /path/to/corpora/data/corpus_name/ ./data/corpora
```

---

### コーパスの投入

`make db-ingest` は `corpora.audio_files` を作り直してコーパスを入れます。評価セットの初期化からは呼びません。

```bash
make db-ingest
```

---

### 評価セットの初期化

`make db-init` は既存の `corpora.audio_files` を変更しません。空の DB、またはレシピスキーマの DB に対して `eval` スキーマを作り、`configs/db/init.yaml` に書いた Manifest、コーパス選択、CHiME3 の公式アノテーション、MP-SENet の VoiceBank-DEMAND リストを登録します。同じコマンドを重ねて実行しても行は増えません。音声や Manifest が無い項目は、`required: false` なら理由を出してスキップします。内容が違う再登録はエラーになり、一部だけは残りません。

現在の `data/db/resource.duckdb` は、レシピより前の `eval.cases` です。このファイルに対する `make db-init` はスキーマを検知して拒否し、中身は変えません。レシピスキーマのコピーは `data/db/rebuild-a.duckdb` と `data/db/rebuild-b.duckdb` です。置き換えは、退避したあとに明示してから行います。

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

`data/staging/` が DuckDB の再構築元です。登録が終わっても消しません。Git 管理外なので、DB と一緒にバックアップします。

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

以前の `eval.cases` からの一度きりの変換は `python -m src.data.migrate_legacy` です。`resource.duckdb` は読み取り専用で開きます。実行時に旧列を読む互換層ではありません。