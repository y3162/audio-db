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

`data/db/resource.duckdb` はレシピスキーマです。切り替え前の固定列ファイルは `data/db/backups/resource-legacy-20261031.duckdb` です。SHA-256 は `f3fdcd89b34cf44eda9853e71278806c1ecc4042f03173821805be054b61d156` で、同じディレクトリの `resource-legacy-20261031.sha256` にも書いてあります。戻すときは、開いている接続が無いことを確認してから、このファイルを `data/db/resource.duckdb` へコピーします。`rebuild-a.duckdb` と `rebuild-b.duckdb` は切り替え前の再構築コピーです。`recipe_json` の無い `eval.cases` に対する `make db-init` は、スキーマを検知して拒否し、中身は変えません。

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

`make db-bootstrap` は空の別ファイルへ 12 コーパスを索引し、`configs/db/init.yaml` の評価セットを登録する。出力先が既にあるときは `--rebuild` と別の未使用 `--backup` パスが必要である。候補は出力先と同じファイルシステム上の一時ディレクトリに空から作り、検証後に退避して切り替える。本番ファイルへの反映は `--replace` で、新しい DB の検証、開いているプロセスの確認、バックアップ、同一ファイルシステム上の `os.replace` の順で行う。

現行の本番再構築元は `data/staging/resource-current` である。`data/staging/` 直下や `resource-full` は過去の再構築元であり、現行の本番には使わない。

`data/staging/` 以下は DuckDB の再構築元です。登録が終わっても消しません。Git 管理外なので、DB と一緒にバックアップします。

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

以前の `eval.cases` からの一度きりの変換は `python -m src.data.migrate_legacy` です。出力先が既にあるときは上書きしません。実行時に旧列を読む互換層ではありません。本番の切り替えは済んでいます。

`--backup` は上書きしない。候補のコピーと検証の失敗時には既存 DB を維持し、通常の失敗時は incoming ファイルを消す。プロセス強制終了やホスト障害では `.incoming-*` や `.build-*` が残り得るため、他の作業が使っていないことを確認してから手動で処理する。置換前に検証と fsync を行い、置換後にディレクトリを fsync する。異常終了後は対象とバックアップをそれぞれ検証し、バックアップからの復旧は別ファイルにコピー・検証してから行う。`/proc` の open 検査は他ホストや権限で見えないプロセス、検査後の新規 open を排除できないため、共有ファイルシステム上では全利用者の停止を別途確保する。候補と staging 全体の比較や transcript の完全性はこの検証だけでは保証しない。
