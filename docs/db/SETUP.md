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

`make db-init` は既存の `corpora.audio_files` を変更しません。`eval` スキーマを作り、`configs/db/init.yaml` に書いた Manifest、コーパス選択、CHiME3 の公式アノテーション、MP-SENet の VoiceBank-DEMAND リストを登録します。同じコマンドを重ねて実行しても行は増えません。音声や Manifest が無い項目は、`required: false` なら理由を出してスキップします。内容が違う再登録はエラーになり、一部だけは残りません。

```bash
make db-init
```

追加するテーブルは次のとおりです。

| テーブル | 役割 |
|---|---|
| `eval.cases` | 評価する音声と、合成する場合の SNR・seed・切り出し位置 |
| `eval.sets` | 評価セットの名前と説明 |
| `eval.set_items` | セットに含まれる case と順序 |

パスはデータベースファイルからの相対パスです。評価の取り出し方は [評価パイプライン](../evaluation/PIPELINE.md) にあります。