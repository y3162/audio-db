# audio-db

---

### 仮想環境の作成方法

```bash
uv venv .venv
source .venv/bin/activate
uv pip install duckdb duckdb-cli
```

---

### コーパスのシンボリックリンク

```bash
ln -s /path/to/corpora/data/corpus_name/ ./data/corpora
```

---

### DB の初期化と投入

```bash
make db-init      # corpora.audio_files テーブルを作成
make db-ingest    # data/corpora を走査して全行を投入（--replace）
```
