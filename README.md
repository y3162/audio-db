# audio-db

---

### 仮想環境の作成方法

```bash
curl -fsSL https://install.duckdb.org | DUCKDB_VERSION=2.0.0 bash
```

```bash
uv venv .venv
source .venv/bin/activate
uv pip install duckdb
```

---

### コーパスのシンボリックリンク

```bash
ln -s /path/to/corpora/data/corpus_name/ ./data/corpora
```

---

### DB への投入

```bash
make db-ingest
```
