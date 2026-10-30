.PHONY: db-ingest

db-ingest:
	uv run python -m src.data.ingest_corpora
