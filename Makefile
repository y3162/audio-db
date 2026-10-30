.PHONY: db-init db-ingest db-parse

db-init:
	uv run python -m src.data.init_db

db-ingest:
	uv run python -m src.data.ingest_corpora --replace

db-parse: db-ingest
