.PHONY: db-init db-ingest db-parse

db-init:
	uv run python -m src.init_db

db-ingest:
	uv run python -m src.ingest_corpora --replace

db-parse: db-ingest
