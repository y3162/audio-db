.PHONY: db-ingest db-init

db-ingest:
	uv run python -m src.data.ingest_corpora

db-init:
	.venvs/lab/bin/python -m src.data.init_eval
