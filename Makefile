.PHONY: db-ingest db-init db-link db-bootstrap

db-link:
	.venvs/lab/bin/python -m src.data.link_corpora

db-ingest:
	uv run python -m src.data.ingest_corpora

db-init:
	.venvs/lab/bin/python -m src.data.init_eval

db-bootstrap:
	.venvs/lab/bin/python -m src.data.bootstrap
