.PHONY: db-ingest db-init db-link db-bootstrap
PYTHON ?= python

db-link:
	$(PYTHON) -m src.data.link_corpora

db-ingest:
	$(PYTHON) -m src.data.ingest_corpora

db-init:
	$(PYTHON) -m src.data.init_eval

db-bootstrap:
	$(PYTHON) -m src.data.bootstrap
