.PHONY: db-ingest db-init db-link db-bootstrap df-rfft
PYTHON ?= python

# DeepFilterNet3 analysis kernels. Paths are relative to the repository.
# Override DF_RFFT_TARGET to keep the Cargo output out of this tree.
DF_RFFT_MANIFEST := src/se/models/deepfilternet3/rfft_kernel/Cargo.toml
DF_RFFT_TARGET ?= $(CURDIR)/build/df-rfft
DF_RFFT_SO := src/se/models/deepfilternet3/libdf_rfft.so

df-rfft:
	cargo build --release --locked \
		--manifest-path $(DF_RFFT_MANIFEST) \
		--target-dir $(DF_RFFT_TARGET)
	cp -f $(DF_RFFT_TARGET)/release/libdf_rfft.so $(DF_RFFT_SO)

db-link:
	$(PYTHON) -m src.data.link_corpora

db-ingest:
	$(PYTHON) -m src.data.ingest_corpora

db-init:
	$(PYTHON) -m src.data.init_eval

db-bootstrap:
	$(PYTHON) -m src.data.bootstrap
