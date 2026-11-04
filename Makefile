.PHONY: help setup setup-env assets assets-check doctor test-setup \
	db-ingest db-init db-link db-bootstrap df-rfft

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
WITH_ENV := $(ROOT)/scripts/setup/with_env.sh
PYTHON ?= $(shell if [ -x $(ROOT)/.venvs/se/bin/python ]; then echo $(ROOT)/.venvs/se/bin/python; else echo python; fi)
SETUP_PY := $(shell if [ -x $(ROOT)/.venvs/se/bin/python ]; then echo $(ROOT)/.venvs/se/bin/python; else uv python find 3.12; fi)

DF_RFFT_MANIFEST := src/se/models/deepfilternet3/rfft_kernel/Cargo.toml
DF_RFFT_TARGET ?= $(CURDIR)/build/df-rfft
DF_RFFT_SO := src/se/models/deepfilternet3/libdf_rfft.so

LINK_FLAGS :=
ifneq ($(strip $(DRY_RUN)),)
LINK_FLAGS += --dry-run
endif
ifneq ($(strip $(ONLY)),)
LINK_FLAGS += --only $(ONLY)
endif
ifneq ($(strip $(ALLOW_PARTIAL)),)
LINK_FLAGS += --allow-partial
endif

help:
	@printf '%s\n' \
	  'Setup (platform is required only here):' \
	  '  make setup PLATFORM=tsubame' \
	  '  make setup PLATFORM=shinoda_lab' \
	  '  make setup-env PLATFORM=tsubame ROLES=se,asr' \
	  '  make assets MODELS=mp_senet' \
	  '  make assets-check' \
	  '  make db-link ONLY=LibriSpeech DRY_RUN=1 ALLOW_PARTIAL=1' \
	  '  make doctor' \
	  '  make test-setup' \
	  '' \
	  'Roles: se, asr, official, semambapp, deepfilternet' \
	  'Platforms: tsubame, shinoda_lab' \
	  'Default setup builds every role, fetches every pinned asset, links configured datasets, then runs doctor.' \
	  'DB init, ingest, and bootstrap are separate and are not run by make setup.' \
	  '' \
	  'Existing targets:' \
	  '  make db-init | make db-ingest | make db-bootstrap | make df-rfft'

setup:
	$(WITH_ENV) $(ROOT)/scripts/setup/setup.sh $(PLATFORM)

setup-env:
	$(WITH_ENV) bash -c 'cd $(ROOT) && source scripts/setup/platforms/$(PLATFORM).sh && platform_prepare && source scripts/setup/env_roles.sh && install_roles "$(or $(ROLES),se,asr,official,semambapp,deepfilternet)"'

assets:
	$(WITH_ENV) $(SETUP_PY) $(ROOT)/scripts/setup/assets.py fetch --models "$(MODELS)"

assets-check:
	$(WITH_ENV) $(SETUP_PY) $(ROOT)/scripts/setup/assets.py check --models "$(MODELS)"

doctor:
	$(WITH_ENV) $(SETUP_PY) $(ROOT)/scripts/setup/doctor.py

test-setup:
	$(WITH_ENV) uv run --python 3.12 --no-project python -m unittest discover -s $(ROOT)/tests/setup -v

df-rfft:
	cargo build --release --locked \
		--manifest-path $(DF_RFFT_MANIFEST) \
		--target-dir $(DF_RFFT_TARGET) \
	|| cargo build --release \
		--manifest-path $(DF_RFFT_MANIFEST) \
		--target-dir $(DF_RFFT_TARGET)
	@if [ -f $(DF_RFFT_SO) ] && ! cmp -s $(DF_RFFT_TARGET)/release/libdf_rfft.so $(DF_RFFT_SO); then \
	  echo "built $(DF_RFFT_TARGET)/release/libdf_rfft.so differs from tracked $(DF_RFFT_SO); tracked file was not replaced"; \
	elif [ ! -f $(DF_RFFT_SO) ]; then \
	  cp -f $(DF_RFFT_TARGET)/release/libdf_rfft.so $(DF_RFFT_SO); \
	fi

db-link:
	$(WITH_ENV) $(PYTHON) -m src.data.link_corpora $(LINK_FLAGS)

db-ingest:
	$(WITH_ENV) $(PYTHON) -m src.data.ingest_corpora

db-init:
	$(WITH_ENV) $(PYTHON) -m src.data.init_eval

db-bootstrap:
	$(WITH_ENV) $(PYTHON) -m src.data.bootstrap
