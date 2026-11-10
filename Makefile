.PHONY: help setup setup-env assets assets-check doctor test-setup \
	db-ingest db-init db-link db-bootstrap df-rfft \
	reproduction-doctor reproduction-init-db reproduction-snapshot \
	reproduction-rebuild-db reproduction-verify reproduction-prepare \
	reproduction-plan-build reproduction-register \
	reproduction-plan-dns reproduction-register-dns \
	reproduction-benchmark-dns-accept reproduction-plan-dns-accept reproduction-register-dns-accept \
	reproduction-register-p8 \
	experiment experiment-preflight test-cpu

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
	  '  make db-init | make db-ingest | make db-bootstrap | make df-rfft' \
	  '' \
	  'Reproduction database:' \
	  '  make reproduction-doctor [MODELS=mp_senet]' \
	  '  make reproduction-init-db' \
	  '  make reproduction-verify DATASET=name@version [HASH=1]' \
	  '  make reproduction-snapshot DATASET=name@version [ALLOW_UNKNOWN_LEAK_CHECK=1]' \
	  '  make reproduction-rebuild-db OUT=path/to/new.duckdb' \
	  '  make reproduction-prepare [STEP=all] [WORKERS=8]' \
	  '  make reproduction-plan-dns-accept | reproduction-register-dns-accept | reproduction-benchmark-dns-accept' \
	  '  make reproduction-register-p8' \
	  '' \
	  'Experiments:' \
	  '  make experiment CONFIG=configs/reproduction/experiments/....yaml' \
	  '  make experiment-preflight CONFIG=configs/reproduction/experiments/....yaml' \
	  '  make test-cpu'

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

REPRO_DB ?= $(ROOT)/data/db/reproduction.duckdb
REPRO_STAGING ?= $(ROOT)/data/staging/reproduction
REPRO_SNAPSHOTS ?= $(ROOT)/data/db/snapshots
REPRO_RESOURCE ?= $(ROOT)/data/db/resource.duckdb
DATASET ?=
OUT ?=
BUILD ?=
MS_SNSD_CONFIG ?= configs/data/builds/ms_snsd_tfgridnet_ondemand.yaml
DNS_CONFIG ?= configs/data/builds/dns_tflocoformer_ondemand.yaml
HASH ?=
ALLOW_UNKNOWN_LEAK_CHECK ?=
STEP ?= all
WORKERS ?= 8

reproduction-doctor:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli doctor --database "$(REPRO_DB)" --resource-db "$(REPRO_RESOURCE)" --models "$(MODELS)"

reproduction-init-db:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli init-db --database "$(REPRO_DB)"

reproduction-snapshot:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli snapshot --database "$(REPRO_DB)" --dataset "$(DATASET)" --snapshots "$(REPRO_SNAPSHOTS)" $(if $(ALLOW_UNKNOWN_LEAK_CHECK),--allow-unknown-leak-check,)

reproduction-rebuild-db:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli rebuild-db --database "$(REPRO_DB)" --staging "$(REPRO_STAGING)" --out "$(OUT)" --reference "$(REPRO_DB)"

reproduction-verify:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli verify --database "$(REPRO_DB)" --staging "$(REPRO_STAGING)" --dataset "$(DATASET)" --build "$(BUILD)" $(if $(HASH),--hash,)

reproduction-prepare:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli prepare --database "$(REPRO_DB)" --staging "$(REPRO_STAGING)" --snapshots "$(REPRO_SNAPSHOTS)" --resource-db "$(REPRO_RESOURCE)" --step "$(STEP)" --workers "$(WORKERS)"

reproduction-plan-build:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli ms-snsd --config "$(MS_SNSD_CONFIG)" --step plan

reproduction-register:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli ms-snsd --config "$(MS_SNSD_CONFIG)" --step register

reproduction-plan-dns:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli dns --config "$(DNS_CONFIG)" --step plan

reproduction-register-dns:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli dns --config "$(DNS_CONFIG)" --step register

DNS_ACCEPT_CONFIG ?= configs/data/builds/dns_tflocoformer_accept_recipe.yaml

reproduction-benchmark-dns-accept:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli dns --config "$(DNS_ACCEPT_CONFIG)" --step benchmark

reproduction-plan-dns-accept:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli dns --config "$(DNS_ACCEPT_CONFIG)" --step plan

reproduction-register-dns-accept:
	$(WITH_ENV) $(PYTHON) -m src.experiments.data.cli dns --config "$(DNS_ACCEPT_CONFIG)" --step register

reproduction-register-p8:
	$(WITH_ENV) $(PYTHON) -c 'from src.experiments.data.prepare.dfn3_online import register_p8; print(register_p8())'

CONFIG ?=
RESUME ?=
INITIALIZE_FROM ?=

experiment:
	@test -n "$(CONFIG)" || { echo "CONFIG is required"; exit 1; }
	CUDA_VISIBLE_DEVICES="$${CUDA_VISIBLE_DEVICES}" $(WITH_ENV) $(PYTHON) -m src.experiments.cli --config "$(CONFIG)" $(if $(RESUME),--resume "$(RESUME)",) $(if $(INITIALIZE_FROM),--initialize-from "$(INITIALIZE_FROM)",)

experiment-preflight:
	@test -n "$(CONFIG)" || { echo "CONFIG is required"; exit 1; }
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m src.experiments.cli --config "$(CONFIG)" --preflight

test-cpu:
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/asr -v
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/data -v
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/evaluation -v
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/experiments -v
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/se -v
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/setup -v
	CUDA_VISIBLE_DEVICES="" $(WITH_ENV) $(PYTHON) -m unittest discover -s tests/numerics -v
