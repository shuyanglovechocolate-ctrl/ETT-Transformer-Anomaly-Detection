# Convenience targets for reviewing and reproducing this project.
# Override the interpreter if needed, e.g. `make test PYTHON=python3.11`.
PYTHON ?= python3

.PHONY: help install test manifest reproduce-core

help:
	@echo "Targets:"
	@echo "  install         Install Python dependencies (requirements.txt)"
	@echo "  test            Run the full pytest suite"
	@echo "  manifest        Write results/metrics/reproducibility_manifest.json"
	@echo "  reproduce-core  Run the full pipeline end-to-end (heavy; trains models)"
	@echo ""
	@echo "Override the interpreter with PYTHON=..., e.g. make test PYTHON=python3.11"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

manifest:
	$(PYTHON) experiments/write_reproducibility_manifest.py

reproduce-core:
	./scripts/reproduce_core.sh
