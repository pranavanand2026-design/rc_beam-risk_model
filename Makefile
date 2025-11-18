PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup run dashboard test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e .

run:
	$(BIN)/python scripts/run_pipeline.py

dashboard:
	$(BIN)/streamlit run src/rcbeam_fire/dashboard/app.py

test:
	$(BIN)/pytest tests/ -v

clean:
	rm -rf data/processed/* models/checkpoints/*.joblib outputs/figs/* outputs/tables/*
