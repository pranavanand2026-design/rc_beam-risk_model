.PHONY: setup run dashboard test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e .

run:
	python scripts/run_pipeline.py

dashboard:
	streamlit run src/rcbeam_fire/dashboard/app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/* models/checkpoints/*.joblib outputs/figs/* outputs/tables/*
