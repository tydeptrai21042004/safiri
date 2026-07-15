PYTHON ?= python
export PYTHONPATH := src
export MPLCONFIGDIR ?= /tmp/matplotlib

.PHONY: setup generate prepare train evaluate report all test api dashboard demo clean-generated

setup:
	$(PYTHON) -m pip install -e ".[dev]"

generate:
	$(PYTHON) -m safiri_eta.cli generate

prepare:
	$(PYTHON) -m safiri_eta.cli prepare

train:
	$(PYTHON) -m safiri_eta.cli train

evaluate:
	$(PYTHON) -m safiri_eta.cli evaluate

report:
	$(PYTHON) -m safiri_eta.cli report

all:
	$(PYTHON) -m safiri_eta.cli all

test:
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

demo:
	@echo "Terminal 1: make api"
	@echo "Terminal 2: make dashboard"

clean-generated:
	$(PYTHON) scripts/clean_generated.py

