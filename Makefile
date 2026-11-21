.PHONY: install data run test lint app all

install:
	pip install -e ".[dev]"

data:
	churn simulate --out data/raw --seed 42

run:
	churn run --data-dir data/raw --report-dir reports

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .
	mypy src

app:
	streamlit run app/streamlit_app.py

all: lint test data run
