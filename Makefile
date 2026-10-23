.PHONY: install train api test lint docker

install:
	pip install -r requirements-dev.txt

train:
	PYTHONPATH=src python -m churn.train

api:
	PYTHONPATH=src uvicorn churn.api:app --reload

test:
	pytest -q

lint:
	ruff check .

docker:
	docker build -t previsao-churn . && docker run -p 8000:8000 previsao-churn
