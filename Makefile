.PHONY: install dev test lint format build generate-data train serve api flask gradio docker-build docker-run clean

# ── Setup ────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

# ── Build (run locally before deploy) ────────────────────
build: generate-data train
	@echo "Build complete. data/sample/ and models/ are ready to deploy."

generate-data:
	python data/sample/generate_data.py

train:
	python -c "from src.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline('data', 'models').run_all()"

# ── Quality ──────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ app/ --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ app/ tests/ --line-length=120
	isort src/ app/ tests/

# ── Run ──────────────────────────────────────────────────
serve:
	streamlit run app/streamlit_app.py --server.port=8501

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

flask:
	python app/flask_app.py

gradio:
	python app/gradio_app.py

# ── Docker ───────────────────────────────────────────────
docker-build:
	docker build -t datapulse .

docker-run:
	docker run -p 8501:8501 -p 8000:8000 datapulse

# ── Cleanup (never deletes models/ or data/sample/) ─────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
