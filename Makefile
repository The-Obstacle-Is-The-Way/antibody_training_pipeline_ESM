.PHONY: install test coverage lint format typecheck hooks all clean help train docs docs-serve docs-build docs-clean docker-dev docker-prod

help:
	@echo "Available commands:"
	@echo "  make install      - Install all dependencies with uv"
	@echo "  make format       - Format code with ruff"
	@echo "  make lint         - Run ruff linting checks"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make hooks        - Run pre-commit hooks on all files"
	@echo "  make test         - Run fast suite (unit + integration; skips e2e/slow/gpu)"
	@echo "  make test-e2e     - Run e2e suite (honors opt-in env vars like RUN_NOVO_E2E)"
	@echo "  make test-all     - Run full suite (e2e/slow included; env-gated tests may still skip)"
	@echo "  make coverage     - Run tests with coverage report (requires unit tests)"
	@echo "  make all          - Run format, lint, typecheck, and test"
	@echo "  make train        - Run training pipeline"
	@echo "  make docker-dev   - Start dev container (auto-detects GPU)"
	@echo "  make docker-prod  - Start prod container (auto-detects GPU)"
	@echo "  make docs-serve   - Serve documentation locally with live reload"
	@echo "  make docs-build   - Build documentation to site/ directory"
	@echo "  make docs-clean   - Remove generated documentation"
	@echo "  make clean        - Remove cache directories"

install:
	uv sync --all-extras

test:
	uv run pytest -m "not e2e and not slow and not gpu"

test-e2e:
	uv run pytest -m "e2e"

test-all:
	uv run pytest

coverage:
	uv run pytest -m "unit or integration" --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=70

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy .

hooks:
	uv run pre-commit run --all-files

all: format lint typecheck test

train:
	uv run antibody-train

# Docker Smart Launchers
docker-dev:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "🚀 NVIDIA GPU detected! Launching with GPU support..."; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev; \
	else \
		echo "💻 No NVIDIA GPU detected (or macOS). Launching in CPU mode..."; \
		docker compose run --rm dev; \
	fi


docker-prod:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "🚀 NVIDIA GPU detected! Launching with GPU support..."; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm prod; \
	else \
		echo "💻 No NVIDIA GPU detected (or macOS). Launching in CPU mode..."; \
		docker compose run --rm prod; \
	fi


docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

docs-clean:
	rm -rf site/

clean: docs-clean
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete