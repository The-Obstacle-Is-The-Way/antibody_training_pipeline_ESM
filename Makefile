.PHONY: install test coverage lint format typecheck hooks all clean help train

help:
	@echo "Available commands:"
	@echo "  make install    - Install all dependencies with uv"
	@echo "  make format     - Format code with ruff"
	@echo "  make lint       - Run ruff linting checks"
	@echo "  make typecheck  - Run mypy type checking"
	@echo "  make hooks      - Run pre-commit hooks on all files"
	@echo "  make test       - Run pytest test suite"
	@echo "  make coverage   - Run tests with coverage report (requires unit tests)"
	@echo "  make all        - Run format, lint, typecheck, and test"
	@echo "  make train      - Run training pipeline"
	@echo "  make clean      - Remove cache directories"

install:
	uv sync --all-extras

test:
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

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
