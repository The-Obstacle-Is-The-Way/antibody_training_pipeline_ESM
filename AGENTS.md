# Repository Guidelines

## Project Structure & Module Organization
Core pipeline code resides in `src/antibody_training_esm` (`core/`, `data/`, `datasets/`, `evaluation/`, `utils/`, `cli/`). Hydra configs live in `conf/` (inside package), preprocessing assets in `preprocessing/`, and automation or research artifacts in `scripts/` and `experiments/`. Tests sit under `tests/{unit,integration,e2e}` with shared fixtures in `tests/fixtures`. Checkpoints and logs belong in `models/`, `logs/`, `outputs/`, curated datasets in `train_datasets/` or `data/test/`, and references in `docs/` and `assets/`.

## Build, Test, and Development Commands
- `make install`: sync deps with uv.
- `make format` / `make lint` / `make typecheck`: Ruff format, Ruff lint, mypy.
- `make test` / `make coverage`: pytest (strict markers) with ≥70% coverage + `htmlcov/`.
- `make hooks`: run the CI pre-commit stack; `make all` chains format → lint → typecheck → test pre-push.
- `make train`: wraps `uv run antibody-train` (uses Hydra config from `conf/config.yaml` by default).

## Coding Style & Naming Conventions
Ruff enforces 4-space indentation, 88-character lines, double quotes, and sorted imports. Mypy (`disallow_untyped_defs=true`) requires every public callable be annotated with precise types. Stick with `snake_case` for modules, functions, configs, and dataset files (timestamps optional), `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants, and `test_*.py` for suites with colocated fixtures.

## Testing Guidelines
Pytest discovers `Test*` classes and `test_*` functions and runs with `--strict-markers`; register new markers in `pyproject.toml` first. Tag suites with `unit`, `integration`, `e2e`, or `slow` for filtering (`uv run pytest -m "unit and not slow"`). Hugging Face downloads remain mocked via `tests/fixtures/mock_models.py`—even e2e—unless a test explicitly opts into real weights. Keep fixtures deterministic, source large CSVs from the dataset folders, and meet the 70% gate.

## Commit & Pull Request Guidelines
Commits follow a Conventional flavor (`fix:`, `docs:`, `feat:`) with imperative subjects ≤72 characters and bodies describing affected datasets/configs. Each PR should summarize scope, link issues, list the commands you ran (`make all`, `make hooks`, `make coverage`, `make train` when relevant), and call out new artifacts or data paths. Separate refactors from feature or data work.

## Security & Configuration Tips
Models, embedding caches, and dataset intermediates rely on `pickle`; only load artifacts produced here and store them in `models/` or `embeddings_cache/`. Update YAML configs instead of hard-coding paths, pin Hugging Face revisions for reproducible training, and capture mitigations in `docs/SECURITY_REMEDIATION_PLAN.md`. Run `make hooks` (or `uv run bandit -c pyproject.toml`) after security-sensitive edits, and keep raw datasets out of Git—drop them into the dataset folders and record the preprocessing command used.
