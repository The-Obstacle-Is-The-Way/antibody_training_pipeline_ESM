# Repository Guidelines

## Project Structure & Module Organization
Core pipeline code lives in `src/antibody_training_esm` (subpackages `core`, `data`, `datasets`, `evaluation`, `utils`, `cli`). Configuration files stay in `configs/`, dataset prep notebooks and scripts in `preprocessing/`, and repeatable automations in `scripts/` or `experiments/`. Store tests in `tests/`, runtime artifacts in `models/`, `logs/`, `outputs/`, datasets under `train_datasets/` or `test_datasets/`, and keep long-form references plus figures in `docs/` and `assets/`.

## Build, Test, and Development Commands
- `make install`: sync dependencies with uv.
- `make all`: run format → lint → typecheck → pytest; use pre-push.
- `make test` / `make coverage`: targeted pytest runs, coverage enforces 70% and emits `htmlcov/`.
- `make train`: wraps `uv run antibody-train --config configs/config.yaml` for the validated pipeline.
- `docker compose run dev bash`: start the hot-reload dev image when local toolchains drift.
- `uv run pre-commit install`: install hooks so commits mirror CI.

## Coding Style & Naming Conventions
Ruff handles formatting and linting (4-space indent, 88-char lines, double quotes, sorted imports). Mypy is strict (`disallow_untyped_defs=true`), so annotate every public callable and avoid `Any`. Use `snake_case` modules/functions, `PascalCase` classes, `SCREAMING_SNAKE_CASE` constants, lowercase dashed config names, and keep tests in `test_*.py` with colocated fixtures.

## Testing Guidelines
Pytest discovers `Test*` classes and `test_*` functions; tag suites with `unit`, `integration`, `e2e`, or `slow` so reviewers can filter (`uv run pytest -m "unit and not slow"`). Keep deterministic fixtures under `tests/fixtures/` and reference large CSVs from the dataset folders instead of duplicating. Exceed the 70% gate, share coverage deltas for risky modules, and mock Hugging Face calls in unit tests while reserving real downloads for marked slow runs.

## Commit & Pull Request Guidelines
Commits follow a Conventional style (`fix:`, `docs:`, `feat:`) with imperative subjects ≤72 chars and bodies noting affected datasets/configs. PRs should summarize scope, link issues, list commands executed (`make all`, `make coverage`, `make train` when applicable), and mention new data paths or artifacts. Attach screenshots or tables for docs/UI tweaks and keep refactors separate from feature or data work.

## Security & Configuration Tips
Model checkpoints, embedding caches, and dataset intermediates rely on `pickle`; only load artifacts produced by this repo and store them in `models/` or `embeddings_cache/`. Prefer updating `configs/*.yaml` instead of hard-coded paths, pin Hugging Face revisions when training, and cite any mitigation in `docs/SECURITY_REMEDIATION_PLAN.md`. Keep raw datasets out of Git—drop them into the provided folders and document the preprocessing command used.
