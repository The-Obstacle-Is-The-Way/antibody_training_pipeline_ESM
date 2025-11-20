# Repository Guidelines

## Project Structure & Module Organization
- Core library in `src/antibody_training_esm/` with `core/` (training loop, models), `datasets/` (loading/preprocessing), `evaluation/`, and `cli/` entry points. Hydra configs live in `src/antibody_training_esm/conf/` (per model/data/classifier) and mirror CLI overrides.
- Data inputs and small artifacts are under `data/`; long-running experiment outputs land in `experiments/runs/` (Hydra multi-runs) and `dist/` (packaged builds). Keep checked-in fixtures in `tests/fixtures/`.
- Documentation: user/developer guides in `docs/`, research notes in `docs/research/`, roadmap/status docs in the repo root.

## Build, Test, and Development Commands
- Environment: `uv venv && source .venv/bin/activate && uv sync --all-extras` (one-time), or simply `make install`.
- Format/lint/typecheck/test pipeline: `make all` (runs ruff format → ruff lint → mypy → pytest fast suite).
- Targeted test commands:
  - `make test`: Fast suite (~95s, skips e2e/slow/gpu) - **use for development**
  - `make test-e2e`: E2E suite (honors opt-in env vars like RUN_NOVO_E2E=1)
  - `make test-all`: Full suite (env-gated tests may still skip without flags)
- Coverage: `make coverage` (runs unit+integration with `--cov-fail-under=70`, HTML in `htmlcov/`).
- Train locally: `make train` or `uv run antibody-train hardware.device=cuda training.batch_size=32` (Hydra overrides allowed). Clean caches: `make clean`.

## Coding Style & Naming Conventions
- Python 3.12, spaces only, line length 88. Ruff handles formatting; prefer double quotes per `tool.ruff.format`.
- Type hints required (`disallow_untyped_defs = true`); keep functions small and pure where practical.
- Module/import ordering enforced by ruff (`I` rules). Ignore unused imports only in `__init__.py`.
- Name tests `test_*.py`; classes `TestSomething`; fixtures live under `tests/fixtures`.

## Testing Guidelines
- Primary frameworks: pytest + pytest-cov. Run fast checks with `make test` (~95s); skip e2e/slow/gpu tests automatically.
- Test markers: `unit`, `integration`, `e2e`, `slow`, `gpu`. Register new markers in `pyproject.toml` before using.
- Coverage target: `make coverage` enforces 70% minimum on unit+integration (`src/antibody_training_esm/` focus) and writes HTML to `htmlcov/`.
- Heavy e2e tests (real ESM model downloads ~650MB) are opt-in via env vars:
  - `RUN_NOVO_E2E=1` for Novo accuracy reproduction test
  - `RUN_PREDICT_CLI_E2E=1` for predict CLI test with real weights
- Place integration/e2e tests under `tests/integration/` and `tests/e2e/`.

## Commit & Pull Request Guidelines
- Follow existing convention: `<type>: <summary>` (lowercase type like `docs`, `feat`, `fix`, `chore`). Keep summaries imperative and scoped.
- Before opening a PR: run `make all`; add/refresh tests and docs for behavior changes.
- PRs should include a brief description, linked issue/roadmap item, notable config overrides, and screenshots or logs when UI/CLI behavior changes.

## Security & Configuration Tips
- Pickle artifacts (`*.pkl`) are trusted/local only; do not load unvetted files. Prefer JSON/NPZ if exchanging externally.
- Hydra configs resolve into `experiments/runs/<name>/<timestamp>/.hydra/`; keep overrides explicit in commit messages or PR notes to aid reproducibility.
