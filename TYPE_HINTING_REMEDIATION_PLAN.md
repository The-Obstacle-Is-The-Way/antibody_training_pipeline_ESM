# Type-Hint Remediation Strategy

_Last updated: 2025-11-08_

## 1. Background

- Commit `5003e6c` flipped `[tool.mypy].disallow_untyped_defs = true` for the entire repo.
- Unit/integration tests, fixtures, and preprocessing scripts still contain large blocks of legacy `def foo(...)` signatures without annotations.
- `make all` (which runs `uv run mypy .`) now fails with hundreds of `no-untyped-def` errors even though runtime behaviour is unchanged.

## 2. Problem Statement

Type enforcement landed before the codebase was ready. This blocks CI / `make all`, hides real typing issues behind noise, and risks regressions when contributors start adding `# type: ignore` everywhere just to get builds green.

## 3. Goals

1. Restore a passing `make all` without weakening the stricter mypy settings.
2. Eliminate “missing type annotation” errors systematically (no mass `Any`, no broad config relaxations).
3. Keep runtime code paths untouched; only add hints or light refactors that clarify intent.

## 4. Scope

Modules currently failing mypy (per latest run):

- `tests/` (unit + integration suites, fixtures, mocks)
- `preprocessing/` scripts for Boughter, Jain, Harvey, Shehata datasets
- `scripts/validation/` and `scripts/testing/`
- Selected `tests/fixtures` helpers

Production `src/` modules are already typed and should stay that way. The plan focuses on test/support code.

## 5. Strategy

### Phase A – Inventory & Ordering

1. Capture the canonical error listing once (e.g., `uv run mypy . > .mypy_failures.txt`).
2. Group failures by path cluster (fixtures, dataset tests, preprocessing, etc.).
3. Prioritise shared utilities first (fixtures, helper modules) because downstream tests inherit their annotations.

### Phase B – Fix Shared Utilities

1. `tests/fixtures/` and `tests/fixtures/mock_models.py` – add `Protocol`/`TypedDict` where useful, otherwise inject `-> None` and typed params.
2. `tests/unit/data/test_loaders.py` + `tests/unit/datasets/test_base.py` – add helper aliases (`Sequence[str]`, `NDArray[np.float32]`, etc.) to reduce duplication.
3. Re-run mypy after each sub-cluster; commit in logical chunks to keep diffs reviewable.

### Phase C – Annotate Dataset/Preprocessing Scripts

1. Introduce lightweight helper types (e.g., `Row = dict[str, str | float]`) to make annotations concise.
2. For CLI-style scripts returning nothing, explicitly declare `-> None`.
3. Where third-party libs return untyped objects (e.g., Pandas), prefer `pd.DataFrame`/`pd.Series` instead of `Any`.

### Phase D – Integration & Regression

1. Once mypy passes locally, run `make all` (ruff, mypy, pytest) to ensure no regressions.
2. Update documentation (`SECURITY_REMEDIATION_PLAN.md` / `CONTRIBUTING.md`) with the requirement to include type hints in new tests/scripts.

## 6. Guardrails

- **No blanket `# type: ignore`** unless there is a library stub gap; justify each with a comment.
- **No config weakening** (do not flip `disallow_untyped_defs` back off).
- **Minimal churn**: only add annotations or micro-refactors needed to express types.

## 7. Deliverables

1. Clean `mypy` run (zero errors).
2. `.mypy_failures.txt` removed (used only as temporary scratch).
3. PR description summarising clusters fixed and future watch areas (if any remain).

## 8. Next Steps

1. Generate the baseline failure file.
2. Start with `tests/fixtures/` and `tests/unit/datasets/test_base*.py`.
3. Iterate through Phases B–D, committing per cluster.

Let’s execute this plan carefully so we keep strict typing **and** a green CI. Homie, we’ve got this. 💪
