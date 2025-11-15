# Development Workflow

**Target Audience:** Contributors to the codebase

**Purpose:** Learn the day-to-day development workflow, commands, git practices, and common tasks

---

## When to Use This Guide

Use this guide if you're:
- ✅ **Setting up your development environment** (first-time setup)
- ✅ **Running tests locally** (before commits)
- ✅ **Formatting/linting code** (code quality checks)
- ✅ **Making commits and pull requests** (git workflow)
- ✅ **Understanding quality gates** (CI requirements)
- ✅ **Performing common tasks** (adding datasets, training models, debugging)

---

## Quick Reference

**Daily Commands:**
```bash
make format      # Auto-format code with ruff
make lint        # Check linting with ruff
make typecheck   # Type check with mypy (strict mode)
make test        # Run tests with pytest
make all         # Run format → lint → typecheck → test (full quality gate)
```

**Before Commit:**
```bash
make hooks       # Run pre-commit checks manually
```

**Training/Testing:**
```bash
make train       # Train with default config
uv run antibody-test --model models/model.pkl --data data/test/jain/fragments/VH_only_jain.csv  # Test model
```

---

## Related Documentation

- **Architecture:** [Architecture Guide](architecture.md) - System design and components
- **Testing:** [Testing Strategy](testing-strategy.md) - Test architecture and patterns
- **Type Checking:** [Type Checking Guide](type-checking.md) - Type safety requirements
- **Preprocessing:** [Preprocessing Internals](preprocessing-internals.md) - Dataset preprocessing patterns

---

## Environment Setup

### Initial Setup

Install all dependencies including dev tools:

```bash
uv sync --all-extras
```

This installs:
- Core dependencies (torch, transformers, scikit-learn, pandas)
- Dev tools (pytest, ruff, mypy, pre-commit, bandit)
- CLI tools (click, pyyaml)

### Pre-commit Hooks

Install pre-commit hooks to catch issues before committing:

```bash
uv run pre-commit install
```

**What runs on commit:**
- `ruff format` - Auto-format code
- `ruff lint` - Lint checks
- `mypy` - Type checking (strict mode)

**Manual run:**
```bash
make hooks
```

**Behavior:** Failures block commits (intended - fix issues before committing)

---

## Development Commands

### Testing

**Run all tests:**
```bash
uv run pytest
```

**Run specific test types:**
```bash
uv run pytest -m unit           # Unit tests only (fast, < 1s each)
uv run pytest -m integration    # Integration tests (multi-component)
uv run pytest -m e2e           # End-to-end tests (expensive, full pipeline)
```

**Run specific test file:**
```bash
uv run pytest tests/unit/core/test_trainer.py
```

**Run specific test by name:**
```bash
uv run pytest -k test_function_name
```

**Coverage report:**
```bash
uv run pytest --cov=. --cov-report=html --cov-report=term-missing --cov-fail-under=70
# HTML report: htmlcov/index.html
# Terminal: Shows missing lines
# Enforced: ≥70% coverage required
```

**Important:** All tests must be tagged with `unit`, `integration`, `e2e`, or `slow` markers. Register new markers in `pyproject.toml` before using.

---

### Code Quality

**Format code:**
```bash
make format      # Auto-format with ruff (modifies files in-place)
```

**Lint code:**
```bash
make lint        # Check linting with ruff (no modifications)
```

**Type check:**
```bash
make typecheck   # Type check with mypy (strict mode)
```

**Run all quality checks:**
```bash
make all         # Format → Lint → Typecheck → Test
```

**Critical:** This repo maintains **100% type safety**. All functions must have complete type annotations. Mypy runs with `disallow_untyped_defs=true`.

---

### Training & Testing

**Train with default config:**
```bash
make train
# Uses: conf/config.yaml (Boughter train, Jain test)
```

**Override parameters from CLI:**
```bash
uv run antibody-train experiment.name=my_experiment hardware.device=cuda
```

**Test trained model:**
```bash
uv run antibody-test --model models/model.pkl --data data/test/jain/fragments/VH_only_jain.csv
```

**All CLI options:**
```bash
uv run antibody-train --help
uv run antibody-train --cfg job  # Show resolved config
uv run antibody-test --help
```

---

### Preprocessing

**Boughter (training set):**
```bash
python3 preprocessing/boughter/stage1_dna_translation.py
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py
```

**Jain (test set - Novo parity benchmark):**
```bash
python3 preprocessing/jain/step1_convert_excel_to_csv.py
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Harvey (nanobody test set):**
```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
python3 preprocessing/harvey/step2_extract_fragments.py
```

**Shehata (PSR assay test set):**
```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
python3 preprocessing/shehata/step2_extract_fragments.py
```

---

## Common Tasks

### Adding a New Dataset

1. **Create preprocessing directory:**
   ```bash
   mkdir -p preprocessing/{dataset}/
   ```

2. **Implement preprocessing pipeline:**
   - Convert Excel/CSV to canonical format
   - Follow patterns in existing preprocessing scripts
   - See [Preprocessing Internals](preprocessing-internals.md)

3. **Create dataset loader:**
   ```bash
   # Create src/antibody_training_esm/datasets/{dataset}.py
   # Extend AntibodyDataset base class
   ```

4. **Add dataset documentation:**
   ```bash
   mkdir -p docs/datasets/{dataset}/
   # Document dataset source, preprocessing, quirks
   ```

5. **Update preprocessing README:**
   ```bash
   # Add dataset to preprocessing/README.md
   ```

---

### Training a New Model

1. **Override parameters from CLI (no need to create new config files):**
   ```bash
   uv run antibody-train \
     experiment.name=my_experiment \
     training.model_name=my_model \
     data.train_file="train_datasets/{dataset}/canonical/VH_only.csv" \
     data.test_file="data/test/{dataset}/canonical/VH_only.csv" \
     classifier.C=1.0 \
     classifier.penalty=l2
   ```

2. **Model saved to:**
   ```
   outputs/{experiment.name}/{timestamp}/{model_name}.pkl
   outputs/{experiment.name}/{timestamp}/training.log
   outputs/{experiment.name}/{timestamp}/.hydra/config.yaml
   ```

---

### Running Hyperparameter Sweeps

1. **See reference implementation:**
   ```bash
   cat preprocessing/boughter/train_hyperparameter_sweep.py
   ```

2. **Create sweep config:**
   - Define parameter grid (e.g., C=[0.01, 0.1, 1.0, 10.0])
   - Train model for each configuration
   - Embeddings auto-cached for fast re-runs

3. **Compare results:**
   - Log cross-validation metrics
   - Select best hyperparameters

---

### Debugging Test Failures

**Run specific test with verbose output:**
```bash
uv run pytest tests/unit/core/test_trainer.py -v
```

**Show print statements:**
```bash
uv run pytest -s
```

**Drop into debugger on failure:**
```bash
uv run pytest --pdb
```

**Check test fixtures:**
```bash
ls tests/fixtures/mock_datasets/
# Deterministic test data lives here
```

---

## Git Workflow

### Main Branches

- **`leroy-jenkins/full-send`**: Main development branch (production-ready code)
- **`dev`**: Development branch (active work)

**Branch Strategy:** Feature branches merge to `dev` → `dev` merges to `leroy-jenkins/full-send`

---

### Commit Conventions

**Use Conventional Commits:**
- `fix:` - Bug fixes
- `feat:` - New features
- `docs:` - Documentation changes
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring (no behavior change)
- `chore:` - Maintenance tasks

**Format:**
- Imperative mood ("Add feature" not "Added feature")
- ≤72 characters for subject line
- Body explains what and why (not how)

**Example:**
```
fix: Correct PSR threshold to 0.5495 for Novo parity

The PSR assay threshold was incorrectly set to 0.5, causing a 6.3pp
accuracy gap on Shehata dataset. Setting to 0.5495 achieves exact
parity with Novo Nordisk benchmarks (58.8% accuracy).

Fixes: #123
```

---

### Pull Requests

**Requirements:**
- ✅ All CI checks pass (quality + tests)
- ✅ Coverage ≥70% maintained
- ✅ All tests have markers (`unit`, `integration`, `e2e`)
- ✅ Type annotations complete (mypy strict mode)

**PR Description Must Include:**
1. **Scope summary:** What changed and why
2. **Issue links:** `Fixes #123`, `Relates to #456`
3. **Commands run:** `make all`, `make coverage`
4. **New artifacts:** Call out new data paths, models, configs
5. **Testing:** How changes were validated

**Best Practices:**
- Keep refactors separate from feature/data work
- One logical change per PR
- Include before/after examples for user-facing changes
- Document breaking changes clearly

---

## Quality Gates

### Pre-commit (Local)

Runs automatically on `git commit`:
- `ruff format` - Code formatting
- `ruff lint` - Linting
- `mypy` - Type checking

**If failures occur:** Fix issues before committing (commits blocked)

---

### CI Pipeline (Remote)

Runs on all PRs and commits to main branches:

**Quality Checks:**
- `ruff` (format + lint)
- `mypy` (type checking, strict mode)
- `bandit` (security scanning)

**Testing:**
- Unit tests (fast, < 1s each)
- Integration tests (multi-component)
- Coverage enforcement (≥70%)

**E2E Tests:**
- Scheduled runs only (expensive)
- Full pipeline validation

**Merge Requirements:**
- All quality checks pass
- All tests pass
- Coverage ≥70%
- Bandit shows 0 findings

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
