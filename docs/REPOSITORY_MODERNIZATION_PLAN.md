# Repository Modernization Plan: 2025 Gold Standards

**Status:** 📋 Planning Phase
**Created:** 2025-11-06
**Target:** Upgrade to 2025 best practices for Python ML projects
**Branch:** leroy-jenkins/full-send

---

## Executive Summary

This document assesses the current repository configuration against November 2025 best practices for Python machine learning projects and provides a comprehensive implementation roadmap to achieve gold-standard developer experience.

**TL;DR:**
- Current: Basic setup with outdated tooling (black, isort, minimal pytest)
- Target: Modern toolchain with Ruff, uv, comprehensive type checking, pre-commit hooks
- Impact: 10-100x faster linting/formatting, reproducible environments, automated quality gates

---

## Table of Contents

1. [Current State Audit](#current-state-audit)
2. [2025 Gold Standard Tools](#2025-gold-standard-tools)
3. [Gap Analysis](#gap-analysis)
4. [Recommended Toolchain](#recommended-toolchain)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Configuration Examples](#configuration-examples)
7. [Migration Strategy](#migration-strategy)
8. [Success Metrics](#success-metrics)

---

## Current State Audit

### ✅ What's Working

**Dependencies & Build:**
- ✅ `pyproject.toml` exists (PEP 621 compliant)
- ✅ Hatchling build backend configured
- ✅ Python 3.12.7 (modern version)
- ✅ uv installed (`/opt/homebrew/bin/uv`)

**Testing:**
- ✅ `tests/` directory with 4 test files
- ✅ pytest in dev dependencies
- ✅ `.pytest_cache/` shows recent usage

**Linting/Formatting (Partial):**
- ✅ black, isort in dev dependencies
- ✅ Basic black config (line-length 88)
- ✅ Basic isort config (profile "black")
- ✅ `.mypy_cache/`, `.ruff_cache/` show previous usage

**Version Control:**
- ✅ Comprehensive `.gitignore`
- ✅ `uv.lock` explicitly ignored (line 47)

### ❌ What's Missing

**Package Management:**
- ❌ No `uv.lock` file (dependency reproducibility)
- ❌ No `.python-version` file (version pinning)
- ❌ Using old-style pip installation pattern

**Code Quality Automation:**
- ❌ No pre-commit hooks configuration
- ❌ No Makefile or task runner (just, taskfile)
- ❌ No ruff configuration (even though cache exists)
- ❌ No mypy configuration (even though cache exists)
- ❌ Using slow, outdated tools (black, isort vs ruff)

**Testing Infrastructure:**
- ❌ No pytest configuration in pyproject.toml
- ❌ No coverage tracking/reporting
- ❌ No test coverage requirements
- ❌ Tests are script-style, not proper pytest tests

**Type Checking:**
- ❌ No type hints enforcement
- ❌ No mypy/pyright in dev dependencies
- ❌ No type checking in workflow

**CI/CD:**
- ❌ No `.github/workflows/` directory
- ❌ No automated testing
- ❌ No linting/formatting checks
- ❌ No type checking in CI

**Documentation:**
- ❌ No docstring linting (pydocstyle, darglint)
- ❌ No API documentation generation
- ❌ No documentation build pipeline

**ML-Specific:**
- ❌ No experiment tracking configuration (MLflow, Weights & Biases)
- ❌ No data versioning (DVC)
- ❌ No model versioning strategy documented
- ❌ No reproducibility guarantees

---

## 2025 Gold Standard Tools

### The Modern Python Stack (November 2025)

Based on industry adoption by FastAPI, pandas, pydantic, Apache Airflow:

| Category | Tool | Why | Speed Improvement |
|----------|------|-----|-------------------|
| **Package Manager** | uv | Rust-based, pip/poetry replacement | 10-100x faster |
| **Linting** | Ruff | Replaces flake8 + plugins | 10-100x faster |
| **Formatting** | Ruff | Replaces black + isort | 10-100x faster |
| **Type Checking** | mypy or pyright | Industry standard | N/A |
| **Testing** | pytest + pytest-cov | De facto standard | N/A |
| **Security** | Bandit (via Ruff) | Vulnerability scanning | Built-in |
| **Pre-commit** | pre-commit | Automated quality gates | N/A |
| **Task Runner** | make or just | Command automation | N/A |
| **CI/CD** | GitHub Actions | Native integration | N/A |

### Key 2025 Trends

1. **Consolidation:** Ruff replaces 5-10 separate tools
2. **Speed:** Rust-based tools (uv, ruff) dominate
3. **Reproducibility:** Lock files + version pinning mandatory
4. **Automation:** Pre-commit hooks + CI/CD are table stakes
5. **Type Safety:** Type hints + strict checking increasingly required

---

## Gap Analysis

### Critical Gaps (P0 - Blocks Gold Standard)

1. **No uv.lock file** - Cannot guarantee reproducible environments
2. **No pre-commit hooks** - No automated quality gates before commits
3. **No CI/CD** - No automated testing/validation
4. **Using outdated tools** - black/isort instead of Ruff (100x slower)
5. **No type checking** - Missing entire quality dimension

### High Priority (P1 - Required for Best Practices)

6. **No Makefile/task runner** - Poor developer experience
7. **No coverage tracking** - Can't measure test quality
8. **No mypy config** - Type checking not enforceable
9. **No pytest config** - Test behavior not standardized
10. **Script-style tests** - Not using pytest properly

### Medium Priority (P2 - Nice to Have)

11. **No docstring linting** - Documentation quality not enforced
12. **No experiment tracking** - ML-specific best practices missing
13. **No data versioning** - Reproducibility incomplete for ML
14. **No model registry** - Model management ad-hoc

---

## Recommended Toolchain

### Core Developer Experience

```toml
[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "pytest-xdist>=3.6.0",  # Parallel testing

    # Linting & Formatting (Ruff replaces black, isort, flake8, etc.)
    "ruff>=0.8.0",

    # Type Checking
    "mypy>=1.13.0",
    "pandas-stubs>=2.2.0",  # Type stubs for pandas

    # Security
    "bandit[toml]>=1.7.0",

    # Pre-commit
    "pre-commit>=4.0.0",
]
```

### Ruff Configuration (replaces black + isort + flake8 + more)

```toml
[tool.ruff]
target-version = "py39"
line-length = 88

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "ARG",  # flake8-unused-arguments
    "SIM",  # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports OK in __init__
"tests/**/*" = ["ARG"]    # Unused arguments OK in tests

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### mypy Configuration

```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start lenient, tighten over time
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true
```

### pytest Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=.",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=70",  # Require 70% coverage
    "-v",
    "-ra",  # Show summary of all test outcomes
]
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["."]
omit = [
    "tests/*",
    "**/__pycache__/*",
    ".venv/*",
]

[tool.coverage.report]
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1) 🏗️

**Goal:** Get basic tooling in place

**Tasks:**
1. ✅ Create `REPOSITORY_MODERNIZATION_PLAN.md` (this doc)
2. ⬜ Initialize uv environment
   ```bash
   uv init
   uv sync
   ```
3. ⬜ Update `pyproject.toml` with modern dev dependencies
4. ⬜ Remove black/isort, add Ruff
5. ⬜ Add mypy with lenient config
6. ⬜ Create `.python-version` file (3.12.7)
7. ⬜ Create `uv.lock` and UN-IGNORE it in .gitignore

**Validation:**
- [ ] `uv sync` completes successfully
- [ ] All dependencies locked in `uv.lock`

---

### Phase 2: Code Quality Automation (Week 1) 🤖

**Goal:** Automate quality checks before commits

**Tasks:**
1. ⬜ Create `.pre-commit-config.yaml`
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.8.0
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format

     - repo: https://github.com/pre-commit/mirrors-mypy
       rev: v1.13.0
       hooks:
         - id: mypy
           additional_dependencies: [pandas-stubs]
   ```

2. ⬜ Install pre-commit hooks
   ```bash
   uv run pre-commit install
   ```

3. ⬜ Run on all files (initial cleanup)
   ```bash
   uv run pre-commit run --all-files
   ```

4. ⬜ Fix any issues found

**Validation:**
- [ ] Pre-commit runs on `git commit`
- [ ] Ruff catches linting errors
- [ ] Formatter runs automatically

---

### Phase 3: Testing Infrastructure (Week 1) 🧪

**Goal:** Proper pytest setup with coverage tracking

**Tasks:**
1. ⬜ Migrate script-style tests to proper pytest
   - Convert print statements to assertions
   - Use fixtures for common setup
   - Use parametrize for multiple test cases

2. ⬜ Add pytest configuration to `pyproject.toml`

3. ⬜ Add coverage configuration

4. ⬜ Run tests with coverage
   ```bash
   uv run pytest
   ```

5. ⬜ Generate coverage report
   ```bash
   uv run pytest --cov-report=html
   open htmlcov/index.html
   ```

**Validation:**
- [ ] All tests pass
- [ ] Coverage report generated
- [ ] Coverage ≥ 70% (or document baseline)

---

### Phase 4: CI/CD Pipeline (Week 2) 🚀

**Goal:** Automated testing on every push/PR

**Tasks:**
1. ⬜ Create `.github/workflows/ci.yml`
   ```yaml
   name: CI

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4

         - name: Install uv
           uses: astral-sh/setup-uv@v3

         - name: Set up Python
           run: uv python install 3.12.7

         - name: Install dependencies
           run: uv sync

         - name: Run Ruff
           run: uv run ruff check .

         - name: Run mypy
           run: uv run mypy .

         - name: Run tests
           run: uv run pytest
   ```

2. ⬜ Test workflow on branch
3. ⬜ Add status badge to README.md

**Validation:**
- [ ] CI runs on push
- [ ] All checks pass
- [ ] Badge shows green

---

### Phase 5: Developer Experience (Week 2) 🛠️

**Goal:** Simple commands for common tasks

**Tasks:**
1. ⬜ Create `Makefile`
   ```makefile
   .PHONY: install test lint format typecheck all clean

   install:
       uv sync

   test:
       uv run pytest

   lint:
       uv run ruff check .

   format:
       uv run ruff format .

   typecheck:
       uv run mypy .

   all: format lint typecheck test

   clean:
       rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
   ```

2. ⬜ Document commands in README.md

**Validation:**
- [ ] `make install` works
- [ ] `make all` runs full pipeline
- [ ] README documents all commands

---

### Phase 6: ML-Specific Tooling (Week 3) 🔬

**Goal:** Add ML best practices

**Tasks:**
1. ⬜ Evaluate experiment tracking
   - MLflow (self-hosted)
   - Weights & Biases (cloud)
   - TensorBoard (simple)

2. ⬜ Evaluate data versioning
   - DVC (Git-based)
   - Delta Lake
   - None (if datasets are small/stable)

3. ⬜ Document model versioning strategy
   - Models in Git LFS?
   - Model registry (MLflow, W&B)?
   - Artifact storage (S3, GCS)?

4. ⬜ Add reproducibility documentation
   - How to reproduce training
   - How to reproduce inference
   - Random seed management

**Validation:**
- [ ] Chosen tools documented
- [ ] Integration plan created
- [ ] No decision if not needed (document why)

---

### Phase 7: Documentation (Week 3) 📚

**Goal:** Quality documentation with enforcement

**Tasks:**
1. ⬜ Add docstring linting to Ruff
   ```toml
   select = ["D"]  # pydocstyle
   ```

2. ⬜ Choose documentation style (Google, NumPy, Sphinx)

3. ⬜ Add type hints to public APIs

4. ⬜ Consider API documentation generation
   - Sphinx
   - MkDocs
   - pdoc

**Validation:**
- [ ] Docstrings on all public functions
- [ ] Type hints on all public APIs
- [ ] Linting enforces standards

---

## Configuration Examples

### Complete pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "antibody-dev-esm"
version = "1.0.0"
description = "Antibody developability analysis using ESM protein language model"
requires-python = ">=3.9"
dependencies = [
    "biopython>=1.80",
    "datasets>=4.2.0",
    "jupyterlab>=4.4.9",
    "matplotlib>=3.7.0",
    "more-itertools",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "plotly",
    "pyparsing>=3.0.0",
    "PyYAML>=6.0.0",
    "riot_na",
    "scikit-learn>=1.3.0",
    "scipy>=1.10.0",
    "seaborn>=0.12.0",
    "torch>=2.6.0",
    "tqdm>=4.65.0",
    "transformers>=4.30.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "pytest-xdist>=3.6.0",

    # Linting & Formatting
    "ruff>=0.8.0",

    # Type Checking
    "mypy>=1.13.0",
    "pandas-stubs>=2.2.0",

    # Security
    "bandit[toml]>=1.7.0",

    # Pre-commit
    "pre-commit>=4.0.0",
]

[tool.ruff]
target-version = "py39"
line-length = 88

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "ARG",  # flake8-unused-arguments
    "SIM",  # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**/*" = ["ARG"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=.",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=70",
    "-v",
    "-ra",
]

[tool.coverage.run]
source = ["."]
omit = [
    "tests/*",
    "**/__pycache__/*",
    ".venv/*",
]

[tool.coverage.report]
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### .gitignore Updates

**REMOVE this line:**
```
uv.lock
```

**ADD these lines:**
```
# Coverage reports
htmlcov/
.coverage
coverage.xml

# Ruff cache
.ruff_cache/
```

---

## Migration Strategy

### Breaking Changes

**None!** All changes are additive:
- ✅ Can run old and new tools side-by-side
- ✅ uv works with existing pyproject.toml
- ✅ Ruff is compatible with black/isort configs
- ✅ Pre-commit is optional per-developer

### Gradual Adoption Path

**Option 1: Big Bang (Recommended)**
- Implement all phases in 1-2 weeks
- Get to gold standard quickly
- Cleaner migration

**Option 2: Incremental**
- Phase 1-2 first (uv + ruff)
- Use for 1-2 weeks
- Then Phase 3-4 (testing + CI)
- Then Phase 5-7 (DX + docs)

**Option 3: Minimal**
- Phase 1-2 only (uv + ruff)
- Skip CI/CD if not needed
- Skip ML tooling if not needed
- Document deviations from gold standard

---

## Success Metrics

### Code Quality Metrics

**Before:**
- ⏱️ Linting/formatting: ~20-40 seconds (black + isort + flake8)
- 📊 Test coverage: Unknown
- 🔒 Type coverage: 0%
- 🤖 Automated checks: None

**After (Target):**
- ⏱️ Linting/formatting: <1 second (Ruff)
- 📊 Test coverage: ≥70% (tracked)
- 🔒 Type coverage: 50%+ public APIs
- 🤖 Automated checks: Pre-commit + CI

### Developer Experience Metrics

**Before:**
- 📦 Install time: Variable (pip install)
- 🔄 Reproducibility: Poor (no lock file)
- 🛠️ Common tasks: Manual commands
- ✅ Quality gates: Manual

**After (Target):**
- 📦 Install time: <30 seconds (`uv sync`)
- 🔄 Reproducibility: 100% (`uv.lock`)
- 🛠️ Common tasks: `make <cmd>` or `uv run <cmd>`
- ✅ Quality gates: Automated (pre-commit + CI)

---

## Open Questions

### For Team Decision

1. **CI/CD Provider:** GitHub Actions (free) vs GitLab CI vs CircleCI?
   - Recommendation: GitHub Actions (native, free, powerful)

2. **Type Checking Tool:** mypy (mature) vs pyright (fast)?
   - Recommendation: mypy (better ecosystem, documentation)

3. **Coverage Target:** 70% vs 80% vs 90%?
   - Recommendation: Start at 70%, increase over time

4. **Experiment Tracking:** MLflow vs W&B vs None?
   - Recommendation: Document experiments in markdown for now, add MLflow if needed

5. **Data Versioning:** DVC vs Delta Lake vs None?
   - Recommendation: None for now (datasets are stable CSV files in git)

6. **Documentation Generator:** Sphinx vs MkDocs vs None?
   - Recommendation: None for now (focus on good docstrings first)

---

## References

### Official Documentation

- **uv:** https://docs.astral.sh/uv/
- **Ruff:** https://docs.astral.sh/ruff/
- **mypy:** https://mypy.readthedocs.io/
- **pytest:** https://docs.pytest.org/
- **pre-commit:** https://pre-commit.com/

### 2025 Best Practices

- [Modern Python Project Setup (2025)](https://albertsikkema.com/python/development/best-practices/2025/10/31/modern-python-project-setup.html)
- [Why Replace Flake8, Black, and isort with Ruff](https://medium.com/@zigtecx/why-you-should-replace-flake8-black-and-isort-with-ruff-the-ultimate-python-code-quality-tool-a9372d1ddc1e)
- [Managing Python Projects With uv (Real Python)](https://realpython.com/python-uv/)

### Major Projects Using These Tools

- FastAPI
- pandas
- pydantic
- Apache Airflow
- SQLAlchemy
- Django (evaluating Ruff)

---

## Next Steps

1. **Review this document** with team/senior developers
2. **Get approval** for recommended toolchain
3. **Choose migration strategy** (Big Bang vs Incremental vs Minimal)
4. **Execute Phase 1** (Foundation)
5. **Iterate through phases** based on priority

---

## Approval

- [ ] **Reviewed by:** _____________
- [ ] **Approved by:** _____________
- [ ] **Migration strategy chosen:** _____________
- [ ] **Start date:** _____________

---

**Last Updated:** 2025-11-06
**Next Review:** After Phase 1 completion
**Maintainer:** Ray + Claude (leroy-jenkins/full-send branch)
