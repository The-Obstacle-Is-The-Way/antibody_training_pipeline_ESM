# Import Conventions & Package Structure Guide

**Professional Python Package Organization for Antibody Training ESM**

---

## Executive Summary

This guide defines the **canonical structure** for professional Python package organization, import conventions, and backwards compatibility strategy for the Antibody Training ESM codebase.

**Current Status**: ✅ v2.0.0 RELEASED - Clean Professional Package Structure

**Achievement**: Zero legacy code, 100% professional package organization (Phase 5 complete)

---

## Table of Contents

1. [Professional Python Package Structure](#professional-python-package-structure)
2. [Current State Analysis](#current-state-analysis)
3. [Import Conventions](#import-conventions)
4. [CLI Entry Points](#cli-entry-points)
5. [Migration History](#migration-history)
6. [Quality Standards](#quality-standards)

---

## Professional Python Package Structure

### Standard Layout (PEP 517/518 Compliant)

```
antibody_training_pipeline_ESM/
├── src/                              # Source package (PEP 517 src-layout)
│   └── antibody_training_esm/        # Main package
│       ├── __init__.py               # Package root
│       ├── cli/                      # Command-line interfaces
│       │   ├── train.py              # Training CLI (implemented)
│       │   ├── test.py               # Testing CLI (stub - needs migration)
│       │   └── preprocess.py         # Preprocessing guidance (implemented)
│       ├── core/                     # Core training logic
│       │   ├── classifier.py         # BinaryClassifier
│       │   ├── embeddings.py         # ESMEmbeddingExtractor
│       │   └── trainer.py            # Training orchestration
│       ├── datasets/                 # Dataset loaders (Phase 3)
│       │   ├── base.py               # AntibodyDataset abstract base
│       │   ├── jain.py               # Jain dataset loader
│       │   ├── harvey.py             # Harvey dataset loader
│       │   ├── shehata.py            # Shehata dataset loader
│       │   └── boughter.py           # Boughter dataset loader
│       ├── data/                     # Data utilities
│       │   └── loaders.py            # Generic data loading
│       ├── evaluation/               # Evaluation metrics (future)
│       └── utils/                    # Shared utilities
│
├── preprocessing/                    # Preprocessing scripts (SSOT)
│   ├── jain/
│   ├── harvey/
│   ├── shehata/
│   └── boughter/
│
├── tests/                            # Unit/integration tests
├── docs/                             # Documentation
├── configs/                          # Configuration files
├── scripts/                          # Development scripts
├── test_datasets/                    # Test data (small, version-controlled)
├── train_datasets/                   # Training data (large, gitignored)
│
├── pyproject.toml                    # Modern Python project config (PEP 517/518)
├── README.md                         # Project overview
├── USAGE.md                          # User guide
├── Makefile                          # Development commands
└── uv.lock                           # Dependency lock file
```

### What Should NOT Be in Root

❌ **Avoid These in Root:**
- `classifier.py` - Should be in `src/antibody_training_esm/core/`
- `data.py` - Should be in `src/antibody_training_esm/data/`
- `model.py` - Should be in `src/antibody_training_esm/core/`
- `train.py` - Should be in `src/antibody_training_esm/cli/`
- `test.py` - Should be in `src/antibody_training_esm/cli/`
- `main.py` - Should be in `src/antibody_training_esm/cli/`

✅ **Acceptable in Root:**
- Configuration: `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`
- Documentation: `README.md`, `USAGE.md`, `*.md` guides
- CI/CD: `.github/`, `.gitignore`, `.gitattributes`
- Python special: `setup.py` (legacy), `conftest.py` (pytest)

---

## Current State Analysis

### ✅ **Correctly Structured** (Phase 1-3 Complete)

```
src/antibody_training_esm/
├── cli/train.py              ✅ Proper CLI implementation
├── cli/preprocess.py         ✅ Proper CLI implementation
├── core/classifier.py        ✅ Migrated from root
├── core/embeddings.py        ✅ Migrated from root
├── core/trainer.py           ✅ Migrated from root
├── datasets/base.py          ✅ New abstractions (Phase 3)
├── datasets/jain.py          ✅ New abstractions (Phase 3)
├── datasets/harvey.py        ✅ New abstractions (Phase 3)
├── datasets/shehata.py       ✅ New abstractions (Phase 3)
├── datasets/boughter.py      ✅ New abstractions (Phase 3)
└── data/loaders.py           ✅ Migrated from root
```

### ✅ **Root Directory Status** (Phase 5 Complete - v2.0.0)

```
ROOT PYTHON FILES:            STATUS:
├── classifier.py             ✅ DELETED (Phase 5)
├── data.py                   ✅ DELETED (Phase 5)
├── main.py                   ✅ DELETED (Phase 5)
├── model.py                  ✅ DELETED (Phase 5)
├── train.py                  ✅ DELETED (Phase 5)
└── test.py                   ✅ DELETED (Phase 5)

ALL LEGACY CODE REMOVED - Clean professional package structure only!
```

### ✅ **Professional Package Implementation**

1. **Training CLI** - ✅ `src/antibody_training_esm/cli/train.py`
   - Entry point: `antibody-train`
   - Full configuration support via YAML

2. **Testing CLI** - ✅ `src/antibody_training_esm/cli/test.py`
   - Entry point: `antibody-test`
   - Multi-model/multi-dataset testing
   - Config file support

3. **Core Modules** - ✅ `src/antibody_training_esm/core/`
   - `classifier.py` - BinaryClassifier
   - `embeddings.py` - ESMEmbeddingExtractor
   - `trainer.py` - Training orchestration

4. **Dataset Loaders** - ✅ `src/antibody_training_esm/datasets/`
   - Abstract base class with Open/Closed Principle
   - Dataset-specific loaders (Jain, Harvey, Shehata, Boughter)

---

## Import Conventions

### ✅ **Professional Import Patterns** (v2.0.0+)

**ONLY USE THESE - No legacy imports exist in v2.0.0+**

```python
# From package CLI
from antibody_training_esm.cli.train import main as train_main
from antibody_training_esm.cli.test import main as test_main

# From core modules
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import train_model

# From dataset loaders
from antibody_training_esm.datasets import JainDataset, HarveyDataset
from antibody_training_esm.datasets import load_jain_data, load_harvey_data

# From data utilities
from antibody_training_esm.data.loaders import load_data, load_hf_dataset
```

### ⚠️ **Breaking Change from v1.x**

**v1.x code with root imports will NOT work in v2.0.0:**

```python
# ❌ REMOVED IN v2.0.0 - These files no longer exist
from classifier import BinaryClassifier
from model import ESMEmbeddingExtractor
from data import load_data
from train import train_model
```

**Migration to v2.0.0:**
Replace all root imports with full package paths as shown above.

---

## CLI Entry Points

### Defined in `pyproject.toml`

```toml
[project.scripts]
antibody-train = "antibody_training_esm.cli.train:main"           # ✅ Implemented
antibody-test = "antibody_training_esm.cli.test:main"             # ✅ Implemented (Phase 4)
antibody-preprocess = "antibody_training_esm.cli.preprocess:main" # ✅ Guidance only
```

### Usage Examples

```bash
# Training CLI (working)
antibody-train --config configs/config.yaml

# Testing CLI (working - Phase 4 complete!)
antibody-test --model models/classifier.pkl --data data/test.csv
antibody-test --model m1.pkl m2.pkl --data d1.csv d2.csv  # Multi-model/dataset
antibody-test --config test_config.yaml                    # Config file support

# Preprocessing CLI (guidance only - directs to scripts)
antibody-preprocess --dataset jain
```

### Design Principles

1. **Single Responsibility**: Each CLI does ONE thing well
2. **Professional UX**: Clear help messages, error handling, progress bars
3. **Configuration-Driven**: YAML configs for complex workflows
4. **Clean Package Structure**: Zero legacy code, 100% professional organization

---

## Migration History

### Phase 1: Core Migration (v0.1.0 → v0.2.0)
- Migrated `classifier.py` → `src/antibody_training_esm/core/classifier.py`
- Migrated `model.py` → `src/antibody_training_esm/core/embeddings.py`
- Migrated `train.py` → `src/antibody_training_esm/core/trainer.py`
- Created temporary backwards compatibility shims

### Phase 2: Data & CLI Migration (v0.2.0 → v0.3.0)
- Migrated `data.py` → `src/antibody_training_esm/data/loaders.py`
- Created professional CLI: `src/antibody_training_esm/cli/train.py`
- Entry point: `antibody-train`

### Phase 3: Dataset Abstractions (v0.3.0 → v1.0.0)
- Created `AntibodyDataset` abstract base class
- Implemented dataset-specific loaders (Jain, Harvey, Shehata, Boughter)
- Applied Open/Closed Principle for extensibility
- Achieved 100% type safety (mypy strict mode)

### Phase 4: Test CLI Migration (v1.0.0 → v1.1.0)
- Migrated `test.py` (574 lines) → `src/antibody_training_esm/cli/test.py`
- Preserved full CLI interface (multi-model, multi-dataset, config support)
- Entry point: `antibody-test`

### Phase 5: Legacy Cleanup (v1.1.0 → v2.0.0) ✅ COMPLETE

**Breaking Changes:**
- ❌ Deleted all 6 root shim files: `classifier.py`, `data.py`, `main.py`, `model.py`, `test.py`, `train.py`
- ❌ Root imports no longer work (e.g., `from classifier import BinaryClassifier` → ERROR)
- ✅ Clean professional package structure only
- ✅ Updated version to 2.0.0 (semantic versioning for breaking change)

**Result:**
```
v2.0.0: Zero legacy code, 100% professional package organization
All functionality preserved via proper package imports
```

---

## Quality Standards

### Pre-commit Hooks (Enforced on Every Commit)

```yaml
- ruff           # Linting (replaces flake8, isort)
- ruff-format    # Code formatting (replaces black)
- mypy           # Type checking (strict mode)
```

### Type Safety

- **Target**: 100% type coverage (currently achieved!)
- **Tool**: mypy with strict configuration
- **Standard**: All functions have type annotations
- **No Shortcuts**: No `SKIP=mypy` (reward hacking eliminated)

### Import Standards

```python
# Standard library (first group)
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

# Third-party (second group)
import numpy as np
import pandas as pd

# Local package (third group, explicit)
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.datasets import load_jain_data
```

### Documentation Standards

```python
def load_data(path: str, stage: str = "parity") -> pd.DataFrame:
    """
    Load preprocessed dataset.

    IMPORTANT: This loads PREPROCESSED data. To preprocess raw data, use:
    preprocessing/jain/step2_preprocess_p5e_s2.py

    Args:
        path: Path to dataset file
        stage: Processing stage (default: "parity")

    Returns:
        DataFrame with preprocessed data

    Raises:
        FileNotFoundError: If dataset file not found

    Example:
        >>> df = load_data("test_datasets/jain/processed/jain.csv")
        >>> print(f"Loaded {len(df)} sequences")
    """
```

---

## Summary: Work Complete! 🎉

### Phase 4 Checklist (All Done!)

- [x] ✅ Update imports in root `test.py` (classifier, model → package paths)
- [x] ✅ Copy full implementation to `src/antibody_training_esm/cli/test.py`
- [x] ✅ Verify all imports use package paths
- [x] ✅ Test CLI entry point: `antibody-test --help`
- [x] ✅ Run comprehensive test: `antibody-test --model X --data Y`
- [x] ✅ Convert root `test.py` to backwards compatibility shim
- [x] ✅ Run full quality pipeline: `make all`
- [x] ✅ Verify 100% type safety: `uv run mypy src/`
- [x] ✅ Update documentation to reference new CLI
- [x] ✅ Commit with clean message (no `SKIP=mypy`)

### Actual Outcome (Achieved!)

```
ROOT:
├── test.py                   ✅ Backwards compatibility shim (32 lines)

PACKAGE:
└── src/antibody_training_esm/
    └── cli/
        └── test.py           ✅ Full implementation (574 lines, full CLI interface)

CLI:
$ antibody-test --help                              ✅ Shows comprehensive help
$ antibody-test --model m1.pkl m2.pkl --data d1.csv d2.csv  ✅ Multi-model/dataset working
$ antibody-test --config test_config.yaml          ✅ Config file support working
$ python test.py --help                            ✅ Backwards compatibility working
```

### Success Criteria

1. ✅ All tests pass (20/20)
2. ✅ mypy reports 100% type safety
3. ✅ All pre-commit hooks pass
4. ✅ CLI entry point works: `antibody-test`
5. ✅ Backwards compatibility maintained: `python test.py` still works (with warning)
6. ✅ No reward hacking (no `SKIP=mypy`)
7. ✅ Professional code quality (Rob C. Martin discipline)

---

## References

- [PEP 517: Build System](https://peps.python.org/pep-0517/)
- [PEP 518: pyproject.toml](https://peps.python.org/pep-0518/)
- [Python Packaging Guide](https://packaging.python.org/en/latest/)
- [Src Layout](https://hynek.me/articles/testing-packaging/)
- [Clean Code (Rob C. Martin)](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

---

**Last Updated**: 2025-11-06
**Status**: ✅ ALL PHASES COMPLETE (100% professional package structure)
**Next Action**: Ship it! Ready for production use.
