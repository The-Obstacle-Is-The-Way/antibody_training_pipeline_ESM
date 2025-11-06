# Import Conventions & Package Structure Guide

**Professional Python Package Organization for Antibody Training ESM**

---

## Executive Summary

This guide defines the **canonical structure** for professional Python package organization, import conventions, and backwards compatibility strategy for the Antibody Training ESM codebase.

**Current Status**: 90% complete (Phase 3 finished, Phase 4 pending)

**Remaining Work**: Migrate `test.py` to package structure

---

## Table of Contents

1. [Professional Python Package Structure](#professional-python-package-structure)
2. [Current State Analysis](#current-state-analysis)
3. [Import Conventions](#import-conventions)
4. [CLI Entry Points](#cli-entry-points)
5. [Backwards Compatibility Strategy](#backwards-compatibility-strategy)
6. [Phase 4 Implementation Plan](#phase-4-implementation-plan)
7. [Quality Standards](#quality-standards)

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
- Backwards compatibility shims (temporary, with deprecation warnings)

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

### ⚠️ **Root Files Status**

```
ROOT FILES:                   STATUS:
├── classifier.py             ✅ Backwards compatibility shim (intentional)
├── data.py                   ✅ Backwards compatibility shim (intentional)
├── main.py                   ✅ Backwards compatibility shim (intentional)
├── model.py                  ✅ Backwards compatibility shim (intentional)
├── train.py                  ✅ Backwards compatibility shim (intentional)
└── test.py                   ❌ FULL LEGACY IMPLEMENTATION (needs migration)
```

### 🔴 **Identified Issues**

1. **`test.py` (575 lines)** - Full-featured legacy script, NOT a shim
   - Contains `ModelTester` class with comprehensive evaluation logic
   - Imports from root shims: `from classifier import BinaryClassifier`
   - Should be migrated to `src/antibody_training_esm/cli/test.py`

2. **`src/antibody_training_esm/cli/test.py` (62 lines)** - Stub implementation
   - Contains TODO comment: "Implement testing logic"
   - Contains fallback message: "Use 'python test.py' for now"
   - Needs to receive full implementation from root `test.py`

3. **CLI Entry Point** - Defined but not implemented
   - `pyproject.toml` defines: `antibody-test = "antibody_training_esm.cli.test:main"`
   - Currently runs stub that does nothing

---

## Import Conventions

### ✅ **Correct Import Patterns**

```python
# From package CLI
from antibody_training_esm.cli.train import main as train_main

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

### ❌ **Legacy Import Patterns (Deprecated)**

```python
# DON'T USE THESE - They work but show deprecation warnings
from classifier import BinaryClassifier
from model import ESMEmbeddingExtractor
from data import load_data
from train import train_model
```

### 🔄 **Backwards Compatibility Shims**

The root shims are **intentional temporary bridges** to avoid breaking existing code:

```python
# Example: classifier.py (root shim)
"""
Binary Classifier Module (BACKWARDS COMPATIBILITY SHIM)

This module is deprecated. Import from antibody_training_esm.core.classifier instead.
"""

import warnings
from antibody_training_esm.core.classifier import BinaryClassifier

warnings.warn(
    "Importing from 'classifier' is deprecated. Use 'from antibody_training_esm.core.classifier import BinaryClassifier' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BinaryClassifier"]
```

**Key Properties:**
1. Re-exports from new package location
2. Emits `DeprecationWarning` with migration guidance
3. Minimal code (5-20 lines)
4. Clear documentation at top
5. Will be removed in v2.0.0 (breaking change)

---

## CLI Entry Points

### Defined in `pyproject.toml`

```toml
[project.scripts]
antibody-train = "antibody_training_esm.cli.train:main"           # ✅ Implemented
antibody-test = "antibody_training_esm.cli.test:main"             # ❌ Stub only
antibody-preprocess = "antibody_training_esm.cli.preprocess:main" # ✅ Guidance only
```

### Usage Examples

```bash
# Training CLI (working)
antibody-train --config configs/config.yaml

# Testing CLI (stub - needs implementation)
antibody-test --model models/classifier.pkl --dataset data/test.csv

# Preprocessing CLI (guidance only - directs to scripts)
antibody-preprocess --dataset jain
```

### Design Principles

1. **Single Responsibility**: Each CLI does ONE thing well
2. **Professional UX**: Clear help messages, error handling, progress bars
3. **Configuration-Driven**: YAML configs for complex workflows
4. **Backwards Compatible**: Root shims delegate to new CLIs

---

## Backwards Compatibility Strategy

### Phase 1: Core Migration (✅ Complete)
- Moved `classifier.py` → `src/antibody_training_esm/core/classifier.py`
- Moved `model.py` → `src/antibody_training_esm/core/embeddings.py`
- Moved `train.py` → `src/antibody_training_esm/core/trainer.py`
- Created root shims with deprecation warnings

### Phase 2: Data & CLI Migration (✅ Complete)
- Moved `data.py` → `src/antibody_training_esm/data/loaders.py`
- Moved training logic → `src/antibody_training_esm/cli/train.py`
- Created `main.py` shim to delegate to new CLI
- Updated all imports in package code

### Phase 3: Dataset Abstractions (✅ Complete)
- Created `src/antibody_training_esm/datasets/base.py` with `AntibodyDataset` ABC
- Implemented dataset-specific loaders (Jain, Harvey, Shehata, Boughter)
- Maintained preprocessing scripts as SSOT
- Fixed mypy errors (100% type safety achieved)

### Phase 4: Test Migration (❌ Pending)
- Migrate `test.py` (575 lines) → `src/antibody_training_esm/cli/test.py`
- Update imports from root shims → package paths
- Convert root `test.py` → backwards compatibility shim
- Verify `antibody-test` CLI works correctly

### Phase 5: Cleanup (❌ Pending)
- Remove all backwards compatibility shims (breaking change)
- Update all documentation to remove legacy import patterns
- Release v2.0.0 with clean package structure

---

## Phase 4 Implementation Plan

### Goal: Migrate `test.py` to Professional Package Structure

#### Current State

```
ROOT:
test.py (575 lines)
├── ModelTester class
├── TestConfig dataclass
├── Comprehensive evaluation logic
├── Confusion matrix plotting
├── Multi-model/multi-dataset testing
├── Imports: from classifier import BinaryClassifier
└── Imports: from model import ESMEmbeddingExtractor

PACKAGE:
src/antibody_training_esm/cli/test.py (62 lines)
├── Stub implementation
├── TODO comment
└── "Use 'python test.py' for now" message
```

#### Migration Steps

**Step 1: Update Imports in Root `test.py`**

```python
# Before (line 43)
from classifier import BinaryClassifier

# After
from antibody_training_esm.core.classifier import BinaryClassifier

# Before (line 130)
from model import ESMEmbeddingExtractor

# After
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
```

**Step 2: Copy Full Implementation to Package CLI**

```bash
# Copy test.py content to src/antibody_training_esm/cli/test.py
# (preserving all functionality: ModelTester, TestConfig, plotting, etc.)
```

**Step 3: Update Package CLI Imports**

Ensure all imports use package paths:
```python
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
```

**Step 4: Verify CLI Entry Point**

```bash
# Install package in editable mode
uv pip install -e .

# Test CLI command
antibody-test --help
antibody-test --model models/classifier.pkl --data test_datasets/jain/test.csv
```

**Step 5: Convert Root `test.py` to Shim**

```python
"""
Test Script (BACKWARDS COMPATIBILITY SHIM)

This script is deprecated. Use the new CLI command instead:
    antibody-test --model models/classifier.pkl --data test_datasets/jain/test.csv

For backwards compatibility, this script delegates to the new CLI.
"""

import sys
import warnings

from antibody_training_esm.cli.test import main as test_main

warnings.warn(
    "Running 'python test.py' is deprecated. Use 'antibody-test' CLI command instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit(test_main())
```

**Step 6: Verify All Tests Pass**

```bash
make all  # Run format → lint → typecheck → test
```

---

## Quality Standards

### Pre-commit Hooks (Enforced on Every Commit)

```yaml
- ruff format    # Code formatting (replaces black)
- ruff lint      # Linting (replaces flake8, isort)
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

## Summary: Remaining Work

### Phase 4 Checklist

- [ ] Update imports in root `test.py` (classifier, model → package paths)
- [ ] Copy full implementation to `src/antibody_training_esm/cli/test.py`
- [ ] Verify all imports use package paths
- [ ] Test CLI entry point: `antibody-test --help`
- [ ] Run comprehensive test: `antibody-test --model X --data Y`
- [ ] Convert root `test.py` to backwards compatibility shim
- [ ] Run full quality pipeline: `make all`
- [ ] Verify 100% type safety: `uv run mypy src/`
- [ ] Update documentation to reference new CLI
- [ ] Commit with clean message (no `SKIP=mypy`)

### Expected Outcome

```
ROOT:
├── test.py                   ✅ Backwards compatibility shim (20 lines)

PACKAGE:
└── src/antibody_training_esm/
    └── cli/
        └── test.py           ✅ Full implementation (575+ lines)

CLI:
$ antibody-test --help        ✅ Shows comprehensive help
$ antibody-test --model X --data Y  ✅ Runs evaluation successfully
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
**Status**: Phase 3 complete, Phase 4 pending
**Next Action**: Senior review and consensus on Phase 4 implementation
