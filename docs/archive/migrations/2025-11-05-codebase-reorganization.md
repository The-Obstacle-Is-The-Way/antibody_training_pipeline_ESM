# Codebase Reorganization Plan

> **✅ COMPLETED IN v2.0.0 (Pre-Phase 5 Directory Reorganization)**
>
> This planning document led to the successful Phase 1-5 package structure migration completed in v2.0.0.
> All root Python files were removed, and the codebase now follows professional package structure.
>
> **Note:** This document references the old output directory structure (`models/`, `data/train/`, `data/test/`) which was later reorganized in Phase 5 (v0.5.0):
> - `models/` → `experiments/checkpoints/{model}/{classifier}/`
> - Training/test data paths remain unchanged
>
> See docs/archive/migrations/v2-structure-migration.md for package structure and docs/developer-guide/directory-organization.md for current directory layout.

**Date:** 2025-11-06
**Status:** ✅ COMPLETED (See v2.0.0 release)
**Goal:** Transform amateur research notebook structure into production-grade professional codebase following Rob C. Martin clean code principles (SOLID, DRY, Single Responsibility)

---

## Current State Analysis (AMATEUR HOUR)

### Critical Issues Identified

1. **EMPTY USELESS `src/` DIRECTORY**
   - `src/antibody_dev_esm/__init__.py` is EMPTY (0 bytes)
   - Nothing imports from it
   - Package name `antibody-dev-esm` doesn't match repo name `antibody_training_pipeline_ESM`
   - This is classic "someone read a tutorial but didn't finish it" code

2. **ROOT-LEVEL SPAGHETTI**
   - 6 core Python modules dumped in repo root: `classifier.py`, `data.py`, `model.py`, `train.py`, `test.py`, `main.py`
   - No namespacing, no organization
   - Everything uses relative imports (`from model import`, `from train import`)
   - This works for a 100-line homework assignment, not production code

3. **INCONSISTENT NAMING**
   - Project name in `pyproject.toml`: `antibody-dev-esm` (kebab-case)
   - Repository name: `antibody_training_pipeline_ESM` (snake_case + camelCase hybrid)
   - No unified naming convention

4. **PREPROCESSING SCRIPTS ORPHANED**
   - `preprocessing/` has dataset-specific subdirs (jain, harvey, shehata, boughter)
   - Each has mix of processing scripts and tests
   - No clear separation between:
     - Data loading/transformation logic
     - Dataset-specific preprocessing
     - Validation/testing

5. **SCRIPTS DUMPED IN `scripts/`**
   - `scripts/validation/` and `scripts/testing/` exist but unclear purpose
   - Overlap with preprocessing validation scripts
   - No clear when to use which

6. **EXPERIMENTS FOLDER UNCONTROLLED**
   - `experiments/` has executable scripts mixed with results
   - No separation between experiment code and experiment outputs
   - `novo_parity/` has scripts that probably belong in main codebase

7. **TESTS ISOLATED FROM CODE**
   - `tests/integration/` exists but only integration tests
   - No unit tests for core modules
   - No test structure matching code structure

---

## Professional Structure (SOLID GANG)

### Target Directory Layout

```
antibody_training_pipeline_esm/          # Root (renamed from ESM)
├── src/
│   └── antibody_training_esm/           # Main package (clear name)
│       ├── __init__.py
│       ├── core/                        # Core ESM embedding & ML
│       │   ├── __init__.py
│       │   ├── embeddings.py           # ESM model wrapper (was model.py)
│       │   ├── classifier.py           # ML classifier (was classifier.py)
│       │   └── trainer.py              # Training logic (was train.py)
│       ├── data/                        # Data handling
│       │   ├── __init__.py
│       │   ├── loaders.py              # Dataset loading (from data.py)
│       │   ├── preprocessing.py        # Common preprocessing utils
│       │   └── transforms.py           # Data transformations
│       ├── datasets/                    # Dataset-specific processors
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract base dataset class
│       │   ├── jain.py                 # Jain dataset processor
│       │   ├── harvey.py               # Harvey dataset processor
│       │   ├── shehata.py              # Shehata dataset processor
│       │   └── boughter.py             # Boughter dataset processor
│       ├── evaluation/                  # Testing & evaluation
│       │   ├── __init__.py
│       │   ├── metrics.py              # Evaluation metrics
│       │   └── tester.py               # Test runner (was test.py)
│       ├── utils/                       # Utilities
│       │   ├── __init__.py
│       │   ├── config.py               # Config handling
│       │   ├── logging.py              # Logging setup
│       │   └── io.py                   # File I/O helpers
│       └── cli/                         # Command-line interface
│           ├── __init__.py
│           ├── train.py                # CLI for training
│           ├── test.py                 # CLI for testing
│           └── preprocess.py           # CLI for preprocessing
│
├── scripts/                             # Standalone utility scripts
│   ├── preprocessing/                   # One-off preprocessing scripts
│   │   ├── jain/
│   │   ├── harvey/
│   │   ├── shehata/
│   │   └── boughter/
│   ├── validation/                      # Validation scripts
│   └── experiments/                     # Experiment runners
│
├── tests/                               # Test suite
│   ├── unit/                           # Unit tests (NEW)
│   │   ├── test_core/
│   │   ├── test_data/
│   │   ├── data/test/
│   │   └── test_utils/
│   └── integration/                     # Integration tests (EXISTING)
│       └── test_*_embedding_compatibility.py
│
├── experiments/                         # Experiment outputs only
│   ├── novo_parity/
│   │   ├── results/                    # Outputs only
│   │   └── datasets/                   # Experiment-specific data
│   └── strict_qc_2025-11-04/
│       ├── data/
│       ├── configs/
│       └── docs/
│
├── configs/                             # Configuration files
│   ├── config.yaml                     # Default config
│   └── datasets/                       # Dataset-specific configs
│
├── docs/                                # Documentation (NO CHANGE)
│
├── models/                              # Trained model artifacts (NO CHANGE)
│
├── data/train/                      # Training data (NO CHANGE)
├── data/test/                       # Test data (NO CHANGE)
├── reference_repos/                     # External repos (NO CHANGE)
├── literature/                          # Papers (NO CHANGE)
│
├── pyproject.toml                       # Project config (UPDATED)
├── README.md
├── USAGE.md
├── Makefile
└── .gitignore
```

---

## Key Design Principles Applied

### 1. Single Responsibility Principle (SRP)
- Each module has ONE clear purpose
- `core/embeddings.py` - ONLY ESM model interaction
- `core/classifier.py` - ONLY ML classification
- `data/loaders.py` - ONLY data loading
- `datasets/jain.py` - ONLY Jain-specific processing

### 2. Open/Closed Principle (OCP)
- `datasets/base.py` defines abstract `Dataset` interface
- Each dataset (jain, harvey, shehata, boughter) extends base
- New datasets can be added without modifying existing code

### 3. Dependency Inversion Principle (DIP)
- High-level modules (`core/trainer.py`) depend on abstractions (`datasets/base.py`)
- Not on concrete implementations (`datasets/jain.py`)

### 4. DRY (Don't Repeat Yourself)
- Common preprocessing logic → `data/preprocessing.py`
- Dataset-specific logic → `datasets/<name>.py`
- No code duplication across preprocessing scripts

### 5. Clear Separation of Concerns
- **Core ML logic** → `src/antibody_training_esm/core/`
- **Data handling** → `src/antibody_training_esm/data/`
- **Dataset-specific** → `src/antibody_training_esm/datasets/`
- **CLI interfaces** → `src/antibody_training_esm/cli/`
- **One-off scripts** → `scripts/`
- **Test outputs** → `experiments/`

---

## Migration Plan

### Phase 1: Setup New Structure (NO BREAKING CHANGES)
1. Create new `src/antibody_training_esm/` package structure
2. Keep old root-level files in place (backwards compatibility)
3. Update `pyproject.toml` package name to match structure

### Phase 2: Migrate Core Modules
1. Move `model.py` → `src/antibody_training_esm/core/embeddings.py`
2. Move `classifier.py` → `src/antibody_training_esm/core/classifier.py`
3. Move `train.py` logic → `src/antibody_training_esm/core/trainer.py`
4. Move `data.py` → `src/antibody_training_esm/data/loaders.py`
5. Update imports in moved files to use absolute imports

### Phase 3: Create Dataset Abstractions
1. Create `src/antibody_training_esm/datasets/base.py` with abstract class
2. Extract Jain logic → `src/antibody_training_esm/datasets/jain.py`
3. Extract Harvey logic → `src/antibody_training_esm/datasets/harvey.py`
4. Extract Shehata logic → `src/antibody_training_esm/datasets/shehata.py`
5. Extract Boughter logic → `src/antibody_training_esm/datasets/boughter.py`

### Phase 4: Migrate CLI & Utils
1. Create CLI wrappers in `src/antibody_training_esm/cli/`
2. Move config handling → `src/antibody_training_esm/utils/config.py`
3. Move test logic → `src/antibody_training_esm/evaluation/tester.py`
4. Update `main.py`, `test.py`, `train.py` to use new CLI modules

### Phase 5: Reorganize Scripts
1. Move `preprocessing/jain/` scripts → `scripts/preprocessing/jain/`
2. Move `preprocessing/harvey/` scripts → `scripts/preprocessing/harvey/`
3. Move `preprocessing/shehata/` scripts → `scripts/preprocessing/shehata/`
4. Move `preprocessing/boughter/` scripts → `scripts/preprocessing/boughter/`
5. Consolidate `scripts/validation/` and preprocessing validation

### Phase 6: Update Tests
1. Create `tests/unit/` structure mirroring `src/antibody_training_esm/`
2. Write unit tests for each new module
3. Keep `tests/integration/` in place
4. Update integration tests to use new imports

### Phase 7: Update Experiments
1. Move executable scripts out of `experiments/benchmarks/novo_parity/scripts/` (now stored in `archive` branch)
2. Keep only results and datasets in `experiments/` (or archive them if historical)
3. Executable experiment code → either `scripts/experiments/` or `src/`

### Phase 8: Update Documentation
1. Update all README files with new import paths
2. Update `USAGE.md` with new CLI commands
3. Add API documentation for new package structure
4. Update all inline documentation references

### Phase 9: Deprecate Old Structure
1. Add deprecation warnings to old root-level modules
2. Create backwards-compatibility shims (old modules import new ones)
3. Update CI/CD to use new structure
4. Update Makefile targets

### Phase 10: Remove Old Structure
1. Delete old root-level `.py` files (after transition period)
2. Delete empty `preprocessing/` directory structure
3. Clean up old `src/antibody_dev_esm/` (useless directory)
4. Final cleanup and linting

---

## Import Examples After Refactor

### Old Way (AMATEUR)
```python
# Root-level spaghetti
from model import ESMEmbeddingExtractor
from classifier import BinaryClassifier
from train import train_model
```

### New Way (PROFESSIONAL)
```python
# Clear package structure
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.trainer import train_model

# Or use module-level imports for cleaner code
from antibody_training_esm import core, data, datasets

model = core.embeddings.ESMEmbeddingExtractor(...)
classifier = core.classifier.BinaryClassifier(...)
dataset = datasets.jain.JainDataset(...)
```

### CLI After Refactor (Achieved in v2.0.0)
```bash
# ❌ v1.x way (REMOVED - files no longer exist)
# python train.py --config configs/config.yaml

# ✅ v2.0.0 way (current - use this)
python -m antibody_training_esm.cli.train --config configs/config.yaml

# ✅ Or with entry point (defined in pyproject.toml) - RECOMMENDED
antibody-train --config configs/config.yaml
```

---

## Verification Checklist

After each phase, verify:
- [ ] All imports resolve correctly
- [ ] `make lint` passes (ruff)
- [ ] `make type-check` passes (mypy)
- [ ] `make test` passes (pytest)
- [ ] Documentation is updated
- [ ] No circular imports
- [ ] No broken references in docs

---

## Benefits of This Reorganization

1. **Discoverability**: Clear package structure makes it obvious where code lives
2. **Maintainability**: Single Responsibility means easier debugging and testing
3. **Extensibility**: New datasets/models can be added without touching existing code
4. **Testability**: Unit tests can target individual modules in isolation
5. **Professional**: This is how real production ML codebases are organized
6. **Reusability**: Core modules can be imported by other projects
7. **Documentation**: Clear structure makes auto-documentation work properly
8. **Onboarding**: New developers can understand the codebase in minutes, not hours

---

## Risk Mitigation

### Risk: Breaking existing scripts/notebooks
**Mitigation**: Keep backwards-compatibility shims in place during transition

### Risk: Import path hell during migration
**Mitigation**: Migrate one module at a time, test after each step

### Risk: Git history gets messy
**Mitigation**: Use `git mv` for file moves to preserve history

### Risk: CI/CD breaks
**Mitigation**: Update CI config in same PR as structure changes

### Risk: Documentation gets out of sync
**Mitigation**: Update docs in same commit as code changes

---

## Timeline Estimate

- **Phase 1-2**: 2-3 hours (setup + core modules)
- **Phase 3**: 3-4 hours (dataset abstractions)
- **Phase 4**: 2 hours (CLI & utils)
- **Phase 5-6**: 3 hours (scripts & tests)
- **Phase 7-8**: 2 hours (experiments & docs)
- **Phase 9-10**: 1-2 hours (deprecation & cleanup)

**Total**: 13-16 hours of focused refactoring work

---

## Success Criteria

1. ✅ No code in repository root except config files and documentation
2. ✅ All Python modules under `src/antibody_training_esm/`
3. ✅ Clear separation: core ML / data / datasets / CLI / utils
4. ✅ All imports use absolute paths from package root
5. ✅ 100% test coverage maintained (currently 20/20 passing)
6. ✅ All linters pass (ruff, mypy, bandit)
7. ✅ Documentation updated to reflect new structure
8. ✅ Makefile targets all work with new structure
9. ✅ No circular dependencies
10. ✅ Can pip install package and import from any directory

---

## Questions to Resolve

~~1. **Package naming**: Keep `antibody-dev-esm` or rename to `antibody-training-esm` to match repo?~~
~~2. **Entry points**: Should we add CLI entry points in `pyproject.toml` for easier execution?~~
~~3. **Preprocessing scripts**: Keep as standalone scripts or integrate into package?~~
~~4. **Experiment scripts**: Move to package or keep in `scripts/experiments/`?~~
~~5. **Config management**: Use hydra/omegaconf or stick with PyYAML?~~

---

## Professional Decisions (RESOLVED)

### 1. Package Naming ✅
**Decision**: Rename to match repository structure
- **PyPI name** (in `pyproject.toml`): `antibody-training-esm` (kebab-case, PyPI standard)
- **Python package**: `antibody_training_esm` (snake_case, Python convention)
- **Rationale**: Consistency with repo name, follows Python/PyPI naming standards

### 2. CLI Entry Points ✅
**Decision**: Add proper CLI entry points
- `antibody-train` - Training interface
- `antibody-test` - Testing/evaluation interface
- `antibody-preprocess` - Data preprocessing interface
- **Rationale**: Professional packages use entry points (like `pytest`, `black`, `ruff`), not `python -m` calls

### 3. Preprocessing Scripts ✅
**Decision**: Hybrid approach
- Keep scripts as standalone in `scripts/preprocessing/` (one-off data converters)
- Extract reusable logic into `src/antibody_training_esm/datasets/` (library code)
- **Rationale**: Scripts are run once to prepare data; library code is imported and reused

### 4. Experiment Scripts ✅
**Decision**: Move to `scripts/experiments/`
- Keep only results/datasets in `experiments/`
- Executable code → `scripts/experiments/`
- **Rationale**: Separation of code and outputs

### 5. Config Management ✅
**Decision**: Stick with PyYAML for now
- Don't force Hydra/OmegaConf unless needed
- Keep existing YAML-based config working
- **Rationale**: No breaking changes; can upgrade later if complexity demands it

---

**Status**: APPROVED FOR IMPLEMENTATION
**Next Step**: Get user approval, then execute Phase 1

---

**Remember**: "Clean code is not written by following a set of rules. Clean code is written by following discipline." - Robert C. Martin

Let's show these master's thesis folks how production code is done. 💪
