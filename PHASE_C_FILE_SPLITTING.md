# Phase C: File Splitting (Core Refactoring)

**Effort:** 3-4 hours
**Risk:** HIGH
**Dependencies:** Phases A & B complete
**Branch:** `claude/refactor-phase-c`

---

## Overview

Split 4 massive files (>500 lines) into modular components following Single Responsibility Principle.

**Goal:** Break monolithic files into maintainable modules while preserving all functionality.

**Why this is HIGH risk:**
- Significant structural changes
- Lots of imports to update
- Requires extensive testing to ensure no regressions

---

## Fixes Included

| File | Current Lines | Target Lines | New Modules |
|------|---------------|--------------|-------------|
| `core/trainer.py` | 961 | ~350 | 3 modules |
| `datasets/base.py` | 627 | ~350 | 2-3 modules/mixins |
| `boughter/stage1_dna_translation.py` | 598 | ~250 | 2 modules |
| `boughter/stage2_stage3_annotation_qc.py` | 519 | ~250 | 2 modules |

**Total:** 4 files split into ~10 focused modules

---

## Task C1: Split trainer.py (1.5 hours)

### Current State
**File:** `src/antibody_training_esm/core/trainer.py` (961 lines)

**What it does:**
- Main `train_model()` function
- Embedding cache management
- Cross-validation logic
- Metrics calculation and logging
- Model serialization (.pkl save/load)

### Target Structure

```
src/antibody_training_esm/core/
├── trainer.py                    # Main orchestration (~300 lines)
└── training/
    ├── __init__.py               # Re-export public APIs
    ├── cache.py                  # Embedding cache ops (~200 lines)
    ├── metrics.py                # Evaluation metrics (~250 lines)
    └── serialization.py          # Model save/load (~150 lines)
```

### Implementation Steps

**Step 1: Create training/ directory (5 min)**
```bash
mkdir -p src/antibody_training_esm/core/training
touch src/antibody_training_esm/core/training/__init__.py
```

**Step 2: Extract cache.py (30 min)**

Create `src/antibody_training_esm/core/training/cache.py`:

```python
"""Embedding cache management for training pipeline."""

import hashlib
import logging
from pathlib import Path
from typing import Optional
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages embedding cache operations."""

    def __init__(self, cache_dir: Path):
        """Initialize cache manager."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(
        self,
        model_name: str,
        dataset_path: Path,
        revision: str,
    ) -> str:
        """
        Generate cache key from model + dataset + revision.

        Returns SHA-256 hash as hex string.
        """
        key_string = f"{model_name}_{dataset_path}_{revision}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path from cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def load_cached_embeddings(
        self,
        cache_key: str,
    ) -> Optional[np.ndarray]:
        """Load embeddings from cache if exists."""
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path}")
            return None

        logger.info(f"Loading cached embeddings from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            return cache_data["embeddings"]
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not load cache {cache_path}: {e}")
            return None

    def save_embeddings_to_cache(
        self,
        cache_key: str,
        embeddings: np.ndarray,
    ) -> None:
        """Save embeddings to cache."""
        cache_path = self.get_cache_path(cache_key)

        cache_data = {
            "embeddings": embeddings,
            "shape": embeddings.shape,
        }

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved embeddings to cache: {cache_path}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not save cache {cache_path}: {e}")

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """Clear cache (all or specific key)."""
        if cache_key:
            cache_path = self.get_cache_path(cache_key)
            try:
                cache_path.unlink()
                logger.info(f"Deleted cache file: {cache_path}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not delete cache {cache_path}: {e}")
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                    logger.info(f"Deleted cache file: {cache_file}")
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not delete cache {cache_file}: {e}")
```

**Step 3: Extract metrics.py (30 min)**

Create `src/antibody_training_esm/core/training/metrics.py`:

```python
"""Metrics calculation and logging for model evaluation."""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Computes and logs evaluation metrics."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metric name -> value
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def log_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
        """Log metrics to console."""
        if prefix:
            logger.info(f"\n{prefix} Metrics:")
        else:
            logger.info("\nMetrics:")

        for key, value in metrics.items():
            if key == "confusion_matrix":
                logger.info(f"  Confusion Matrix:\n{np.array(value)}")
            elif isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    @staticmethod
    def compute_cv_summary(
        cv_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute summary statistics across CV folds.

        Args:
            cv_metrics: List of metric dicts from each fold

        Returns:
            Dict with mean/std for each metric
        """
        summary = {}

        # Get metric names (excluding confusion matrix)
        metric_names = [k for k in cv_metrics[0].keys() if k != "confusion_matrix"]

        for metric_name in metric_names:
            values = [m[metric_name] for m in cv_metrics if m[metric_name] is not None]
            if values:
                summary[f"{metric_name}_mean"] = np.mean(values)
                summary[f"{metric_name}_std"] = np.std(values)

        return summary
```

**Step 4: Extract serialization.py (20 min)**

Create `src/antibody_training_esm/core/training/serialization.py`:

```python
"""Model serialization for saving/loading trained models."""

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelSerializer:
    """Handles model save/load operations."""

    @staticmethod
    def save_model(model: Any, output_path: Path) -> None:
        """
        Save trained model to disk.

        Args:
            model: Trained sklearn model
            output_path: Path to save .pkl file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to: {output_path}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to save model to {output_path}: {e}")
            raise

    @staticmethod
    def load_model(model_path: Path) -> Any:
        """
        Load trained model from disk.

        Args:
            model_path: Path to .pkl file

        Returns:
            Loaded model object
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from: {model_path}")
            return model
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
```

**Step 5: Update training/__init__.py (5 min)**

```python
"""Training utilities for antibody classification."""

from antibody_training_esm.core.training.cache import CacheManager
from antibody_training_esm.core.training.metrics import MetricsLogger
from antibody_training_esm.core.training.serialization import ModelSerializer

__all__ = [
    "CacheManager",
    "MetricsLogger",
    "ModelSerializer",
]
```

**Step 6: Update trainer.py imports (10 min)**

At the top of `src/antibody_training_esm/core/trainer.py`:

```python
# Add new imports
from antibody_training_esm.core.training import (
    CacheManager,
    MetricsLogger,
    ModelSerializer,
)

# Inside train_model function, replace:
# - Direct cache operations → CacheManager methods
# - Metrics calculations → MetricsLogger methods
# - Model save/load → ModelSerializer methods
```

**Step 7: Update tests (15 min)**

Update `tests/unit/core/test_trainer.py` to import from new modules if needed.

### Verification
```bash
# Check file size reduced
wc -l src/antibody_training_esm/core/trainer.py
# Should be <400 lines

# All trainer tests pass
uv run pytest tests/unit/core/test_trainer.py -v

# CLI still works
uv run antibody-train --help

# Type checking passes
uv run mypy src/antibody_training_esm/core --strict
```

### Success Criteria
- [ ] `trainer.py` reduced to <400 lines
- [ ] 3 new modules created (cache, metrics, serialization)
- [ ] All tests pass
- [ ] `antibody-train` CLI works
- [ ] Type checking passes

---

## Task C2: Split stage1_dna_translation.py (1 hour)

### Current State
**File:** `preprocessing/boughter/stage1_dna_translation.py` (598 lines)

**What it does:**
- DNA → Protein translation
- Translation validation
- CSV I/O operations

### Target Structure

```
preprocessing/boughter/
├── stage1_dna_translation.py     # Main orchestration (~200 lines)
└── translation/
    ├── __init__.py               # Re-exports
    ├── dna_translator.py         # Translation logic (~250 lines)
    └── validation.py             # Validation (~150 lines)
```

### Implementation Steps

**Step 1: Create translation/ directory (5 min)**
```bash
mkdir -p preprocessing/boughter/translation
touch preprocessing/boughter/translation/__init__.py
```

**Step 2: Extract dna_translator.py (30 min)**

Create `preprocessing/boughter/translation/dna_translator.py`:
- Move DNATranslator class
- Move codon table
- Move translation logic

**Step 3: Extract validation.py (15 min)**

Create `preprocessing/boughter/translation/validation.py`:
- Move validation functions
- Move quality check logic

**Step 4: Update main script (10 min)**

Update `stage1_dna_translation.py` to import from new modules.

### Verification
```bash
# Check file size reduced
wc -l preprocessing/boughter/stage1_dna_translation.py
# Should be <300 lines

# Script still works
uv run python preprocessing/boughter/validate_stage1.py
```

### Success Criteria
- [ ] `stage1_dna_translation.py` reduced to <300 lines
- [ ] 2 new modules created
- [ ] Validation script passes

---

## Task C3: Split stage2_stage3_annotation_qc.py (1 hour)

### Current State
**File:** `preprocessing/boughter/stage2_stage3_annotation_qc.py` (519 lines)

**What it does:**
- ANARCI annotation
- QC filtering
- CSV processing

### Target Structure

```
preprocessing/boughter/
├── stage2_stage3_annotation_qc.py  # Main orchestration (~200 lines)
└── annotation/
    ├── __init__.py                 # Re-exports
    ├── anarci.py                   # ANARCI logic (~200 lines)
    └── qc.py                       # QC filters (~150 lines)
```

### Implementation Steps

**Step 1: Create annotation/ directory (5 min)**
```bash
mkdir -p preprocessing/boughter/annotation
touch preprocessing/boughter/annotation/__init__.py
```

**Step 2: Extract anarci.py (30 min)**

Create `preprocessing/boughter/annotation/anarci.py`:
- Move ANARCI annotation logic
- Move numbering functions

**Step 3: Extract qc.py (15 min)**

Create `preprocessing/boughter/annotation/qc.py`:
- Move QC filter logic
- Move quality check functions

**Step 4: Update main script (10 min)**

Update `stage2_stage3_annotation_qc.py` to import from new modules.

### Verification
```bash
# Check file size reduced
wc -l preprocessing/boughter/stage2_stage3_annotation_qc.py
# Should be <300 lines

# Script still works
uv run python preprocessing/boughter/validate_stages2_3.py
```

### Success Criteria
- [ ] `stage2_stage3_annotation_qc.py` reduced to <300 lines
- [ ] 2 new modules created
- [ ] Validation script passes

---

## Task C4: Split datasets/base.py (1 hour)

### Current State
**File:** `src/antibody_training_esm/datasets/base.py` (627 lines)

**What it does:**
- Base class definition + logger setup
- Sequence validation utilities
- ANARCI annotation helpers
- Fragment construction and CSV writing

### Target Structure

```
src/antibody_training_esm/datasets/
├── base.py                   # Abstract interface + high-level helpers (~200 lines)
└── base_components/
    ├── __init__.py
    ├── validation.py         # sanitize_sequence, validate_sequences, print_statistics
    ├── annotation.py         # annotate_sequence, annotate_all
    └── fragments.py          # create_fragments, create_fragment_csvs
```

### Implementation Steps

**Step 1: Create base_components/ package (5 min)**
```bash
mkdir -p src/antibody_training_esm/datasets/base_components
touch src/antibody_training_esm/datasets/base_components/__init__.py
```

**Step 2: Move utilities (30 min)**
- `validation.py`: `VALID_AMINO_ACIDS`, `sanitize_sequence`, `validate_sequences`, `print_statistics`
- `annotation.py`: `annotate_sequence`, `annotate_all`
- `fragments.py`: `create_fragments`, `create_fragment_csvs`

**Step 3: Keep abstract class lean (15 min)**
- `AntibodyDataset` stays in `base.py` and imports helpers from `base_components`
- Re-export helpers in `base_components/__init__.py` for tests

**Step 4: Update imports/tests (10 min)**
- Update dataset subclasses and tests to import from new locations if needed.

### Verification
```bash
wc -l src/antibody_training_esm/datasets/base.py  # Target <300 lines
uv run pytest tests/unit/datasets tests/integration/test_*embedding_compatibility.py
uv run mypy src/antibody_training_esm/datasets --strict
```

### Success Criteria
- [ ] `datasets/base.py` reduced to <300 lines
- [ ] `base_components/` package created with validation/annotation/fragment helpers
- [ ] Dataset unit/integration tests pass

---

## Phase Completion Checklist

### All Tasks Complete
- [ ] Task C1: Split trainer.py (3 modules)
- [ ] Task C2: Split stage1_dna_translation.py (2 modules)
- [ ] Task C3: Split stage2_stage3_annotation_qc.py (2 modules)
- [ ] Task C4: Split datasets/base.py (3 modules)

### File Size Verification
```bash
# No files should exceed 500 lines
find src preprocessing -name "*.py" -exec wc -l {} \; | awk '$1 > 500 {print $2": "$1" lines"}'
# Should return NOTHING
```

### Quality Gates
- [ ] All tests pass: `uv run pytest`
- [ ] Type checking: `uv run mypy src/ --strict`
- [ ] Linting: `uv run ruff check src/ preprocessing/`
- [ ] CLI works: `uv run antibody-train --help`
- [ ] Preprocessing scripts work
- [ ] `make all` passes

### Git Workflow
```bash
# Create branch
git checkout dev
git pull origin dev
git checkout -b claude/refactor-phase-c

# Make changes (complete all 3 tasks above)

# Commit
git add -A
git commit -m "$(cat <<'EOF'
refactor: Phase C - Split oversized files into modules

Split 4 massive files (>500 lines) into maintainable modular components.
Follows Single Responsibility Principle while preserving all functionality.

**Task C1: Split trainer.py (961 → ~350 lines)**
Created src/antibody_training_esm/core/training/ with 3 modules:
- cache.py: CacheManager for embedding cache operations
- metrics.py: MetricsLogger for evaluation metrics
- serialization.py: ModelSerializer for .pkl save/load

**Task C2: Split stage1_dna_translation.py (598 → ~250 lines)**
Created preprocessing/boughter/translation/ with 2 modules:
- dna_translator.py: DNATranslator class and codon tables
- validation.py: Translation validation logic

**Task C3: Split stage2_stage3_annotation_qc.py (519 → ~250 lines)**
Created preprocessing/boughter/annotation/ with 2 modules:
- anarci.py: ANARCI annotation logic
- qc.py: QC filtering and quality checks

**Task C4: Split datasets/base.py (627 → ~300 lines)**
Created datasets/base_components/ with 3 modules:
- validation.py: sanitize/validate/print statistics
- annotation.py: ANARCI helpers
- fragments.py: fragment construction + CSV writers

**Quality Gates: ✅ ALL PASSED**
- pytest (full suite): PASSED
- mypy strict: PASSED
- ruff check: PASSED
- CLI functionality: PASSED
- Preprocessing scripts: PASSED
- make all: PASSED

**Impact:**
- Improved maintainability (smaller, focused modules)
- Better testability (isolated components)
- Clearer architecture (SRP compliance)
- No functional changes (pure refactoring)

**Files Changed:**
- SPLIT: 4 files → ~10 focused modules
- Line count: reduced per-file, better organized

**Next:** Phase D - Code Deduplication
EOF
)"

# Push and create PR
git push -u origin claude/refactor-phase-c
gh pr create --title "Phase C: File Splitting - Split 3 Massive Files" \
  --body "Completes Phase C of technical debt cleanup. See commit message for details." \
  --base dev
```

---

## Success Metrics

**Before Phase C (validated 2025-11-20):**
- Files >500 lines: 4 files (2705 lines total)
- trainer.py: 961 lines
- datasets/base.py: 627 lines
- stage1_dna_translation.py: 598 lines
- stage2_stage3_annotation_qc.py: 519 lines

**After Phase C (target):**
- Files >500 lines: 0 ✅
- trainer.py: ~350 lines ✅
- datasets/base.py: ~300 lines ✅
- stage1_dna_translation.py: ~250 lines ✅
- stage2_stage3_annotation_qc.py: ~250 lines ✅
- New modules: ~10 focused files ✅

---

**Phase C Complete! Ready for Phase D (Code Deduplication)**
