# Output Directory Investigation & Analysis

**Date**: 2025-11-15
**Context**: Post-migration validation - verifying all outputs go to correct locations
**Status**: 🔍 **INVESTIGATION COMPLETE - ACTION ITEMS IDENTIFIED**

---

## TL;DR - Directory Structure Analysis

**Three output systems exist with different purposes:**

1. **`outputs/`** - Hydra-managed training scratch space ✅ **CORRECT** (gitignored)
2. **`test_results/`** - Curated test predictions ✅ **CORRECT** (tracked in git)
3. **`outputs/test_dataset/`** - Unit test artifact ⚠️ **HARMLESS** (auto-regenerates)

**Critical Finding**: CV results are **NOT saved to files** - only logged. This may need addressing for reproducibility.

---

## Directory Structure - Full Tree

### 1. `outputs/` - Training Runs (Hydra-Managed)

```
outputs/
├── novo_replication/           # ESM2-650M baseline runs (Nov 11)
│   ├── 2025-11-11_18-00-50/   # 8 timestamped runs
│   ├── 2025-11-11_18-01-14/
│   ├── 2025-11-11_18-01-38/
│   ├── 2025-11-11_21-48-22/
│   ├── 2025-11-11_21-50-06/
│   ├── 2025-11-11_21-51-11/
│   ├── 2025-11-11_21-52-43/
│   ├── 2025-11-11_23-22-41/
│   └── 2025-11-11_23-22-49/   # Latest run (VERIFIED)
│       ├── .hydra/             # Hydra config snapshots
│       │   ├── config.yaml    # Full config used for this run
│       │   ├── hydra.yaml
│       │   └── overrides.yaml
│       ├── logs/
│       │   └── training.log   # COMPLETE training log with CV results
│       └── trainer.log        # Top-level trainer log
│
├── post_migration_smoke_test/  # ESM1v validation run (Nov 15)
│   └── 2025-11-15_15-43-37/
│       ├── .hydra/
│       ├── logs/
│       │   └── training.log   # 2-fold CV results logged here
│       └── trainer.log
│
└── test_dataset/               # ⚠️ Created by unit tests (EMPTY)
```

**Purpose**: Hydra working directory for training experiments
**Configured in**: `src/antibody_training_esm/conf/hydra/default.yaml:3`
**Git Status**: ✅ **GITIGNORED** (`.gitignore:91`)
**Contents**:
- Hydra config snapshots (`.hydra/`)
- Training logs (`logs/training.log`)
- Trainer logs (`trainer.log`)
- **NO MODELS** (models saved to `models/` separately)
- **NO CV RESULT FILES** (only in logs)

---

### 2. `test_results/` - Test Predictions (Curated Artifacts)

```
test_results/
├── esm1v/
│   └── logreg/
│       ├── harvey/
│       │   ├── confusion_matrix_VHH_only_harvey.png
│       │   ├── detailed_results_VHH_only_harvey_20251106_223905.yaml
│       │   └── predictions_boughter_vh_esm1v_logreg_VHH_only_harvey_20251106_223905.csv
│       ├── jain/
│       │   ├── confusion_matrix_VH_only_jain_test_PARITY_86.png
│       │   ├── detailed_results_VH_only_jain_test_PARITY_86_20251106_211815.yaml
│       │   └── predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251106_211815.csv
│       └── shehata/
│           ├── confusion_matrix_VH_only_shehata.png
│           ├── detailed_results_VH_only_shehata_20251106_212500.yaml
│           └── predictions_boughter_vh_esm1v_logreg_VH_only_shehata_20251106_212500.csv
│
└── esm2_650m/
    └── logreg/
        ├── VHH_only_harvey/
        │   ├── confusion_matrix_boughter_vh_esm2_650m_logreg_VHH_only_harvey.png
        │   ├── detailed_results_boughter_vh_esm2_650m_logreg_VHH_only_harvey_20251112_051907.yaml
        │   └── predictions_boughter_vh_esm2_650m_logreg_VHH_only_harvey_20251112_051907.csv
        ├── VH_only_jain_test_PARITY_86/
        │   ├── confusion_matrix_boughter_vh_esm2_650m_logreg_VH_only_jain_test_PARITY_86.png
        │   ├── detailed_results_boughter_vh_esm2_650m_logreg_VH_only_jain_test_PARITY_86_20251111_235027.yaml
        │   └── predictions_boughter_vh_esm2_650m_logreg_VH_only_jain_test_PARITY_86_20251111_235027.csv
        └── VH_only_shehata/
            ├── confusion_matrix_boughter_vh_esm2_650m_logreg_VH_only_shehata.png
            ├── detailed_results_boughter_vh_esm2_650m_logreg_VH_only_shehata_20251111_235531.yaml
            └── predictions_boughter_vh_esm2_650m_logreg_VH_only_shehata_20251111_235531.csv
```

**Purpose**: Historical test set predictions for benchmarking
**Created by**: `antibody-test` CLI (`src/antibody_training_esm/cli/test.py`)
**Git Status**: ✅ **TRACKED IN GIT** (committed for reproducibility)
**Hierarchy**: `test_results/{model}/{classifier}/{dataset}/`
**Contents**:
- Confusion matrices (PNG)
- Detailed YAML results (metrics, config)
- Per-sample predictions (CSV)

**Note**: These are from **pre-migration** runs (Nov 6-12) using old model paths. They remain valid historical benchmarks.

---

### 3. `outputs/test_dataset/` - Unit Test Artifact

**Purpose**: Auto-created by `tests/unit/datasets/test_base.py:81`
**Source**: `src/antibody_training_esm/datasets/base.py:79-80`
```python
self.output_dir = (
    Path(output_dir) if output_dir else Path(f"outputs/{dataset_name}")
)
```

**Status**: ⚠️ **HARMLESS**
**Action**: Can be deleted locally, but will regenerate when running unit tests
**Impact**: None (no production code depends on it)

---

## Critical Finding: CV Results Not Saved to Files

### Current Behavior

**CV results are ONLY logged, never persisted to files:**

```bash
# outputs/novo_replication/2025-11-11_23-22-49/logs/training.log
2025-11-11 23:24:23,855 - antibody_training_esm.core.trainer - INFO - Cross-validation Results:
2025-11-11 23:24:23,855 - antibody_training_esm.core.trainer - INFO -   cv_accuracy: 0.6413 (+/- 0.0972)
2025-11-11 23:24:23,855 - antibody_training_esm.core.trainer - INFO -   cv_f1: 0.6604 (+/- 0.0994)
2025-11-11 23:24:23,855 - antibody_training_esm.core.trainer - INFO -   cv_roc_auc: nan (+/- nan)
```

**No structured files created** (e.g., `cv_results.yaml` or `cv_results.json`)

### Where CV Results Live

| Run | Type | CV Folds | Location | Accessible? |
|-----|------|----------|----------|-------------|
| `novo_replication` (Nov 11) | **10-fold CV** | 10 | `outputs/novo_replication/.../logs/training.log` | ✅ YES |
| `post_migration_smoke_test` (Nov 15) | 2-fold CV (validation) | 2 | `outputs/post_migration_smoke_test/.../logs/training.log` | ✅ YES |

**Verification**:
```bash
# 10-fold CV results from novo_replication run:
cv_accuracy: 0.6413 (+/- 0.0972)
cv_f1: 0.6604 (+/- 0.0994)
cv_roc_auc: nan (+/- nan)

# 2-fold CV results from smoke test:
cv_accuracy: 0.6684 (+/- 0.0869)
cv_f1: 0.6780 (+/- 0.0912)
cv_roc_auc: 0.7403 (+/- 0.0890)
```

### Problem Statement

**For reproducibility and paper writing, we need:**
1. Structured CV results files (YAML/JSON)
2. Per-fold predictions (for error analysis)
3. Aggregated metrics across runs

**Current workaround**: Parse `logs/training.log` manually 😬

---

## Verified Configuration Paths

### Hydra Output Configuration
**File**: `src/antibody_training_esm/conf/hydra/default.yaml`
```yaml
run:
  dir: outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}

sweep:
  dir: outputs/sweeps/${experiment.name}
  subdir: ${hydra.job.num}

job:
  chdir: false  # Don't change working directory
```

### Test Results Configuration
**File**: `src/antibody_training_esm/cli/test.py:75`
```python
output_dir: str = "./test_results"
```

**Hierarchical path generation**: `src/antibody_training_esm/core/directory_utils.py`
```python
test_results/{model_shortname}/{classifier_type}/{dataset}/
```

---

## Git Tracking Status

**Gitignored** (`.gitignore:91-93`):
```gitignore
# Outputs
outputs/*
!outputs/.gitkeep
```

**Tracked in Git**:
- `test_results/` ✅ (curated benchmarks)
- `models/` ✅ (trained models - new hierarchical structure)

**Not tracked**:
- `outputs/` ✅ CORRECT (ephemeral Hydra runs)
- `embeddings_cache/` ✅ (cached embeddings)

---

## Answers to Specific Questions

### Q1: Is `outputs/` the canonical place for all outputs?

**Answer**: **NO - Split responsibility:**
- **Training runs** → `outputs/` (Hydra-managed, gitignored)
- **Trained models** → `models/` (hierarchical, tracked in git)
- **Test predictions** → `test_results/` (hierarchical, tracked in git)

### Q2: What is `outputs/test_dataset/`?

**Answer**: Unit test artifact from `tests/unit/datasets/test_base.py`
**Action**: Ignore it (harmless, auto-regenerates)
**Impact**: None

### Q3: Is `test_results/` from the old workflow?

**Answer**: **YES and NO**
- Directory structure: ✅ Current (hierarchical)
- Files inside: From pre-migration runs (Nov 6-12)
- Still valid: ✅ YES (historical benchmarks)
- Needs refresh: Only if we want post-migration benchmarks

### Q4: Where are the 10-fold CV results?

**Answer**: **In logs ONLY** (not saved to files)
- Location: `outputs/novo_replication/2025-11-11_23-22-49/logs/training.log`
- Format: Log lines (not structured YAML/JSON)
- **This is a gap** - should be addressed for reproducibility

### Q5: Do we need final cleanup?

**Answer**: **Minor cleanup + enhancement needed:**

**Cleanup** (optional):
1. Delete stale `outputs/test_dataset/` locally (will regenerate)
2. Archive old `novo_replication/` runs (keep latest only)

**Enhancement** (recommended):
1. Save CV results to structured files (YAML/JSON)
2. Save per-fold predictions for error analysis
3. Update `test_results/` with post-migration benchmarks

---

## Recommendations

### Immediate (Optional Cleanup)

1. **Delete stale outputs** (keep latest runs):
   ```bash
   # Keep only the latest novo_replication run
   cd outputs/novo_replication/
   ls -t | tail -n +2 | xargs rm -rf
   ```

2. **Delete test_dataset** (will regenerate):
   ```bash
   rm -rf outputs/test_dataset/
   ```

### Follow-Up (v0.6.0 - CV Results Enhancement)

**Problem**: CV results not saved to files (only logged)

**Solution**: Enhance `src/antibody_training_esm/core/trainer.py` to save:
```yaml
# Example: outputs/{experiment}/cv_results.yaml
cv_metrics:
  cv_accuracy:
    mean: 0.6413
    std: 0.0972
    folds: [0.65, 0.63, 0.64, ...]  # Per-fold results
  cv_f1:
    mean: 0.6604
    std: 0.0994
    folds: [...]
  cv_roc_auc:
    mean: nan
    std: nan
    folds: [...]

# Plus per-fold predictions for error analysis
fold_predictions/
  fold_0.csv
  fold_1.csv
  ...
```

**Benefits**:
- Reproducible paper results
- Error analysis across folds
- Easy comparison of hyperparameter sweeps

---

## Conclusion

**Directory Organization**: ✅ **95% CORRECT**

- **`outputs/`** - Hydra training scratch ✅ CORRECT
- **`test_results/`** - Curated benchmarks ✅ CORRECT
- **`models/`** - Hierarchical model storage ✅ CORRECT (Phase 2 migration)

**Remaining 5%**:
- ⚠️ `outputs/test_dataset/` - Harmless test artifact (ignore)
- ⚠️ CV results not saved to files - Minor reproducibility gap (enhancement opportunity)

**Verdict**: No blocking issues. System is production-ready. CV file saving is a "nice-to-have" for v0.6.0.

---

**Investigation Team**: Claude Code
**Review Status**: Awaiting Senior Approval
**Next Steps**:
1. Senior review of findings
2. Decide: Archive old outputs or keep for historical reference?
3. Decide: Implement CV file saving now or defer to v0.6.0?
