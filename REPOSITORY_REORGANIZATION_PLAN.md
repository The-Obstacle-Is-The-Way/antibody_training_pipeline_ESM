# Repository Reorganization Plan - Phase 5

**Date**: 2025-11-15
**Status**: 📋 **PENDING SENIOR APPROVAL**
**Prerequisites**: Phase 1-4 cleanup committed
**Purpose**: Consolidate outputs and align with professional ML research repository standards

---

## Executive Summary

**Goal**: Transform repository structure from **inherited legacy patterns** to **professional ML research organization** matching Google DeepMind / Meta AI / OpenAI standards.

**Scope**: Reorganize output directories, consolidate scattered artifacts, and establish single source of truth for experimental results.

**Risk Level**: 🟡 **MEDIUM** - Requires code path updates but no logic changes

**Validation**: Full test suite (374 tests), training smoke test, Novo parity benchmark reproduction

---

## Context & Motivation

### Current Problems

From `REPOSITORY_STRUCTURE_ANALYSIS.md`:

1. **Scattered Outputs**: 5 root-level directories (`outputs/`, `models/`, `embeddings_cache/`, `logs/`, `test_results/`)
2. **Empty Redundant Directories**: `test_results/` contains only `.gitkeep`
3. **Competing Patterns**: `experiments/novo_parity/results/` vs `test_results/` for benchmarks
4. **Inconsistent Gitignore**: Some outputs versioned, others gitignored

### Professional Pattern

Single `experiments/` directory with clear hierarchy:
```
experiments/
├── runs/         # Hydra outputs (gitignored)
├── checkpoints/  # Models (gitignored or LFS)
├── cache/        # Embeddings, preprocessing (gitignored)
└── benchmarks/   # Published results (versioned)
```

**Principles**:
- **Single source of truth** for all outputs
- **Clear ephemeral vs published separation**
- **Consistent gitignore patterns**
- **Scalable organization** (easy to add new experiments)

---

## Phase 5: Directory Reorganization

### Overview

Reorganize repository from **current state** (post-Phase 1-4) to **professional state**:

**Current**:
```
antibody_training_pipeline_ESM/
├── outputs/                  # Hydra runs (gitignored)
├── models/                   # Trained models (versioned, 56KB)
├── embeddings_cache/         # ESM embeddings (gitignored, 4.5MB)
├── logs/                     # Training logs (versioned)
├── test_results/             # Empty (only .gitkeep)
└── experiments/              # Mixed purposes
    ├── archive/
    ├── hyperparameter_sweeps/  # Empty
    ├── novo_parity/
    └── strict_qc_2025-11-04/
```

**Target**:
```
antibody_training_pipeline_ESM/
└── experiments/              # SINGLE source of truth
    ├── runs/                 # Hydra outputs (gitignored)
    ├── checkpoints/          # Trained models (gitignored or LFS)
    ├── cache/                # Embeddings, preprocessing (gitignored)
    └── benchmarks/           # Published results (versioned)
        ├── novo_parity/
        ├── strict_qc/
        └── archive/
```

---

## Implementation Plan

### **Step 1: Create New Directory Structure** (Low Risk)

```bash
# Create new subdirectories under experiments/
mkdir -p experiments/runs
mkdir -p experiments/checkpoints
mkdir -p experiments/cache
mkdir -p experiments/benchmarks

# Create .gitkeep files to preserve empty directories
touch experiments/runs/.gitkeep
touch experiments/checkpoints/.gitkeep
touch experiments/cache/.gitkeep

# Create README for each directory
cat > experiments/runs/README.md << 'EOF'
# Training Runs

**Purpose**: Ephemeral Hydra training outputs (gitignored)

**Contents**:
- Timestamped run directories from `antibody-train`
- Logs, configs, model checkpoints (temporary)
- Cross-validation results (cv_results.yaml)

**Cleanup**: Safe to delete old runs after validation
EOF

cat > experiments/checkpoints/README.md << 'EOF'
# Model Checkpoints

**Purpose**: Trained model artifacts (gitignored or Git LFS)

**Contents**:
- Model checkpoints (.pkl, .npz, .pt)
- Model configurations (.json, .yaml)
- Training metadata

**Versioning**: Use Git LFS for large models (>10MB)
EOF

cat > experiments/cache/README.md << 'EOF'
# Intermediate Cache

**Purpose**: Cached intermediate artifacts (gitignored)

**Contents**:
- ESM embedding caches (.pkl, .npy)
- Preprocessed datasets
- Feature extractions

**Cleanup**: Safe to delete and regenerate
EOF

cat > experiments/benchmarks/README.md << 'EOF'
# Published Benchmarks

**Purpose**: Curated experimental results (versioned in Git)

**Contents**:
- Novo parity replication (66.28% accuracy)
- Strict QC dataset variant
- Ablation studies
- Historical baselines (archive/)

**Versioning**: All files tracked in Git for reproducibility
EOF
```

**Validation**:
```bash
tree experiments -L 2
# Should show: runs/, checkpoints/, cache/, benchmarks/ with READMEs
```

---

### **Step 2: Consolidate Output Artifacts** (Medium Risk)

#### **2.1: Move Hydra Outputs**

```bash
# Move outputs/ to experiments/runs/
mv outputs/* experiments/runs/ 2>/dev/null || echo "No runs to move"
rmdir outputs  # Should be empty now
```

**Affected Files**: None (gitignored)

#### **2.2: Move Model Checkpoints**

```bash
# Move models/ to experiments/checkpoints/
mv models/* experiments/checkpoints/
rmdir models  # Should be empty now
```

**Affected Files**:
- `src/antibody_training_esm/core/config.py` - `DEFAULT_MODEL_DIR`
- `configs/config.yaml` - `training.model_save_dir`
- Any hardcoded `models/` paths in scripts

**Code Update**:
```python
# src/antibody_training_esm/core/config.py
- DEFAULT_MODEL_DIR = "models"
+ DEFAULT_MODEL_DIR = "experiments/checkpoints"
```

```yaml
# configs/config.yaml
training:
-  model_save_dir: models
+  model_save_dir: experiments/checkpoints
```

#### **2.3: Move Embeddings Cache**

```bash
# Move embeddings_cache/ to experiments/cache/
mv embeddings_cache/* experiments/cache/
rmdir embeddings_cache  # Should be empty now
```

**Affected Files**:
- `src/antibody_training_esm/core/embeddings.py` - Cache directory logic

**Code Update**:
```python
# src/antibody_training_esm/core/embeddings.py (approximate location)
- cache_dir = Path("embeddings_cache")
+ cache_dir = Path("experiments/cache")
```

#### **2.4: Move Logs**

```bash
# Move logs/ to experiments/benchmarks/logs/ (since they're versioned)
mkdir -p experiments/benchmarks/logs
mv logs/* experiments/benchmarks/logs/
rmdir logs  # Should be empty now
```

**Affected Files**: None (logs are output-only)

#### **2.5: Delete Empty test_results/**

```bash
# Delete empty test_results/ directory
rm -rf test_results/
```

**Rationale**: Directory is empty (only `.gitkeep` from Phase 2) and serves no purpose. Test results now go to `experiments/benchmarks/{experiment}/`.

**Affected Files**: None (directory was unused)

#### **2.6: Reorganize experiments/**

```bash
# Move existing experiments to benchmarks/
mv experiments/novo_parity experiments/benchmarks/
mv experiments/strict_qc_2025-11-04 experiments/benchmarks/strict_qc
mv experiments/archive experiments/benchmarks/archive

# Delete empty hyperparameter_sweeps/ (already archived)
rmdir experiments/hyperparameter_sweeps
```

**Result**:
```
experiments/
├── runs/              # New (from outputs/)
├── checkpoints/       # New (from models/)
├── cache/             # New (from embeddings_cache/)
└── benchmarks/        # Reorganized
    ├── novo_parity/
    ├── strict_qc/
    └── archive/
```

---

### **Step 3: Update .gitignore** (Low Risk)

**Current .gitignore**:
```
outputs/*
models/scratch/
models/ginkgo_*/
embeddings_cache/*
```

**New .gitignore**:
```
# Ephemeral outputs under experiments/
experiments/runs/*
experiments/checkpoints/*
experiments/cache/*

# Keep directory structure
!experiments/runs/.gitkeep
!experiments/checkpoints/.gitkeep
!experiments/cache/.gitkeep
!experiments/runs/README.md
!experiments/checkpoints/README.md
!experiments/cache/README.md

# Benchmarks are versioned
!experiments/benchmarks/
```

**Implementation**:
```bash
# Update .gitignore
cat >> .gitignore << 'EOF'

# Phase 5 Reorganization: Consolidated experiments/ structure
experiments/runs/*
experiments/checkpoints/*
experiments/cache/*

# Keep directory structure
!experiments/runs/.gitkeep
!experiments/checkpoints/.gitkeep
!experiments/cache/.gitkeep
!experiments/runs/README.md
!experiments/checkpoints/README.md
!experiments/cache/README.md

# Benchmarks are versioned
!experiments/benchmarks/
EOF

# Remove old patterns
sed -i.bak '/^outputs\/\*$/d' .gitignore
sed -i.bak '/^models\/scratch\/$/d' .gitignore
sed -i.bak '/^models\/ginkgo_\*\/$/d' .gitignore
sed -i.bak '/^embeddings_cache\/\*$/d' .gitignore
rm .gitignore.bak
```

---

### **Step 4: Update Code References** (High Risk)

#### **Files to Update**:

1. **src/antibody_training_esm/core/config.py**
   ```python
   - DEFAULT_MODEL_DIR = "models"
   + DEFAULT_MODEL_DIR = "experiments/checkpoints"

   - DEFAULT_CACHE_DIR = "embeddings_cache"
   + DEFAULT_CACHE_DIR = "experiments/cache"
   ```

2. **configs/config.yaml**
   ```yaml
   training:
   -  model_save_dir: models
   +  model_save_dir: experiments/checkpoints
   ```

3. **src/antibody_training_esm/core/embeddings.py**
   - Search for `embeddings_cache` or `cache_dir` references
   - Update to `experiments/cache`

4. **src/antibody_training_esm/core/trainer.py**
   - Check for hardcoded `models/` paths (should use config)

5. **CLAUDE.md**
   - Update directory structure documentation
   - Update example commands

6. **docs/developer-guide/directory-organization.md**
   - Rewrite with new structure

#### **Search for Hardcoded Paths**:

```bash
# Find all hardcoded references to old directories
grep -r "models/" src/ --include="*.py" | grep -v "__pycache__"
grep -r "embeddings_cache" src/ --include="*.py" | grep -v "__pycache__"
grep -r "outputs/" src/ --include="*.py" | grep -v "__pycache__"
grep -r "test_results" src/ --include="*.py" | grep -v "__pycache__"
```

**Action**: Review each result and update to use `experiments/{subdir}/` or config variables.

---

### **Step 5: Update Documentation** (Low Risk)

#### **Files to Update**:

1. **CLAUDE.md** - Directory structure section
2. **docs/developer-guide/directory-organization.md** - Complete rewrite
3. **docs/overview.md** - Update architecture diagram
4. **README.md** - Update quickstart paths

#### **Example Update (CLAUDE.md)**:

```markdown
### Directory Structure
```
experiments/                     # SINGLE source of truth for ALL outputs
├── runs/                       # Hydra training runs (gitignored)
│   └── {exp_name}/{timestamp}/
├── checkpoints/                # Trained models (gitignored or LFS)
│   └── {model}/{classifier}/
├── cache/                      # Embeddings, intermediate artifacts (gitignored)
└── benchmarks/                 # Published results (versioned)
    ├── novo_parity/            # Main Novo replication
    ├── strict_qc/              # Strict QC variant
    └── archive/                # Historical results
```
```

---

### **Step 6: Validation** (Critical)

#### **6.1: Test Suite**

```bash
# Run full test suite
uv run pytest tests/unit/ -v

# Expected: 374/374 passing
```

**If failures**: Check for hardcoded paths in test fixtures.

#### **6.2: Training Smoke Test**

```bash
# Run quick training test
uv run antibody-train \
  training.n_splits=2 \
  training.batch_size=4 \
  hardware.device=cpu \
  experiment.name=phase5_validation

# Check outputs
ls experiments/runs/phase5_validation/
# Should see: .hydra/, cv_results.yaml, trainer.log, model files
```

#### **6.3: Model Loading Test**

```bash
# Verify models load from new location
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --dataset jain

# Expected: 66.28% accuracy (Novo parity)
```

#### **6.4: Cache Validation**

```bash
# Check embeddings cache location
ls experiments/cache/
# Should see: train_0e097a6a2cea_embeddings.pkl (moved from embeddings_cache/)
```

---

## Rollback Plan

**If validation fails**:

```bash
# Restore from git
git checkout -- .

# Or manual rollback
mv experiments/runs/* outputs/
mv experiments/checkpoints/* models/
mv experiments/cache/* embeddings_cache/
mv experiments/benchmarks/novo_parity experiments/
mv experiments/benchmarks/strict_qc experiments/strict_qc_2025-11-04
mv experiments/benchmarks/archive experiments/

# Restore old .gitignore
git checkout .gitignore
```

---

## Before & After Comparison

### **Before (Current State)**

```
antibody_training_pipeline_ESM/
├── outputs/                     # Hydra runs
├── models/                      # Trained models (56KB, versioned)
├── embeddings_cache/            # ESM cache (4.5MB, gitignored)
├── logs/                        # Training logs (versioned)
├── test_results/                # Empty (.gitkeep only)
└── experiments/
    ├── archive/                 # Old benchmarks
    ├── hyperparameter_sweeps/   # Empty
    ├── novo_parity/             # Active experiment
    └── strict_qc_2025-11-04/    # Alternative dataset
```

**Issues**:
- 5 root-level output directories
- Inconsistent gitignore (models/ versioned, outputs/ not)
- Empty redundant directories
- Unclear where test results go

### **After (Professional State)**

```
antibody_training_pipeline_ESM/
└── experiments/                 # SINGLE source of truth
    ├── runs/                    # Hydra outputs (gitignored)
    ├── checkpoints/             # Models (gitignored or LFS)
    ├── cache/                   # Embeddings (gitignored)
    └── benchmarks/              # Published results (versioned)
        ├── novo_parity/
        ├── strict_qc/
        ├── logs/                # Historical logs
        └── archive/
```

**Benefits**:
- ✅ Single source of truth for all outputs
- ✅ Clear ephemeral vs published separation
- ✅ Consistent gitignore patterns
- ✅ Scalable organization
- ✅ Matches professional ML research standards

---

## Risk Assessment

| Step | Risk Level | Mitigation |
|------|-----------|------------|
| **1. Create directories** | 🟢 LOW | Read-only, no breakage |
| **2. Move artifacts** | 🟡 MEDIUM | Update config paths, test thoroughly |
| **3. Update .gitignore** | 🟢 LOW | Non-breaking |
| **4. Update code paths** | 🔴 HIGH | Comprehensive grep search, full test suite |
| **5. Update docs** | 🟢 LOW | Non-breaking |
| **6. Validation** | 🟡 MEDIUM | Full test suite + training smoke test |

**Overall Risk**: 🟡 **MEDIUM** (requires code changes but no logic changes)

---

## Estimated Effort

- **Step 1 (Create structure)**: 15 min
- **Step 2 (Move artifacts)**: 30 min
- **Step 3 (Update .gitignore)**: 10 min
- **Step 4 (Update code)**: 1-2 hours (search, update, test)
- **Step 5 (Update docs)**: 30 min
- **Step 6 (Validation)**: 30 min

**Total**: ~3-4 hours

---

## Success Criteria

1. ✅ All tests passing (374/374)
2. ✅ Training smoke test succeeds
3. ✅ Model loading from new paths works
4. ✅ Embeddings cache in correct location
5. ✅ Novo parity benchmark reproduces (66.28% accuracy)
6. ✅ No hardcoded old paths in codebase
7. ✅ Clean `git status` (no unintended gitignore changes)

---

## Timeline

**Prerequisites**:
1. Phase 1-4 cleanup committed ✅
2. Senior approval of this plan 📋 PENDING

**Execution**:
1. **Day 1**: Steps 1-3 (create structure, move artifacts, update .gitignore)
2. **Day 2**: Step 4 (update code references - most time-consuming)
3. **Day 3**: Steps 5-6 (update docs, validation)

**Commit**: Single commit after full validation

---

## Open Questions for Senior Review

1. **Model Versioning**: Should models be Git LFS or fully gitignored?
   - Current: 56KB models versioned in git
   - Professional: Usually gitignored or LFS (>10MB threshold)

2. **Logs Placement**: Should historical logs go in `experiments/benchmarks/logs/` or be deleted?
   - Current plan: Move to benchmarks (they're versioned)
   - Alternative: Delete if not needed for reproducibility

3. **strict_qc Experiment**: Rename to `strict_qc` or keep `strict_qc_2025-11-04`?
   - Current plan: Rename to `strict_qc` for cleaner naming
   - Alternative: Keep timestamp for historical context

---

## Appendix: Professional Repository Examples

### **AlphaFold (DeepMind)**
```
alphafold/
├── data/
├── alphafold/          # Core package
├── scripts/
├── docker/
└── run_alphafold.py
```

**Outputs**: Not in repo (generated at runtime)

### **ESM (Meta AI)**
```
esm/
├── data/
├── esm/                # Core package
├── examples/
├── scripts/
└── setup.py
```

**Outputs**: Not in repo (users download pretrained models)

### **CLIP (OpenAI)**
```
CLIP/
├── clip/               # Core package
├── notebooks/
└── setup.py
```

**Outputs**: Not in repo (models downloaded from HuggingFace)

**Common Pattern**: Clean root, minimal top-level directories, outputs not committed (or in single `experiments/` dir).

---

**Document Status**: 📋 **READY FOR SENIOR REVIEW**
**Next Step**: Senior approval → Execute Phase 5
