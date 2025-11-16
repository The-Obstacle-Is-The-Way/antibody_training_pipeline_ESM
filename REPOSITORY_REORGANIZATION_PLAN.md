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

## ⚠️ **CRITICAL: AGENTS.md Policy Conflict**

**WARNING**: This plan conflicts with repository house style defined in AGENTS.md.

**AGENTS.md:4** states:
> "Checkpoints and logs belong in `models/`, `logs/`, `outputs/`"

**Phase 5 Plan**: Moves models → experiments/checkpoints/, logs → experiments/runs/logs/ (ephemeral), embeddings_cache → experiments/cache/

**Resolution Required BEFORE Execution**:
1. **Option A**: Update AGENTS.md to align with Phase 5 (move to experiments/ hierarchy)
2. **Option B**: Revise Phase 5 to keep models/, logs/, outputs/ at root (align with current house style)

**Recommendation**: Seek senior decision on repository organization philosophy.

---

## Context & Motivation

### Current Problems

From `REPOSITORY_STRUCTURE_ANALYSIS.md`:

1. **Scattered Outputs**: 4 root-level directories (`outputs/`, `models/`, `embeddings_cache/`, `logs/`)
2. **Stale Code References**: CLI defaults still point to nonexistent `test_results/` directory (will recreate on first run)
3. **AGENTS.md Policy Conflict**: House style says "logs belong in logs/, models in models/" but Phase 5 moves them to experiments/
4. **Inconsistent Gitignore**: Some outputs versioned (models/), others gitignored (outputs/, embeddings_cache/, logs/)

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
├── logs/                     # Build/test logs (gitignored per .gitignore:57, ~180KB ephemeral)
└── experiments/              # Mixed purposes
    ├── archive/
    │   ├── hyperparameter_sweeps_2025-11-02/  # Archived sweeps
    │   └── test_results_pre_migration_2025-11-06/  # Archived test results
    ├── hyperparameter_sweeps/  # Contains .gitkeep only
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

**Affected Files (Config Files Only - No Python Constant Exists)**:
- `src/antibody_training_esm/conf/config_schema.py:84` - `model_save_dir: str = "./models"`
- `src/antibody_training_esm/conf/config.yaml:21` - `model_save_dir: ./models`
- `configs/config.yaml:62` - `model_save_dir: "./models"` (legacy root config, may be deleted by v0.5.0)

**Code Updates**:
```python
# src/antibody_training_esm/conf/config_schema.py
- model_save_dir: str = "./models"
+ model_save_dir: str = "./experiments/checkpoints"
```

```yaml
# src/antibody_training_esm/conf/config.yaml
training:
-  model_save_dir: ./models
+  model_save_dir: ./experiments/checkpoints

# configs/config.yaml (if still exists after v0.5.0 cleanup)
training:
-  model_save_dir: "./models"
+  model_save_dir: "./experiments/checkpoints"
```

**NOTE**: There is NO `DEFAULT_MODEL_DIR` constant in `src/antibody_training_esm/core/config.py`. Model paths are configured via config files only.

#### **2.3: Move Embeddings Cache**

```bash
# Move embeddings_cache/ to experiments/cache/
mv embeddings_cache/* experiments/cache/
rmdir embeddings_cache  # Should be empty now
```

**Affected Files (Config Files, NOT embeddings.py)**:
- `src/antibody_training_esm/conf/config_schema.py:53` - `embeddings_cache_dir: str = "./embeddings_cache"`
- `src/antibody_training_esm/conf/data/boughter_jain.yaml:16` - `embeddings_cache_dir: ./embeddings_cache`
- `configs/config.yaml:33` - `embeddings_cache_dir: "./embeddings_cache"`

**Code Updates**:
```python
# src/antibody_training_esm/conf/config_schema.py
- embeddings_cache_dir: str = "./embeddings_cache"
+ embeddings_cache_dir: str = "./experiments/cache"
```

```yaml
# src/antibody_training_esm/conf/data/boughter_jain.yaml
- embeddings_cache_dir: ./embeddings_cache
+ embeddings_cache_dir: ./experiments/cache

# configs/config.yaml
- embeddings_cache_dir: "./embeddings_cache"
+ embeddings_cache_dir: "./experiments/cache"
```

**NOTE**: `src/antibody_training_esm/core/embeddings.py` does NOT contain hardcoded cache paths - it reads dynamically from config via trainer.py:797.

#### **2.4: Move Logs** ⚠️ **BLOCKED - REQUIRES DECISION**

**⚠️ CRITICAL DISCOVERY**: logs/ is **NOT** "no code references" - it's hardcoded in 3 active configs + trainer.py!

**Active References**:
1. `src/antibody_training_esm/conf/config_schema.py:88` - `log_file: str = "logs/training.log"`
2. `src/antibody_training_esm/conf/config.yaml:25` - `log_file: logs/training.log`
3. `configs/config.yaml:66` - `log_file: "./logs/boughter_training.log"`
4. `src/antibody_training_esm/core/trainer.py:183` - `log_file = Path.cwd() / "logs" / log_file_str` (legacy mode)

**Current Behavior**:
- **Hydra mode** (antibody-train CLI): Logs go to `outputs/{experiment}/{timestamp}/` (Hydra's output dir, ignores logs/)
- **Legacy mode** (train_model() function): Logs go to `logs/` at root (will be auto-created if missing via trainer.py:185)

**Problem**: If we delete logs/ without updating configs:
- Legacy mode will recreate `logs/` at root (defeats reorganization)
- Or training will fail if mkdir fails

**⚠️ DECISION REQUIRED - Three Options**:

**Option A: Keep logs/ at root (RECOMMENDED)**
- ✅ Aligns with AGENTS.md:4 policy ("logs belong in logs/")
- ✅ No config changes needed
- ✅ Works for both Hydra and legacy mode
- ✅ Simple, no migration needed
- ⚠️ Doesn't consolidate outputs (but logs are ephemeral scratch space anyway)

**Action**:
```bash
# No action - keep logs/ at root
# Only clean out stale logs if desired:
# find logs/ -type f -mtime +30 -delete
```

**Affected Files**: None

---

**Option B: Move logs/ + Update all configs**
- ✅ Consolidates all ephemeral outputs under experiments/
- ❌ Requires updating 3 config files
- ❌ Breaks any external scripts using `logs/` path
- ⚠️ Complex migration

**Action**:
```bash
# 1. Update configs first
sed -i '' 's|logs/training.log|experiments/runs/logs/training.log|g' \
  src/antibody_training_esm/conf/config_schema.py \
  src/antibody_training_esm/conf/config.yaml \
  configs/config.yaml

# 2. Move logs
mkdir -p experiments/runs/logs
mv logs/* experiments/runs/logs/ 2>/dev/null || true
rmdir logs
```

**Affected Files**:
- `src/antibody_training_esm/conf/config_schema.py:88`
- `src/antibody_training_esm/conf/config.yaml:25`
- `configs/config.yaml:66`

---

**Option C: Defer until v0.5.0 removes legacy mode**
- ✅ Cleaner after legacy train_model() removed
- ✅ Hydra mode doesn't use logs/ anyway
- ⚠️ Delays consolidation

**Action**:
```bash
# No action - revisit after V0.5.0_CLEANUP_PLAN.md execution
```

---

**RECOMMENDATION**: **Option A** - Keep logs/ at root per AGENTS.md policy. Rationale:
1. logs/ is ephemeral scratch space (all files gitignored)
2. Legacy mode will recreate it anyway if deleted
3. Hydra mode ignores it (uses outputs/ instead)
4. AGENTS.md:4 explicitly says "logs belong in logs/"
5. No breaking changes, no config migrations needed

**NOTE**: logs/* is completely gitignored per `.gitignore:57`. Current contents are build/test logs (~180KB, ephemeral).

#### **2.5: Reorganize experiments/**

```bash
# Move existing experiments to benchmarks/
mv experiments/novo_parity experiments/benchmarks/
mv experiments/strict_qc_2025-11-04 experiments/benchmarks/strict_qc
mv experiments/archive experiments/benchmarks/archive

# Remove hyperparameter_sweeps/ (contains only .gitkeep, past sweeps archived)
git rm experiments/hyperparameter_sweeps/.gitkeep
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

### **Step 4: Update Code References** (High Risk - 34+ Files)

⚠️ **CRITICAL**: This step is NOT optional. Skipping any file will cause breakage.

#### **4.1: Update test_results/ References (7 code + 12 docs)**

**Code Files (MUST UPDATE)**:
1. `src/antibody_training_esm/cli/test.py:75` - Change `output_dir: str = "./test_results"` → `"./experiments/benchmarks"`
2. `src/antibody_training_esm/cli/test.py:686` - Update example config
3. `src/antibody_training_esm/cli/test.py:727` - Change argparse `default="./test_results"` → `"experiments/benchmarks"`
4. `src/antibody_training_esm/core/directory_utils.py:6` - Update docstring
5. `src/antibody_training_esm/core/directory_utils.py:111-143` - Update `get_hierarchical_test_results_dir()` function (consider renaming parameter)
6. `src/antibody_training_esm/core/directory_utils.py:123` - Update param doc
7. `src/antibody_training_esm/core/directory_utils.py:133-138` - Update example

**Documentation Files (MUST UPDATE)**:
- README.md
- AGENTS.md
- CLAUDE.md
- ROADMAP.md
- USAGE.md
- docs/user-guide/testing.md
- docs/user-guide/getting-started.md
- docs/developer-guide/directory-organization.md
- docs/developer-guide/development-workflow.md
- experiments/strict_qc_2025-11-04/EXPERIMENT_README.md
- REPOSITORY_STRUCTURE_ANALYSIS.md
- PROBLEMS.md

**🔴 CRITICAL TEST FILES (MISSING FROM ORIGINAL PLAN)**:
1. `tests/unit/cli/test_test.py:568` - `assert call_args.output_dir == "./test_results"`
2. `tests/unit/cli/test_model_tester.py:45` - `output_dir=str(tmp_path / "test_results")` (TestConfig fixture)
3. `tests/integration/test_model_tester.py:94,563,626,636` - All use `tmp_path / "test_results"`
4. `tests/unit/core/test_directory_utils.py:164,176,188,200` - 4 assertions expecting `Path("test_results/...")`

**Impact if Missed**: ALL CLI and directory utils tests will FAIL when test_results defaults change.

#### **4.2: Update outputs/ References (5 code + 3 tests + 1 script)**

**Code Files (MUST UPDATE)**:
1. `src/antibody_training_esm/datasets/base.py:80` - Change `Path(f"outputs/{dataset_name}")` → `Path(f"experiments/runs/{dataset_name}")`
2. `src/antibody_training_esm/conf/hydra/default.yaml:3` - Change `dir: outputs/${experiment.name}` → `experiments/runs/${experiment.name}`
3. `src/antibody_training_esm/conf/hydra/default.yaml:6` - Change `dir: outputs/sweeps/` → `experiments/runs/sweeps/`

**Test Files (MUST UPDATE)**:
4. `tests/unit/datasets/test_base.py:84-87` - Update assertion OR verify autouse fixture handles it
5. `tests/unit/datasets/conftest.py:24` - Update comment
6. `tests/unit/core/test_trainer_hydra.py:117` - Change `tmp_path / "outputs"` → `tmp_path / "experiments/runs"`
7. `tests/unit/core/test_trainer.py:66,888-890,1309` - Embeddings cache and model hierarchical path tests (may need updates if cache/model paths change)

**Script Files (MUST UPDATE)**:
7. `scripts/migrate_train_datasets_to_data_train.sh:68,96` - Update filter pattern

#### **4.3: Update embeddings_cache/ References (3 configs)**

**Config Files (MUST UPDATE)**:
1. `configs/config.yaml:33` - Change `embeddings_cache_dir: "./embeddings_cache"` → `"experiments/cache"`
2. `src/antibody_training_esm/conf/data/boughter_jain.yaml:16` - Change `embeddings_cache_dir: ./embeddings_cache` → `experiments/cache`
3. `src/antibody_training_esm/conf/config_schema.py:53` - Change `embeddings_cache_dir: str = "./embeddings_cache"` → `"experiments/cache"`

#### **4.4: Update models/ References (7 docstrings)**

**Docstring Files (SHOULD UPDATE for consistency)**:
1. `src/antibody_training_esm/core/trainer.py:594` - Update example path
2. `src/antibody_training_esm/core/trainer.py:604-606` - Update comment examples
3. `src/antibody_training_esm/cli/test.py:11` - Update usage docstring
4. `src/antibody_training_esm/cli/test.py:682` - Update example config
5. `src/antibody_training_esm/cli/test.py:705,708` - Update usage examples
6. `src/antibody_training_esm/core/directory_utils.py:5` - Update docstring
7. `src/antibody_training_esm/core/directory_utils.py:103` - Update example

#### **4.5: Update .gitignore and Helper Scripts**

**Files (MUST UPDATE)**:
1. `.gitignore` - Already updated in Step 3
2. Documentation referencing old cleanup commands (e.g., "rm -rf embeddings_cache/")

#### **Verification Commands**:

After updates, run these to confirm NO remaining references:
```bash
# Should return ZERO results (except comments/docs):
rg "test_results" src/ --type py | grep -v "#"
rg 'Path.*"outputs/' src/ --type py | grep -v "#"
rg "embeddings_cache" src/antibody_training_esm/conf/ --type yaml

# These are OK to have (docstrings):
rg "models/" src/ --type py | grep ">>>"  # Example outputs
```

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
├── outputs/                     # Hydra runs (gitignored)
├── models/                      # Trained models (56KB, versioned)
├── embeddings_cache/            # ESM cache (4.5MB, gitignored)
├── logs/                        # Build/test logs (7 .log files, ~180KB, all gitignored, none tracked)
└── experiments/
    ├── archive/                 # Archived benchmarks
    ├── hyperparameter_sweeps/   # Placeholder (.gitkeep only)
    ├── novo_parity/             # Active experiment
    └── strict_qc_2025-11-04/    # Alternative dataset
```

**Note**: test_results/ doesn't exist at root (archived to experiments/archive/test_results_pre_migration_2025-11-06/)

**Issues**:
- 4 root-level output directories (models/, outputs/, embeddings_cache/, logs/)
- Inconsistent gitignore (models/ versioned, others gitignored)
- CLI defaults point to non-existent test_results/ (will recreate on first run)
- Placeholder directory (hyperparameter_sweeps/ with only .gitkeep)

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

2. **Logs Placement**: Should current logs/ be moved or kept at root?
   - **Option A**: Move to `experiments/runs/logs/` (consolidate all ephemeral outputs)
   - **Option B**: Keep at root per AGENTS.md:4 policy ("logs belong in logs/")
   - **NOTE**: logs/* is completely gitignored (.gitignore:57) - these are build/test logs, NOT versioned artifacts

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
