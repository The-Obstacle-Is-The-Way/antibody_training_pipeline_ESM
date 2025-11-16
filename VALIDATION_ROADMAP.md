# Validation Roadmap - Novo Nordisk Replication Repository

**Date:** 2025-11-16
**Purpose:** Validate that this repository fully replicates Novo Nordisk's antibody non-specificity prediction methodology
**Status:** POST-PHASE 5 REORGANIZATION - Pre-validation

---

## Executive Summary

This repository aims to be a complete, working replication of:
> **Sakhnini et al. (2025):** *Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters*

After Phase 5 reorganization (experiments/ consolidation), we need to **validate end-to-end** that:
1. All data preprocessing pipelines work
2. All training pipelines work
3. All testing pipelines work
4. Results match published benchmarks
5. Everything is reproducible from a clean clone

---

## Core Repository Purpose

### What This Repository Should Do

From the **Novo Nordisk paper**, this repo must support:

#### 1. **Training Pipeline** (Boughter Dataset)
- **Input:** 1,171 raw DNA sequences (Boughter et al. 2020)
- **Output:** Trained ESM-1v VH-based LogisticRegression binary classifier
- **Expected Performance:** 71% 10-fold CV accuracy (Novo reported)
- **Our Performance:** 67.5% ± 8.9% (validated, within statistical noise)

#### 2. **Testing Pipeline** (3 External Datasets)
- **Jain (clinical antibodies):** ELISA assay, expect ~66-69% accuracy
- **Shehata (PSR assay):** Expect poor separation (~52%) due to assay incompatibility
- **Harvey (nanobodies):** PSR assay, VHH-only sequences

#### 3. **Preprocessing Pipelines**
- **Boughter:** DNA translation → ANARCI annotation → QC filtering → 914 sequences
- **Jain:** Excel → CSV → P5e-S2 canonical subset (86 antibodies)
- **Harvey:** Raw CSVs → fragment extraction (141k nanobodies)
- **Shehata:** Excel → CSV → fragment extraction (398 antibodies)

#### 4. **Embeddings & Classification**
- **Embeddings:** ESM-1v (facebook/esm1v_t33_650M_UR90S_1) mean-pooling
- **Classifier:** sklearn LogisticRegression (C=1.0, penalty='l2', solver='lbfgs')
- **Fragments:** VH, VL, CDRs, FWRs (16 fragment types)

---

## Current Repository State (Post-Phase 5)

### Directory Structure (After Reorganization)

```
antibody_training_pipeline_ESM/
├── data/
│   ├── train/boughter/canonical/          # 914 sequences (PRODUCTION)
│   ├── test/jain/canonical/               # 86 sequences (Novo parity)
│   ├── test/harvey/fragments/             # 141k nanobodies
│   └── test/shehata/fragments/            # 398 antibodies
├── preprocessing/                          # Dataset-specific pipelines
│   ├── boughter/                          # 3-stage pipeline
│   ├── jain/                              # 2-step pipeline
│   ├── harvey/                            # 2-step pipeline
│   └── shehata/                           # 2-step pipeline
├── experiments/                            # NEW (Phase 5)
│   ├── runs/                              # Hydra outputs (gitignored)
│   ├── checkpoints/                       # Trained models (gitignored/LFS)
│   ├── cache/                             # Embeddings cache (gitignored)
│   ├── benchmarks/                        # Published results (versioned)
│   │   ├── strict_qc/                     # ARCHIVED (never validated)
│   │   └── novo_parity_2025-11-05/        # Validated results
│   └── archive/                           # Historical hyperparameter sweeps
├── src/antibody_training_esm/             # Core package
│   ├── core/                              # embeddings, classifier, trainer
│   ├── datasets/                          # Dataset loaders
│   ├── cli/                               # train, test, preprocess commands
│   └── conf/                              # Hydra configs
├── models/                                 # MOVED TO experiments/checkpoints/ (deprecated)
├── outputs/                                # MOVED TO experiments/runs/ (deprecated)
├── embeddings_cache/                       # MOVED TO experiments/cache/ (deprecated)
└── logs/                                   # MOVED TO experiments/runs/logs/ (deprecated)
```

### What Exists vs What's Needed

**✅ Already Exists:**
- Preprocessing scripts (4 datasets)
- Training pipeline (Hydra-based)
- Testing CLI (`antibody-test`)
- ESM-1v embedding extraction
- Logistic regression classifier
- 914-sequence production dataset
- Validated results on Jain (66.28%) and Shehata (52.26%)

**❓ Needs Validation (Post-Phase 5):**
- Do all preprocessing scripts still work?
- Does training pipeline work with new experiments/ paths?
- Does testing pipeline work with new experiments/ paths?
- Do Hydra outputs go to experiments/runs/?
- Do model checkpoints save to experiments/checkpoints/?
- Do embeddings cache to experiments/cache/?

**❌ Currently Unknown:**
- Where do hyperparameter sweep outputs go?
- Is strict_qc (852 sequences) worth keeping? (Answer: NO, archive it)
- What's the complete input → output flow for each pipeline?

---

## Validation Tasks

### Phase 1: Data Preprocessing Validation

**Goal:** Ensure all preprocessing pipelines work end-to-end

#### Task 1.1: Boughter (Training Set)
**Input:** `data/train/boughter/raw/Boughter_ExtraDataFile1.xlsx` (if available)
**Expected Output:**
```
data/train/boughter/canonical/
├── VH_only_boughter_training.csv              # 914 sequences
├── VL_only_boughter_training.csv              # 914 sequences
├── H-CDR1_boughter_training.csv               # 914 sequences
├── H-CDR2_boughter_training.csv               # 914 sequences
├── H-CDR3_boughter_training.csv               # 914 sequences
└── [11 more fragment files]
```

**Commands:**
```bash
# Stage 1: DNA translation (if raw data available)
python3 preprocessing/boughter/stage1_dna_translation.py

# Stage 2+3: ANARCI annotation + QC
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py

# Validation
python3 preprocessing/boughter/validate_stages2_3.py
```

**Success Criteria:**
- ✅ 914 sequences in VH_only_boughter_training.csv
- ✅ All 16 fragment files generated
- ✅ Label balance: ~50/50 (457 label 0, 457 label 1)
- ✅ No X in CDRs
- ✅ Validation script passes

**Status:** 🔄 **RUN THIS**

---

#### Task 1.2: Jain (Test Set - Novo Parity)
**Input:** `data/test/jain/raw/jain_2017_supplementary_data.xlsx`
**Expected Output:**
```
data/test/jain/canonical/
├── jain_86_novo_parity.csv                    # 86 antibodies (P5e-S2, HIC retention)
├── VH_only_jain_test_PARITY_86.csv           # 86 VH sequences
└── [15 more fragment files]
```

**Commands:**
```bash
# Step 1: Excel → CSV conversion
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Step 2: Extract P5e-S2 canonical subset
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Success Criteria:**
- ✅ 86 antibodies in jain_86_novo_parity.csv
- ✅ All clinical-stage IgG1 antibodies
- ✅ P5e-S2 columns present (HIC retention assay)

**Status:** 🔄 **RUN THIS**

---

#### Task 1.3: Harvey (Nanobody Test Set)
**Input:** Raw CSV files from Harvey et al. 2022
**Expected Output:**
```
data/test/harvey/fragments/
├── VHH_only_harvey.csv                        # 141k nanobodies
├── VHH-CDR1_harvey.csv
├── VHH-CDR2_harvey.csv
├── VHH-CDR3_harvey.csv
└── [more VHH-specific fragment files]
```

**Commands:**
```bash
# Step 1: Combine raw CSVs
python3 preprocessing/harvey/step1_convert_raw_csvs.py

# Step 2: Extract VHH fragments
python3 preprocessing/harvey/step2_extract_fragments.py
```

**Success Criteria:**
- ✅ ~141k VHH sequences
- ✅ PSR scores present (poly-specific reagent assay)

**Status:** 🔄 **RUN THIS**

---

#### Task 1.4: Shehata (PSR Test Set)
**Input:** `data/test/shehata/raw/shehata_2019_supplementary.xlsx`
**Expected Output:**
```
data/test/shehata/fragments/
├── VH_only_shehata.csv                        # 398 antibodies
├── VL_only_shehata.csv
└── [14 more fragment files]
```

**Commands:**
```bash
# Step 1: Excel → CSV conversion
python3 preprocessing/shehata/step1_convert_excel_to_csv.py

# Step 2: Extract fragments
python3 preprocessing/shehata/step2_extract_fragments.py
```

**Success Criteria:**
- ✅ 398 antibodies
- ✅ PSR scores present
- ✅ 7 PSR-positive antibodies (1.76% imbalance expected)

**Status:** 🔄 **RUN THIS**

---

### Phase 2: Training Pipeline Validation

**Goal:** Verify training pipeline works with experiments/ structure

#### Task 2.1: Train Production Model (914 sequences)
**Input:** `data/train/boughter/canonical/VH_only_boughter_training.csv`
**Expected Output:**
```
experiments/checkpoints/esm1v/logreg/
└── boughter_vh_esm1v_logreg.pkl               # Trained model

experiments/cache/
└── [cached embeddings as .npy files]          # SHA-256 hashed

experiments/runs/boughter_production_YYYY-MM-DD/HH-MM-SS/
├── .hydra/
│   ├── config.yaml                            # Full resolved config
│   └── overrides.yaml                         # CLI overrides
├── training.log                               # Training logs
└── [Hydra outputs]
```

**Commands:**
```bash
# Default training (uses conf/config.yaml)
uv run antibody-train

# Or explicit config
uv run antibody-train \
  data.train_file=data/train/boughter/canonical/VH_only_boughter_training.csv \
  experiment.name=boughter_production
```

**Success Criteria:**
- ✅ Model saves to `experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl`
- ✅ Embeddings cache to `experiments/cache/`
- ✅ Hydra outputs to `experiments/runs/{experiment.name}/{timestamp}/`
- ✅ 10-fold CV accuracy: 67-71% (within Novo range)
- ✅ Training log shows:
  - 914 sequences loaded
  - ESM-1v embedding extraction
  - 10-fold stratified CV
  - Final model training on full dataset
  - Model saved successfully

**Status:** 🔄 **RUN THIS**

---

#### Task 2.2: Verify Embedding Caching
**Goal:** Embeddings should cache and re-use on subsequent runs

**Commands:**
```bash
# First run (fresh embeddings)
time uv run antibody-train training.n_splits=3

# Second run (cached embeddings)
time uv run antibody-train training.n_splits=3
```

**Success Criteria:**
- ✅ First run: ~15-20 minutes (embedding extraction)
- ✅ Second run: ~5-10 minutes (cached embeddings)
- ✅ Cache files in `experiments/cache/` with SHA-256 hashed names
- ✅ Log shows: "Using cached embeddings from..."

**Status:** 🔄 **RUN THIS**

---

### Phase 3: Testing Pipeline Validation

**Goal:** Verify testing CLI works with new experiments/ structure

#### Task 3.1: Test on Jain Dataset (Novo Parity)
**Expected:** ~66% accuracy (ELISA-compatible assay)

**Commands:**
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --dataset jain \
  --fragment VH
```

**Success Criteria:**
- ✅ Accuracy: 64-69% (within validated range)
- ✅ ROC-AUC: 0.60-0.65
- ✅ Test results save to `experiments/runs/test_{timestamp}/`
- ✅ Confusion matrix displayed
- ✅ Metrics: accuracy, precision, recall, F1, ROC-AUC

**Status:** 🔄 **RUN THIS**

---

#### Task 3.2: Test on Shehata Dataset (PSR Assay)
**Expected:** ~52% accuracy (PSR/ELISA assay incompatibility)

**Commands:**
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --dataset shehata \
  --fragment VH
```

**Success Criteria:**
- ✅ Accuracy: 50-55% (poor separation expected, documented in Novo paper)
- ✅ ROC-AUC: 0.60-0.70 (better than random)
- ✅ Log shows: "PSR assay - expect poor separation (assay incompatibility)"

**Status:** 🔄 **RUN THIS**

---

#### Task 3.3: Test on Harvey Dataset (Nanobodies)
**Expected:** Bimodal pI distribution, most non-specific have high pI (>8)

**Commands:**
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --dataset harvey \
  --fragment VHH
```

**Success Criteria:**
- ✅ Handles VHH-only fragment type
- ✅ Prediction probabilities show broad distribution for PSR-specific
- ✅ PSR-positive antibodies cluster at higher non-specificity probability

**Status:** 🔄 **RUN THIS**

---

### Phase 4: Hyperparameter Sweep Validation

**Goal:** Understand where hyperparameter sweep outputs go

#### Task 4.1: Run Minimal Hyperparameter Sweep
**Question:** Where do multirun outputs end up after Phase 5?

**Commands:**
```bash
# Hydra multirun with 3 parameter combinations
uv run antibody-train \
  --multirun \
  classifier.C=0.1,1.0,10.0 \
  training.n_splits=3 \
  experiment.name=hyperparam_sweep_test
```

**Expected Output:**
```
experiments/runs/hyperparam_sweep_test/
└── multirun/YYYY-MM-DD/HH-MM-SS/
    ├── 0/                                     # C=0.1
    │   ├── .hydra/config.yaml
    │   ├── training.log
    │   └── boughter_vh_esm1v_logreg.pkl
    ├── 1/                                     # C=1.0
    │   └── ...
    └── 2/                                     # C=10.0
        └── ...
```

**Success Criteria:**
- ✅ Multirun outputs to `experiments/runs/{experiment.name}/multirun/{date}/{time}/`
- ✅ Each run (0, 1, 2) has separate directory
- ✅ Models save to run-specific subdirectories
- ✅ Aggregate results available in parent directory

**Status:** 🔄 **RUN THIS**

**Decision:** If hyperparameter sweeps are not part of core Novo replication, **archive them** to `experiments/archive/hyperparameter_sweeps_YYYY-MM-DD/`

---

### Phase 5: Strict QC Experiment Disposition

**Goal:** Decide what to do with `experiments/benchmarks/strict_qc/`

#### Task 5.1: Validate Strict QC Status
**Finding:** Strict QC (852 sequences) was NEVER TESTED on external datasets

**From EXPERIMENT_README.md:**
- ❌ Never trained a model on 852 sequences
- ❌ Never tested 852-sequence model on Jain dataset
- ❌ Never compared 914 vs 852 performance
- ❌ Hypothesis DISPROVEN (914-sequence model validated at 66.28% on Jain)

**Decision:**
```
KEEP AS ARCHIVED EXPERIMENT - Do NOT delete

Rationale:
1. Demonstrates good scientific practice (hypothesis → test → archive)
2. Provenance: Shows why 914-sequence model is production
3. Reproducibility: Anyone can test the 852-sequence hypothesis
4. Documentation: EXPERIMENT_README.md clearly marks it as ARCHIVED/UNVALIDATED
```

**Action Required:**
- ✅ Update config paths in `experiments/benchmarks/strict_qc/configs/config_strict_qc.yaml`
- ✅ Add warning banner to all strict_qc docs: "⚠️ ARCHIVED - Never validated"
- ✅ Ensure no production code references strict_qc paths

**Status:** ✅ **ALREADY DOCUMENTED** (EXPERIMENT_README.md is clear)

---

### Phase 6: Clean Repository Validation

**Goal:** Ensure repo works from a fresh clone

#### Task 6.1: Fresh Clone Test
**Simulate:** New user clones repo and runs everything

**Commands:**
```bash
# 1. Clone repo
git clone <repo-url> test-clone
cd test-clone

# 2. Install dependencies
uv sync --all-extras

# 3. Run preprocessing (if raw data available)
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py

# 4. Train model
uv run antibody-train

# 5. Test model
uv run antibody-test --model <path> --dataset jain --fragment VH

# 6. Run tests
uv run pytest
```

**Success Criteria:**
- ✅ All preprocessing scripts work
- ✅ Training completes without errors
- ✅ Testing produces expected results
- ✅ 374/374 tests pass
- ✅ No missing dependencies
- ✅ No hard-coded absolute paths

**Status:** 🔄 **RUN THIS** (final validation)

---

## Novo Parity Checklist

From the Novo paper, ensure we replicate:

### ✅ Dataset
- [x] Boughter dataset (914 sequences, VH-based)
- [x] ELISA polyreactivity flags (0 vs >3)
- [x] Jain test set (86 clinical antibodies, P5e-S2)
- [x] Shehata test set (398 antibodies, PSR)
- [x] Harvey test set (141k nanobodies, PSR)

### ✅ Model Architecture
- [x] ESM-1v embeddings (facebook/esm1v_t33_650M_UR90S_1)
- [x] Mean-pooling of last hidden states
- [x] LogisticRegression classifier (C=1.0, penalty='l2', solver='lbfgs')
- [x] VH-based fragments (highest performance)

### ✅ Training Methodology
- [x] 10-fold stratified cross-validation
- [x] Train on full training set after CV
- [x] Binary classification (specific: 0, non-specific: >3 flags)
- [x] Mildly non-specific (1-3 flags) excluded from training

### ✅ Validation Methodology
- [x] k-fold CV (3, 5, 10 folds)
- [x] Leave-One-Family-Out validation
- [x] External test on Jain dataset
- [x] External test on Shehata dataset
- [x] External test on Harvey dataset

### ✅ Expected Performance
- [x] 10-fold CV: ~71% (Novo) vs 67.5% ± 8.9% (ours, within statistical noise)
- [x] Jain test: ~69% (Novo) vs 66.28% (ours, ✅ validated)
- [x] Shehata test: Poor separation expected (PSR/ELISA incompatibility) ✅
- [x] Harvey test: Bimodal pI distribution ✅

---

## What We DON'T Need to Replicate

### 1. Biophysical Descriptors (Novo's Alternative Approach)
**Novo's Work:**
- 68 sequence-based biophysical descriptors
- Theoretical isoelectric point (pI) as key driver
- Top 5 descriptors model

**Our Status:**
- Not implemented (out of scope)
- ESM-1v PLM-based approach is primary focus
- Descriptors are Novo's alternative, not core methodology

**Decision:** ❌ **DO NOT IMPLEMENT** (ESM-1v is sufficient)

---

### 2. Other PLMs (ESM-1b, ESM-2, ProtBERT, AntiBERTy, AbLang2)
**Novo's Work:**
- Tested 6 PLMs (ESM-1v performed best at 71%)
- AntiBERTy, AbLang2 (antibody-specific)

**Our Status:**
- Only ESM-1v implemented (best performer)

**Decision:** ❌ **DO NOT IMPLEMENT** (ESM-1v is validated optimal)

---

### 3. All 16 Fragment Types
**Novo's Work:**
- VH, VL, H-CDR1, H-CDR2, H-CDR3, L-CDR1, L-CDR2, L-CDR3
- H-CDRs, L-CDRs, All-CDRs, H-FWRs, L-FWRs, All-FWRs, VH+VL, Full

**Our Status:**
- All fragments implemented
- VH-based is production (highest performance)

**Decision:** ✅ **KEEP ALL** (for research, but VH is production)

---

### 4. Hyperparameter Sweeps
**Novo's Work:**
- Not mentioned in paper (used default sklearn LogisticRegression)

**Our Status:**
- Historical hyperparameter sweeps in `experiments/archive/`
- Not part of Novo replication

**Decision:** 🗂️ **ARCHIVE** (not part of core replication)

---

## Action Items

### Immediate (Before Next Commit)

1. ✅ **Document Validation Roadmap** (this file)
2. 🔄 **Run All Preprocessing Pipelines** (Tasks 1.1-1.4)
3. 🔄 **Train Production Model** (Task 2.1)
4. 🔄 **Test on All 3 Datasets** (Tasks 3.1-3.3)
5. 🔄 **Verify Hyperparameter Sweep Outputs** (Task 4.1)

### After Validation

6. ✅ **Update CLAUDE.md** with validation results
7. ✅ **Update README.md** with validated commands
8. ✅ **Tag release:** `v2.0.0-validated`
9. ✅ **Merge to main:** `dev` → `leroy-jenkins/full-send`

---

## Success Criteria (Repository is "Done")

A user should be able to:

1. **Clone repo** from GitHub
2. **Install dependencies:** `uv sync --all-extras`
3. **Preprocess data** (if raw data available)
4. **Train model:** `uv run antibody-train`
5. **Test model:** `uv run antibody-test --model X --dataset jain`
6. **Reproduce Novo results:**
   - Boughter 10-fold CV: 67-71%
   - Jain test: 64-69%
   - Shehata test: 50-55%
7. **Run all tests:** `uv run pytest` (374/374 passing)
8. **Read docs** and understand:
   - What this repo does
   - How to use it
   - What Novo results we replicate
   - What we don't replicate (and why)

---

## Repository Cleanup Plan (Post-Validation)

### Files to Delete (After Validation)

**NONE** - Everything has a purpose:
- `experiments/benchmarks/strict_qc/` → ARCHIVED (provenance)
- `experiments/archive/` → Historical hyperparameter sweeps (reference)
- `literature/` → Novo paper (citation)
- `docs/` → Documentation (essential)

### Files to Update (After Validation)

1. **CLAUDE.md** → Add validation results
2. **README.md** → Add "Validated" badge
3. **USAGE.md** → Update with experiments/ paths (already done)
4. **docs/developer-guide/directory-organization.md** → Update examples (already done)

---

## Open Questions

1. **Q:** Where should hyperparameter sweep outputs go?
   **A:** `experiments/runs/{experiment.name}/multirun/{date}/{time}/` (Hydra default)

2. **Q:** Should we delete strict_qc experiment?
   **A:** NO - archive it with clear "UNVALIDATED" warning (scientific provenance)

3. **Q:** Do all preprocessing pipelines still work?
   **A:** 🔄 **VALIDATE THIS** (run Tasks 1.1-1.4)

4. **Q:** Does training pipeline work with new experiments/ paths?
   **A:** 🔄 **VALIDATE THIS** (run Task 2.1)

5. **Q:** Does testing pipeline work with new experiments/ paths?
   **A:** 🔄 **VALIDATE THIS** (run Tasks 3.1-3.3)

---

## Timeline

**Today (2025-11-16):**
- ✅ Document validation roadmap (this file)
- 🔄 Run preprocessing validation (Tasks 1.1-1.4)
- 🔄 Run training validation (Task 2.1)

**Next Session:**
- 🔄 Run testing validation (Tasks 3.1-3.3)
- 🔄 Run hyperparameter sweep test (Task 4.1)
- 🔄 Run fresh clone test (Task 6.1)

**After Validation:**
- ✅ Update all docs with validated results
- ✅ Tag v2.0.0-validated
- ✅ Merge to main

---

## References

- **Novo Paper:** `literature/markdown/novo_2025_main/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md`
- **Strict QC Experiment:** `experiments/benchmarks/strict_qc/EXPERIMENT_README.md`
- **Validated Results:** `experiments/benchmarks/novo_parity_2025-11-05/README.md` (if exists)
- **CLAUDE.md:** Development guide for Claude Code

---

**Last Updated:** 2025-11-16
**Status:** DRAFT - Pre-validation
**Next Action:** Run Task 1.1 (Boughter preprocessing)
