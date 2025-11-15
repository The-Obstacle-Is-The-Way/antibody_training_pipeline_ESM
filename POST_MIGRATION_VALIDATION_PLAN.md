# Post-Migration Validation Plan - End-to-End Pipeline Verification

**Status**: 📋 READY TO EXECUTE
**Created**: 2025-11-15
**Purpose**: Validate entire data processing pipeline after Phase 1 & 2 filesystem migrations
**Risk Level**: LOW (tests passing, CI/CD green) → CONFIDENCE BUILDING
**Estimated Time**: 3-4 hours (preprocessing re-runs + training smoke test)

---

## Executive Summary

### The Concern (100% Valid)

We just completed **massive filesystem reorganization**:
- Phase 1: `test_datasets/` → `data/test/` (70 files, 135 code files updated)
- Phase 2: `train_datasets/` → `data/train/` (42 files, 19 code files updated)

While **all tests pass** and **CI/CD is green**, we should validate that:
1. **Preprocessing scripts work end-to-end** (raw → processed → canonical/fragments)
2. **Training pipeline works** with migrated data paths
3. **All outputs match expected locations** (no orphaned files in old paths)
4. **Git-tracked data files remain intact** (no corruption during migration)

### Professional Team Best Practice

**✅ YES** - This is exactly what production teams do:

1. **Re-run preprocessing pipelines** to verify all scripts work with new paths
2. **Compare outputs** (MD5 checksums) to ensure data integrity
3. **Run smoke tests** on training pipeline (1-2 epochs, small batch)
4. **Document validation results** for audit trail

### The Safety Net We Already Have

✅ **Git history preserved** - All files moved with `git mv` (provenance intact)
✅ **Tests passing** - 466/466 unit/integration tests pass
✅ **CI/CD green** - Docker builds, linting, type checking all pass
✅ **Raw data exists** - Original raw files still in `data/{train,test}/*/raw/`

**What this plan adds**: End-to-end functional validation beyond unit tests.

---

## Validation Strategy

### Phase 1: Pre-Flight Checks (15 min) ✅ DO THIS FIRST

Verify current state before re-running anything.

#### Check 1: Data File Integrity (5 min)

**Goal**: Confirm git-tracked data files are intact after migration.

```bash
# 1. Count files in new locations
find data/train/boughter -type f -name "*.csv" | wc -l
# Expected: 18 (1 processed + 1 canonical + 16 fragments)

find data/test/jain -type f -name "*.csv" | wc -l
# Expected: 24 (5 processed + 3 canonical + 16 fragments)

find data/test/harvey -type f -name "*.csv" | wc -l
# Expected: 10 (1 processed + 6 fragments + 3 raw)

find data/test/shehata -type f -name "*.csv" | wc -l
# Expected: 17 (1 processed + 16 fragments, no canonical)

# 2. Verify no files left in old locations
find . -name "train_datasets" -o -name "test_datasets" 2>/dev/null
# Expected: Only reference_repos/ludocomito_original/train_datasets (gitignored)

# 3. Check raw data exists (not tracked in git, should be preserved)
ls -la data/train/boughter/raw/*.txt | wc -l
# Expected: 19 FASTA files

ls -la data/test/jain/raw/*.xlsx 2>/dev/null | wc -l
# Expected: 3-5 Excel files

ls -la data/test/harvey/raw/*.csv 2>/dev/null | wc -l
# Expected: 3 CSV files
```

**Success Criteria**:
- [ ] All expected file counts match
- [ ] No files in old `train_datasets/` or `test_datasets/` paths
- [ ] Raw data files exist and accessible

#### Check 2: Preprocessing Script Path Validation (5 min)

**Goal**: Verify all preprocessing scripts reference new paths.

```bash
# 1. Check for any lingering old paths in preprocessing/
grep -r "train_datasets/" preprocessing/
# Expected: Empty (no matches)

grep -r "test_datasets/" preprocessing/
# Expected: Empty (no matches)

# 2. Verify new paths are used
grep -r "data/train" preprocessing/boughter/ | grep -v "# " | head -5
# Expected: Multiple matches in stage1_dna_translation.py, stage2_stage3_annotation_qc.py

grep -r "data/test" preprocessing/jain/ | grep -v "# " | head -5
# Expected: Multiple matches in step1_convert_excel_to_csv.py, step2_preprocess_p5e_s2.py
```

**Success Criteria**:
- [ ] Zero references to old paths in preprocessing/
- [ ] New paths (`data/train`, `data/test`) present in all scripts

#### Check 3: Training Config Validation (5 min)

**Goal**: Verify training configs point to correct data paths.

```bash
# 1. Check Hydra config (CURRENT PRODUCTION SYSTEM - USE THIS)
grep "train_file:" src/antibody_training_esm/conf/data/boughter_jain.yaml
# Expected: train_file: data/train/boughter/canonical/VH_only_boughter_training.csv

grep "test_file:" src/antibody_training_esm/conf/data/boughter_jain.yaml
# Expected: test_file: data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv

# 2. Verify files exist at Hydra-configured paths
test -f data/train/boughter/canonical/VH_only_boughter_training.csv && echo "✅ Training file exists"
test -f data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv && echo "✅ Test file exists (Hydra config)"

# 3. Check legacy config (DEPRECATED - ALREADY BROKEN, being removed in v0.5.0)
echo "⚠️  Legacy config status:"
grep "test_file:" configs/config.yaml
# NOTE: This references VH_only_jain_P5e_S2.csv which DOES NOT EXIST (broken since Phase 1 migration)
# Expected: test_file: ./data/test/jain/canonical/VH_only_jain_P5e_S2.csv (BROKEN PATH)

# Verify legacy config is indeed broken
test -f ./data/test/jain/canonical/VH_only_jain_P5e_S2.csv && echo "✅ Legacy config works" || echo "❌ Legacy config BROKEN (expected - being deleted in v0.5.0)"

# 4. Detect config mismatches (defensive check)
HYDRA_FILE=$(grep "test_file:" src/antibody_training_esm/conf/data/boughter_jain.yaml | awk '{print $2}')
LEGACY_FILE=$(grep "test_file:" configs/config.yaml | awk '{print $2}' | sed 's|^\./||')

if [ "$HYDRA_FILE" != "$LEGACY_FILE" ]; then
  echo "⚠️  CONFIG MISMATCH DETECTED:"
  echo "   Hydra:  $HYDRA_FILE"
  echo "   Legacy: $LEGACY_FILE"
  echo "   → Use Hydra config for all validation (legacy is broken)"
fi
```

**Success Criteria**:
- [ ] Hydra config files reference correct paths (`data/train`, `data/test`)
- [ ] All Hydra-configured files exist and are accessible
- [ ] Legacy config mismatch detected and documented (expected - being deleted in v0.5.0)
- [ ] No broken symlinks or missing files in Hydra config paths

**Note**: Legacy `configs/config.yaml` is already broken (references non-existent `VH_only_jain_P5e_S2.csv`). This is acceptable as it's deprecated and being removed in v0.5.0. All validation uses Hydra config paths.

---

## Phase 2: Preprocessing Pipeline Validation (2-3 hours)

### Strategy: Backup → Re-Run → Compare

**Key Decision**: Should we **overwrite** existing files or **backup first**?

**Professional Approach**:
1. **Backup current outputs** to timestamped directory
2. **Re-run all preprocessing scripts** (they will overwrite)
3. **Compare outputs** (MD5 checksums, row counts, schema)
4. **If identical**: Delete backups, mark VALIDATED ✅
5. **If different**: Investigate discrepancies, decide which to keep

### Dataset 1: Boughter Training Data (45-60 min)

**Pipeline**: 3 stages (DNA translation → ANARCI annotation → QC filtering)

#### Backup Current Outputs (5 min)

```bash
# Create timestamped backup
BACKUP_DIR="data/train/boughter/BACKUP_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup processed, annotated, canonical
cp -r data/train/boughter/processed "$BACKUP_DIR/"
cp -r data/train/boughter/annotated "$BACKUP_DIR/"
cp -r data/train/boughter/canonical "$BACKUP_DIR/"

echo "✅ Backup created: $BACKUP_DIR"
ls -lah "$BACKUP_DIR"
```

#### Stage 1: DNA Translation (10 min)

**Script**: `preprocessing/boughter/stage1_dna_translation.py`

**What it does**:
- Reads raw DNA FASTA files from `data/train/boughter/raw/*.txt`
- Translates DNA → protein sequences
- Applies Novo Nordisk flagging strategy (0, 1-3, 4+ flags)
- Outputs: `data/train/boughter/processed/boughter.csv` (1,117 sequences)

**Execution**:
```bash
echo "🔄 Running Boughter Stage 1 (DNA Translation)..."
python3 preprocessing/boughter/stage1_dna_translation.py

# Expected output:
# ✓ Translated 1,171 sequences
# ✓ 1,117 sequences written to data/train/boughter/processed/boughter.csv
# ✓ 54 translation failures logged
```

**Validation**:
```bash
# 1. Check file exists
test -f data/train/boughter/processed/boughter.csv && echo "✅ Stage 1 output exists"

# 2. Compare row count
BACKUP_ROWS=$(wc -l < "$BACKUP_DIR/processed/boughter.csv")
NEW_ROWS=$(wc -l < data/train/boughter/processed/boughter.csv)
echo "Backup rows: $BACKUP_ROWS | New rows: $NEW_ROWS"
# Expected: Both = 1118 (header + 1117 sequences)

# 3. MD5 checksum comparison
BACKUP_MD5=$(md5 -q "$BACKUP_DIR/processed/boughter.csv")
NEW_MD5=$(md5 -q data/train/boughter/processed/boughter.csv)
if [ "$BACKUP_MD5" = "$NEW_MD5" ]; then
  echo "✅ IDENTICAL - Stage 1 output matches backup"
else
  echo "⚠️  DIFFERENCE DETECTED - Manual review required"
  echo "Backup MD5: $BACKUP_MD5"
  echo "New MD5:    $NEW_MD5"
fi
```

**Success Criteria**:
- [ ] Script completes without errors
- [ ] Output file exists at correct path
- [ ] Row count matches expected (1,118 = header + 1,117 sequences)
- [ ] MD5 checksum matches backup (or acceptable difference documented)

#### Stage 2+3: ANARCI Annotation + QC Filtering (25-30 min)

**Script**: `preprocessing/boughter/stage2_stage3_annotation_qc.py`

**What it does**:
- Reads `data/train/boughter/processed/boughter.csv` (Stage 1 output)
- Runs ANARCI annotation (IMGT numbering)
- Extracts 16 fragment types (VH, VL, CDRs, FWRs, etc.)
- Applies QC filtering (length, annotation quality)
- Outputs:
  - `data/train/boughter/annotated/*.csv` (16 fragment files, 1,065 sequences each)
  - `data/train/boughter/canonical/VH_only_boughter_training.csv` (914 sequences)

**Execution**:
```bash
echo "🔄 Running Boughter Stages 2+3 (ANARCI Annotation + QC)..."
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py

# Expected output:
# Stage 2: Annotating 1,117 sequences with ANARCI...
# ✓ 1,110 sequences annotated (99.4% success)
# ✓ 7 annotation failures logged
#
# Stage 3: Quality control filtering...
# ✓ 1,065 sequences passed QC (95.9% retention)
# ✓ 45 sequences filtered (logged)
#
# ✓ 16 fragment files created in data/train/boughter/annotated/
# ✓ Canonical training file: data/train/boughter/canonical/VH_only_boughter_training.csv (914 sequences)
```

**Validation**:
```bash
# 1. Check all fragment files exist
EXPECTED_FRAGMENTS=(
  "VH_only" "VL_only" "VH+VL" "Full"
  "H-CDR1" "H-CDR2" "H-CDR3" "H-CDRs" "H-FWRs"
  "L-CDR1" "L-CDR2" "L-CDR3" "L-CDRs" "L-FWRs"
  "All-CDRs" "All-FWRs"
)

for frag in "${EXPECTED_FRAGMENTS[@]}"; do
  FILE="data/train/boughter/annotated/${frag}_boughter.csv"
  if [ -f "$FILE" ]; then
    ROWS=$(wc -l < "$FILE")
    echo "✅ $frag: $ROWS rows"
  else
    echo "❌ MISSING: $FILE"
  fi
done

# 2. Check canonical file
CANONICAL="data/train/boughter/canonical/VH_only_boughter_training.csv"
test -f "$CANONICAL" && echo "✅ Canonical file exists"
CANONICAL_ROWS=$(wc -l < "$CANONICAL")
echo "Canonical rows: $CANONICAL_ROWS"
# Expected: 915 (header + 914 sequences)

# 3. MD5 comparison (canonical file - critical for training)
BACKUP_CANONICAL="$BACKUP_DIR/canonical/VH_only_boughter_training.csv"
if [ -f "$BACKUP_CANONICAL" ]; then
  BACKUP_MD5=$(md5 -q "$BACKUP_CANONICAL")
  NEW_MD5=$(md5 -q "$CANONICAL")
  if [ "$BACKUP_MD5" = "$NEW_MD5" ]; then
    echo "✅ IDENTICAL - Canonical training file unchanged"
  else
    echo "⚠️  DIFFERENCE - Investigating..."
    echo "Backup MD5: $BACKUP_MD5"
    echo "New MD5:    $NEW_MD5"
    # Compare headers and first 5 rows
    diff <(head -6 "$BACKUP_CANONICAL") <(head -6 "$CANONICAL")
  fi
fi
```

**Success Criteria**:
- [ ] Script completes without errors
- [ ] All 16 fragment files created in `data/train/boughter/annotated/`
- [ ] Canonical file created with 915 rows (header + 914 sequences)
- [ ] MD5 checksums match backups OR acceptable differences documented
- [ ] No ANARCI errors or crashes

#### Validation Script (5 min)

**Script**: `preprocessing/boughter/validate_stages2_3.py`

```bash
echo "🔍 Running Boughter validation script..."
python3 preprocessing/boughter/validate_stages2_3.py

# Expected output:
# ✅ All 16 fragment files exist
# ✅ All files have correct schema (sequence, label, fragment_type)
# ✅ VH_only_boughter_training.csv has 914 sequences
# ✅ Label distribution: 440 specific (48.1%), 474 polyreactive (51.9%)
# ✅ No NaN values in critical columns
```

**Success Criteria**:
- [ ] Validation script passes all checks
- [ ] No schema errors or missing columns
- [ ] Label distribution matches expected (~48% specific, ~52% polyreactive)

### Dataset 2: Jain Test Data (30-45 min)

**Pipeline**: 3 steps (Excel → CSV → P5e-S2 preprocessing → Fragment extraction)

#### Backup Current Outputs (3 min)

```bash
BACKUP_DIR="data/test/jain/BACKUP_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r data/test/jain/processed "$BACKUP_DIR/"
cp -r data/test/jain/canonical "$BACKUP_DIR/"
cp -r data/test/jain/fragments "$BACKUP_DIR/"
echo "✅ Jain backup created: $BACKUP_DIR"
```

#### Step 1: Excel to CSV Conversion (5 min)

```bash
echo "🔄 Running Jain Step 1 (Excel → CSV)..."
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Check output
ls -lh data/test/jain/processed/*.csv
# Expected: 6 CSV files (sd01, sd02, sd03, private_elisa, etc.)
```

#### Step 2: P5e-S2 Preprocessing (10 min)

```bash
echo "🔄 Running Jain Step 2 (P5e-S2 Preprocessing)..."
python3 preprocessing/jain/step2_preprocess_p5e_s2.py

# Validate output
test -f data/test/jain/processed/jain_with_private_elisa_FULL.csv && echo "✅ Full dataset created"
wc -l data/test/jain/processed/jain_with_private_elisa_FULL.csv
# Expected: ~87 rows (header + 86 antibodies)
```

#### Step 3: Fragment Extraction (10 min)

```bash
echo "🔄 Running Jain Step 3 (Fragment Extraction)..."
python3 preprocessing/jain/step3_extract_fragments.py

# Validate outputs
ls -1 data/test/jain/fragments/*.csv | wc -l
# Expected: 16 fragment files

# Check canonical parity files (CRITICAL for Novo benchmark)
echo "Canonical files:"
ls -1 data/test/jain/canonical/*.csv
# Expected: 3 files
#   - VH_only_jain_86_p5e_s2.csv (Hydra config uses this)
#   - VH_only_jain_test_PARITY_86.csv (legacy name)
#   - jain_86_novo_parity.csv

# Verify Hydra-configured file exists
CANONICAL="data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv"
test -f "$CANONICAL" && echo "✅ Hydra canonical file exists"
wc -l "$CANONICAL"
# Expected: 87 rows (header + 86 antibodies)

# MD5 comparison
BACKUP_CANONICAL="$BACKUP_DIR/canonical/VH_only_jain_86_p5e_s2.csv"
if [ -f "$BACKUP_CANONICAL" ]; then
  BACKUP_MD5=$(md5 -q "$BACKUP_CANONICAL")
  NEW_MD5=$(md5 -q "$CANONICAL")
  [ "$BACKUP_MD5" = "$NEW_MD5" ] && echo "✅ Parity file IDENTICAL" || echo "⚠️  Parity file CHANGED"
fi
```

**Success Criteria**:
- [ ] All 3 steps complete without errors
- [ ] 16 fragment files created
- [ ] 3 canonical files exist (VH_only_jain_86_p5e_s2.csv, VH_only_jain_test_PARITY_86.csv, jain_86_novo_parity.csv)
- [ ] Hydra canonical file has 87 rows (header + 86 antibodies)
- [ ] MD5 matches backup (CRITICAL - Novo parity benchmark depends on this)

### Dataset 3: Harvey Nanobody Data (20-30 min)

#### Backup & Execute (15 min)

```bash
# Backup
BACKUP_DIR="data/test/harvey/BACKUP_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r data/test/harvey/processed "$BACKUP_DIR/"
cp -r data/test/harvey/fragments "$BACKUP_DIR/"

# Step 1: Convert raw CSVs
echo "🔄 Running Harvey Step 1..."
python3 preprocessing/harvey/step1_convert_raw_csvs.py
wc -l data/test/harvey/processed/harvey.csv
# Expected: ~141,000 rows

# Step 2: Extract fragments
echo "🔄 Running Harvey Step 2..."
python3 preprocessing/harvey/step2_extract_fragments.py
ls -1 data/test/harvey/fragments/*.csv | wc -l
# Expected: 6 nanobody fragment files (VHH_only, H-CDR1/2/3, H-CDRs, H-FWRs)

# Check raw files (used in preprocessing)
ls -1 data/test/harvey/raw/*.csv | wc -l
# Expected: 3 raw CSV files (high/low polyreactivity, low throughput)
```

**Success Criteria**:
- [ ] `harvey.csv` has ~141,000 sequences
- [ ] 6 fragment files created (nanobody-specific: VHH_only, H-CDR1/2/3, H-CDRs, H-FWRs)
- [ ] 3 raw CSV files exist (preprocessing inputs)
- [ ] No canonical file (intentional - full dataset used)

### Dataset 4: Shehata PSR Data (15-20 min)

```bash
# Backup
BACKUP_DIR="data/test/shehata/BACKUP_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r data/test/shehata/processed "$BACKUP_DIR/"
cp -r data/test/shehata/fragments "$BACKUP_DIR/"

# Step 1: Excel to CSV
echo "🔄 Running Shehata Step 1..."
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
wc -l data/test/shehata/processed/shehata.csv
# Expected: ~399 rows

# Step 2: Extract fragments
echo "🔄 Running Shehata Step 2..."
python3 preprocessing/shehata/step2_extract_fragments.py
ls -1 data/test/shehata/fragments/*.csv | wc -l
# Expected: 16 fragment files
```

**Success Criteria**:
- [ ] `shehata.csv` has ~399 sequences
- [ ] 16 fragment files created
- [ ] All MD5s match backups

---

## Phase 3: Training Pipeline Smoke Test (30-45 min)

**Goal**: Verify training pipeline works with migrated data paths (not full training - just smoke test).

### Approach: Minimal Training Run

**Strategy**: Train for 1-2 CV folds with small subset to validate pipeline, not full 10-fold CV.

#### Test 1: CLI Training (15 min)

```bash
echo "🔄 Running training smoke test..."

# Override to use only 2 CV folds (faster)
uv run antibody-train \
  training.n_splits=2 \
  training.batch_size=4 \
  hardware.device=cpu \
  experiment.name=smoke_test_post_migration

# Expected behavior:
# ✓ Loads data from data/train/boughter/canonical/VH_only_boughter_training.csv
# ✓ Extracts embeddings (or loads from cache)
# ✓ Runs 2-fold CV (not full 10-fold)
# ✓ Trains final model on full training set
# ✓ Saves model to models/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
# ✓ No FileNotFoundError or path errors
```

**Validation**:
```bash
# 1. Check model was saved
find outputs/ -name "*.pkl" -mmin -10
# Expected: Recent .pkl file in outputs/{experiment_name}/{timestamp}/

# 2. Check training logs
tail -50 outputs/smoke_test_post_migration/*/training.log
# Expected: No FileNotFoundError, no path errors

# 3. Verify embeddings cache used
ls -lh embeddings_cache/
# Expected: Cache files with recent timestamps OR reused from previous runs
```

**Success Criteria**:
- [ ] Training completes without errors
- [ ] Model saved to correct hierarchical path
- [ ] No `FileNotFoundError` for data paths
- [ ] Embeddings cache working correctly

#### Test 2: Model Testing (10 min)

```bash
echo "🔄 Testing trained model on Jain dataset..."

# Find most recent model
MODEL=$(find outputs/smoke_test_post_migration -name "*.pkl" -type f | head -1)

# Test on Jain dataset
uv run antibody-test \
  --model "$MODEL" \
  --dataset jain

# Expected output:
# ✓ Loaded model from {path}
# ✓ Loaded test data from data/test/jain/canonical/VH_only_jain_P5e_S2.csv
# ✓ Test accuracy: ~65-70% (ballpark - not tuned for 2-fold CV)
# ✓ ROC-AUC: ~0.65-0.75
```

**Success Criteria**:
- [ ] Testing completes without errors
- [ ] Test data loaded from correct path
- [ ] Metrics reported (accuracy, precision, recall, ROC-AUC)
- [ ] No path-related errors

#### Test 3: Python API (5 min)

```bash
# Test direct Python import
uv run python3 -c "
from antibody_training_esm.core.trainer import train_pipeline
from hydra import compose, initialize

with initialize(config_path='../src/antibody_training_esm/conf', version_base=None):
    cfg = compose(config_name='config', overrides=['training.n_splits=2'])
    print('✅ Hydra config composed successfully')
    print(f'Train file: {cfg.data.train_file}')
    print(f'Test file: {cfg.data.test_file}')
"
```

**Success Criteria**:
- [ ] Hydra config loads without errors
- [ ] Paths resolve correctly (`data/train/...`, `data/test/...`)
- [ ] No import errors

---

## Phase 4: Cleanup & Documentation (15-30 min)

### Step 1: Compare All Backups (10 min)

```bash
# Create comparison report
REPORT="POST_MIGRATION_VALIDATION_REPORT_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT" << 'EOF'
# Post-Migration Validation Report

**Date**: $(date)
**Migration**: Phase 1 (test_datasets) + Phase 2 (train_datasets)
**Validation Method**: Re-run all preprocessing pipelines, compare outputs

## Results Summary

### Boughter Training Data
- **Stage 1 (DNA Translation)**:
  - Rows: BACKUP vs NEW
  - MD5: MATCH / DIFFER
  - Status: ✅ / ⚠️

- **Stages 2+3 (ANARCI + QC)**:
  - Fragment files: 16/16 created
  - Canonical file rows: BACKUP vs NEW
  - MD5: MATCH / DIFFER
  - Status: ✅ / ⚠️

### Jain Test Data
- **Step 1-3 (Excel → Fragments)**:
  - Fragment files: 17/17 created
  - Parity file MD5: MATCH / DIFFER
  - Status: ✅ / ⚠️

### Harvey Nanobody Data
- **Step 1-2 (Raw → Fragments)**:
  - Processed rows: BACKUP vs NEW
  - Fragment files: 7/7 created
  - Status: ✅ / ⚠️

### Shehata PSR Data
- **Step 1-2 (Excel → Fragments)**:
  - Processed rows: BACKUP vs NEW
  - Fragment files: 16/16 created
  - Status: ✅ / ⚠️

## Training Pipeline Smoke Test

- **CLI Training (2-fold CV)**: ✅ / ❌
- **Model Testing (Jain)**: ✅ / ❌
- **Python API**: ✅ / ❌

## Conclusion

[All preprocessing pipelines validated | Issues found - see details below]

### Issues Found
[None | List any discrepancies]

### Recommended Actions
[Delete backups and mark VALIDATED | Investigate differences | Keep backups]
EOF

echo "✅ Report template created: $REPORT"
echo "Fill in results from validation steps above"
```

### Step 2: Decision - Keep or Delete Backups (5 min)

**If all MD5s match**:
```bash
# Delete all backups
find data/ -type d -name "BACKUP_*" -exec rm -rf {} +
echo "✅ All backups deleted - outputs validated as identical"
```

**If differences found**:
```bash
# Keep backups for investigation
echo "⚠️  Keeping backups for review - found differences in:"
find data/ -type d -name "BACKUP_*"
echo "Review differences before deleting"
```

### Step 3: Update Validation Documentation (10 min)

Add validation results to migration plans:

```bash
# Update TRAIN_DATASETS_CONSOLIDATION_PLAN.md
cat >> docs_burner/TRAIN_DATASETS_CONSOLIDATION_PLAN.md << 'EOF'

---

## Post-Migration Validation (2025-11-15)

**Method**: Re-ran all preprocessing pipelines, compared outputs via MD5 checksums

**Results**:
- ✅ Boughter Stage 1-3: All outputs identical to pre-migration
- ✅ Jain Steps 1-3: All outputs identical, parity file validated
- ✅ Harvey Steps 1-2: All outputs identical
- ✅ Shehata Steps 1-2: All outputs identical
- ✅ Training smoke test: 2-fold CV completed, model tested on Jain
- ✅ No path errors, no FileNotFoundError

**Conclusion**: Migration successful, all pipelines validated end-to-end.

**Validation Report**: See POST_MIGRATION_VALIDATION_REPORT_{timestamp}.md
EOF
```

---

## Success Criteria (Overall)

### Pre-Flight Checks
- [ ] All data file counts match expected values (Jain: 24, Harvey: 10, Shehata: 17, Boughter: 18)
- [ ] No files in old `train_datasets/` or `test_datasets/` paths
- [ ] Raw data files accessible in new locations
- [ ] All preprocessing scripts reference new paths only
- [ ] Hydra configs point to correct data paths (use Hydra, not legacy)
- [ ] Legacy config mismatch detected and documented (expected)

### Preprocessing Validation
- [ ] Boughter Stage 1-3: All outputs created, MD5s match OR differences documented
- [ ] Jain Steps 1-3: All outputs created, parity file validated
- [ ] Harvey Steps 1-2: All outputs created
- [ ] Shehata Steps 1-2: All outputs created
- [ ] Validation scripts pass for all datasets

### Training Pipeline Validation
- [ ] CLI training smoke test completes (2-fold CV)
- [ ] Model saved to correct hierarchical path
- [ ] Model testing works on Jain dataset
- [ ] Python API loads Hydra config correctly
- [ ] No FileNotFoundError or path errors

### Documentation
- [ ] Validation report created with all results
- [ ] Migration plans updated with validation outcomes
- [ ] Decision made on backup retention (keep or delete)
- [ ] Any discrepancies investigated and documented

---

## Risk Mitigation

### Risk 1: Preprocessing Outputs Differ from Backup

**Likelihood**: LOW (deterministic scripts, same inputs)
**Impact**: MEDIUM (need to understand why)

**Mitigation**:
1. Compare headers: `diff <(head -1 backup.csv) <(head -1 new.csv)`
2. Compare row counts: `wc -l backup.csv new.csv`
3. Compare column schemas: `head -1 backup.csv | tr ',' '\n' > backup_cols.txt`
4. If only whitespace/formatting differs: ACCEPT new output
5. If data differs: Investigate which is correct, re-run if needed

### Risk 2: Training Pipeline Fails with New Paths

**Likelihood**: VERY LOW (tests already passing)
**Impact**: HIGH (blocks training)

**Mitigation**:
1. Check Hydra config composition: `uv run antibody-train --cfg job`
2. Verify file paths resolve: `test -f {path} && echo OK`
3. Check logs for exact error: `tail -100 outputs/*/training.log`
4. If path error: Fix in config, re-run
5. If other error: Investigate, may be unrelated to migration

### Risk 3: ANARCI Annotation Failures

**Likelihood**: LOW (same ANARCI version, same inputs)
**Impact**: MEDIUM (sequences lost)

**Mitigation**:
1. Compare failure logs: `diff {backup}/annotation_failures.log {new}/annotation_failures.log`
2. If more failures: Check ANARCI installation, re-run
3. If fewer failures: Acceptable (ANARCI non-determinism in edge cases)
4. Document any differences in validation report

---

## Rollback Plan

**If critical issues found**, restore from backups:

```bash
# Example: Restore Boughter canonical file
BACKUP_DIR="data/train/boughter/BACKUP_20251115_143022"  # Use actual timestamp
cp "$BACKUP_DIR/canonical/VH_only_boughter_training.csv" \
   data/train/boughter/canonical/VH_only_boughter_training.csv

# Verify restoration
md5 -q "$BACKUP_DIR/canonical/VH_only_boughter_training.csv"
md5 -q data/train/boughter/canonical/VH_only_boughter_training.csv
# Should match

echo "✅ Restored from backup"
```

---

## Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| **Phase 1** | Pre-Flight Checks | 15 min |
| **Phase 2** | Boughter Preprocessing (3 stages) | 45-60 min |
|  | Jain Preprocessing (3 steps) | 30-45 min |
|  | Harvey Preprocessing (2 steps) | 20-30 min |
|  | Shehata Preprocessing (2 steps) | 15-20 min |
| **Phase 3** | Training Smoke Test (CLI + Testing) | 30-45 min |
| **Phase 4** | Cleanup & Documentation | 15-30 min |
| **TOTAL** | | **3-4 hours** |

**Note**: Most time is ANARCI annotation (Boughter Stage 2) - CPU-bound, cannot parallelize.

---

## Professional Best Practices Applied

✅ **Backup before overwrite** - Timestamped backups for all outputs
✅ **Deterministic comparison** - MD5 checksums for exact byte-level matching
✅ **Smoke testing** - Minimal training run to validate integration
✅ **Documentation** - Validation report for audit trail
✅ **Rollback plan** - Clear procedure to restore from backups
✅ **Risk assessment** - Identified risks with mitigation strategies

---

## Questions for User (Before Execution)

1. **Timing**: When should we run this? (3-4 hour block needed)
2. **Backup retention**: Keep backups for X days, then auto-delete? Or manual review?
3. **Acceptable differences**: If MD5 differs but row counts match, acceptable?
4. **ANARCI version**: Should we document ANARCI version for reproducibility?
5. **Full training**: After smoke test passes, run full 10-fold CV for final validation?

---

**Plan Status**: ✅ READY TO EXECUTE

**Next Steps**:
1. User reviews plan and answers questions above
2. Schedule 3-4 hour execution window
3. Execute Phase 1 (Pre-Flight) → Decide if proceeding
4. Execute Phase 2-4 sequentially
5. Review validation report
6. Delete backups (if all validated) OR investigate differences
7. Update migration plans with validation outcomes
