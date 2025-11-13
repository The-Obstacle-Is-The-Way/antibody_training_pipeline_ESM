# Tessier 2024 Dataset Preprocessing Plan

**Status:** ✅ APPROVED - Implementation in progress
**Priority:** HIGH - Critical for Ginkgo 2025 competition improvement
**Expected Impact:** Transfer learning with 246k CHO antibodies → CV score 0.50664 → 0.70+ (estimated)

---

## Baseline Confirmation (2025-11-13 08:10)

**✅ Clean baseline established:** 0.50664 Spearman (LightGBM + ESM-1v + p-IgGen)

**Bug investigation completed:**
- Investigated p-IgGen formatting bugs (sentinel spacing, eval mode)
- **Finding:** Tokenizer treats both formats identically - bug was cosmetic
- Token IDs are 100% identical for old vs new format
- Predictions are bit-for-bit identical (0.0 difference)
- **Conclusion:** 0.50664 is our legitimate, reproducible baseline

**Why transfer learning is critical:**
- Gap to leader: -0.38 (-43% below 0.89)
- All folds stopped at iteration 0 (no learnable patterns in embeddings)
- Better heads (LightGBM, ElasticNet) don't help without better features
- Need 246k CHO antibodies to learn polyreactivity patterns

---

## Executive Summary

**Problem:** LightGBM experiment showed minimal improvement (+0.006, now at 0.50664 vs leader 0.89)
- Root cause: Embeddings don't capture task-specific patterns (all folds stopped at iteration 0)
- Fold 3 still catastrophic (0.336 vs others 0.41-0.70)

**Solution:** Transfer learning with Tessier 2024 dataset (246k antibodies, CHO assay)

**Why Tessier Beats Boughter:**
| Dataset | Size | Assay | Match to GDPa1 | Result |
|---------|------|-------|----------------|--------|
| **Boughter** | 914 | ELISA (binary) | ❌ Different assay | -1.93% (FAILED) |
| **Tessier** | 246,295 | CHO cell (binary) | ✅ Same cell type! | 🎯 TBD |
| **GDPa1** | 197 | PR_CHO (continuous) | — | 0.50664 baseline |

**Key Advantage:** Tessier uses **CHO cell-based polyreactivity assays** (SCP60, SMP60) — same as GDPa1's PR_CHO!

---

## Dataset Overview

**Paper:** Feld et al. (2024) "Human antibody polyreactivity is governed primarily by the heavy-chain complementarity-determining regions" *Cell Reports* (Oct 2024)

**Location:** `external_datasets/tessier_2024_polyreactivity/`

**Downloaded Files:**
```
├── Feld_pos.csv                 # 115,039 polyreactive antibodies
├── Feld_neg.csv                 # 131,256 specific antibodies
└── Supplemental Datasets/
    ├── Human Ab Poly Dataset S1_v2.xlsx   # Sequences + labels
    └── Human Ab Poly Dataset S2_v2.xlsx   # Additional sequences
```

**Dataset S1 Structure (confirmed):**
- Columns: `['Name', 'VL', 'VH']`
- Name: "high non-spec_1", "high non-spec_2", ..., "low non-spec_1", ...
- VL: Full light chain amino acid sequence
- VH: Full heavy chain amino acid sequence
- Labels: Inferred from name prefix ("high non-spec" → polyreactive, "low non-spec" → specific)

---

## Preprocessing Pipeline (3 Stages)

### Stage 1: Extract Sequences from Excel (`step1_extract_sequences.py`)

**Inputs:**
- `external_datasets/tessier_2024_polyreactivity/Supplemental Datasets/Human Ab Poly Dataset S1_v2.xlsx`
- `external_datasets/tessier_2024_polyreactivity/Supplemental Datasets/Human Ab Poly Dataset S2_v2.xlsx`

**Process:**
1. Load Excel sheets with pandas
2. Extract VH and VL sequences
3. Assign binary labels based on `Name` column:
   - "high non-spec" → label = 1 (polyreactive)
   - "low non-spec" → label = 0 (specific)
4. Merge datasets S1 + S2
5. Deduplicate by (VH, VL) pair
6. Validate: Check no NaN sequences, confirm 246k total

**Outputs:**
- `train_datasets/tessier/processed/tessier_raw.csv`
  - Columns: `['antibody_id', 'vh_sequence', 'vl_sequence', 'label_binary', 'source_name']`
  - Expected: ~246,295 rows

**Validation:**
- Assert 115,039 positive + 131,256 negative = 246,295 total
- Check sequence lengths (VH: 110-130 aa, VL: 105-115 aa typical)
- Confirm no empty sequences

---

### Stage 2: ANARCI Annotation (`step2_annotate_anarci.py`)

**Inputs:**
- `train_datasets/tessier/processed/tessier_raw.csv`

**Process:**
1. Initialize ANARCI with IMGT numbering scheme
2. Annotate VH and VL sequences
3. Extract fragments:
   - VH: H-CDR1, H-CDR2, H-CDR3, H-FWRs, VH_only
   - VL: L-CDR1, L-CDR2, L-CDR3, L-FWRs, VL_only
   - Combined: H-CDRs, L-CDRs, All-CDRs, H-FWRs, L-FWRs, All-FWRs, VH+VL
4. Log annotation failures

**Outputs:**
- `train_datasets/tessier/annotated/tessier_annotated.csv`
- `train_datasets/tessier/annotated/annotation_failures.log`

**Expected Success Rate:** ~99% (based on Boughter: 99.4%)

---

### Stage 3: Quality Control & Split (`step3_qc_and_split.py`)

**Inputs:**
- `train_datasets/tessier/annotated/tessier_annotated.csv`

**Process:**
1. **Quality filters:**
   - Remove sequences with 'X' in CDRs (ambiguous residues)
   - Remove sequences with empty CDRs (annotation artifacts)
   - Remove sequences with CDR3 length > 30 aa (outliers)
2. **Train/val split:**
   - Stratified split: 80% train, 20% val
   - Preserve label balance (polyreactive vs specific)
3. **Create fragment CSVs:**
   - 16 fragment types (same as Boughter structure)
   - Columns: `['id', 'sequence', 'label', 'split']`

**Outputs:**
- `train_datasets/tessier/canonical/VH_only_tessier_training.csv` (~196k rows)
- `train_datasets/tessier/canonical/VH_only_tessier_validation.csv` (~49k rows)
- `train_datasets/tessier/canonical/VH_VL_tessier_training.csv`
- `train_datasets/tessier/annotated/qc_filtered_sequences.txt`

**Expected Retention:** ~95% (based on Boughter: 95.9%)

---

## Transfer Learning Strategy

### Approach: Pre-train → Fine-tune

**Step 1: Pre-train on Tessier (~196k samples)**
```python
# Extract ESM-1v + p-IgGen embeddings for Tessier training set
tessier_embeddings = extract_embeddings(tessier_train)  # Shape: (196k, 3328)

# Pre-train Ridge/LightGBM on Tessier
pretrain_model = LGBMRegressor(**params)
pretrain_model.fit(tessier_embeddings, tessier_labels)  # Binary labels
```

**Step 2: Fine-tune on GDPa1 (197 samples)**
```python
# Extract embeddings for GDPa1
ginkgo_embeddings = extract_embeddings(ginkgo_train)  # Shape: (197, 3328)

# Initialize with pre-trained weights (if using neural head)
# OR use pre-trained embeddings as features (if using GBDT)

# Fine-tune on GDPa1
finetune_model = LGBMRegressor(**params)
finetune_model.fit(ginkgo_embeddings, ginkgo_pr_cho_labels)  # Continuous labels
```

**Key Difference from Failed Boughter Transfer:**
- ✅ Tessier uses **CHO cells** (same as GDPa1!)
- ✅ 270x more data (196k vs 914)
- ❌ Labels still binary (not continuous like PR_CHO)

**Expected Improvement:**
- Baseline: 0.50664 (LightGBM alone)
- Target: 0.70+ (transfer learning + LightGBM)
- Stretch: 0.85+ (if CHO cell patterns transfer well)

---

## Implementation Timeline

### Phase 1: Preprocessing (4-6 hours)
1. **Stage 1** (1 hour): Extract sequences from Excel
2. **Stage 2** (2-3 hours): ANARCI annotation (slow for 246k sequences)
3. **Stage 3** (1 hour): QC + split

### Phase 2: Transfer Learning (2-3 hours)
1. **Extract Tessier embeddings** (1 hour): ESM-1v + p-IgGen on 196k sequences
2. **Pre-train model** (30 min): Ridge/LightGBM on Tessier
3. **Fine-tune on GDPa1** (30 min): Transfer to PR_CHO prediction
4. **Generate submission** (30 min): CV + test predictions

### Phase 3: Iteration (if needed)
- Try different transfer strategies (frozen embeddings, full fine-tuning)
- Experiment with label transformations (binary → continuous)
- Ensemble with non-transfer models

---

## Risk Assessment

### Risk 1: Binary vs Continuous Labels
**Issue:** Tessier has binary labels (0/1), GDPa1 has continuous PR_CHO (0-0.547)

**Mitigation:**
- Pre-train on binary classification task
- Fine-tune on continuous regression task
- Model learns "polyreactive patterns" from Tessier, adapts to PR_CHO scale in fine-tuning

**Precedent:** This is standard practice in NLP (pre-train on classification, fine-tune on regression)

### Risk 2: ANARCI Annotation Time
**Issue:** 246k sequences × 0.05 sec/sequence = 3.4 hours

**Mitigation:**
- Run ANARCI in parallel (multiprocessing)
- Use batch processing (1000 sequences per batch)
- Cache intermediate results

### Risk 3: Embedding Storage
**Issue:** 246k sequences × 3328 dims × 4 bytes/float = 3.3 GB

**Mitigation:**
- Use float16 instead of float32 (halves storage)
- Cache embeddings on disk (`.npy` files)
- Process in batches if memory constrained

### Risk 4: Still Doesn't Beat Leader (0.89)
**Issue:** Tessier transfer might not be enough

**Mitigation:**
- **Plan B:** Ensemble Tessier-transfer + non-transfer models
- **Plan C:** Try TabPFN v2.5 (foundation model for tabular data)
- **Plan D:** Investigate Fold 3 specifically (0.336 bottleneck)

---

## Success Criteria

**Minimum Viable Success:**
- ✅ Preprocessing completes without errors
- ✅ Transfer learning beats non-transfer baseline (+0.01 Spearman)
- ✅ CV score improves from 0.50664 → 0.52+ (submit to leaderboard)

**Target Success:**
- ✅ CV score reaches 0.70+ (major improvement)
- ✅ Beats current leader on private test (0.89+ target)
- ✅ Fold 3 improves from 0.336 → 0.50+

**Stretch Success:**
- ✅ CV score reaches 0.85+ (top of leaderboard)
- ✅ All folds perform consistently (0.70-0.90 range)
- ✅ Publishable methodology (ESM + transfer learning for antibody developability)

---

## File Structure After Preprocessing

```
train_datasets/tessier/
├── processed/
│   └── tessier_raw.csv                    # Stage 1 output (246k rows)
├── annotated/
│   ├── tessier_annotated.csv              # Stage 2 output (~244k rows)
│   ├── annotation_failures.log
│   └── qc_filtered_sequences.txt
├── canonical/
│   ├── VH_only_tessier_training.csv       # Stage 3 output (~196k rows)
│   ├── VH_only_tessier_validation.csv     # Stage 3 output (~49k rows)
│   ├── VH_VL_tessier_training.csv
│   ├── VH_VL_tessier_validation.csv
│   └── [14 other fragment CSVs]
└── README.md                               # Dataset documentation

preprocessing/tessier/
├── step1_extract_sequences.py              # Extract from Excel
├── step2_annotate_anarci.py                # ANARCI annotation
├── step3_qc_and_split.py                   # QC + train/val split
└── validate_stages.py                      # Validation script

scripts/
└── ginkgo_tessier_transfer.py              # Transfer learning experiment
```

---

## Implementation Progress

**Phase 1: Preprocessing Scripts** ⏳ IN PROGRESS
1. ✅ Senior approval received
2. ⏳ Create `preprocessing/tessier/` directory
3. ⏳ Implement 3 preprocessing scripts (step1, step2, step3)
4. ⬜ Run preprocessing pipeline (4-6 hours)
5. ⬜ Validate outputs

**Phase 2: Transfer Learning** ⬜ PENDING
1. ⬜ Implement transfer learning script (`ginkgo_tessier_transfer.py`)
2. ⬜ Run transfer learning experiment (2-3 hours)
3. ⬜ Generate submission if CV > 0.52
4. ⬜ Submit to leaderboard and iterate

---

## Implementation Decisions

1. **Label transformation:** Use binary labels as-is (standard pre-train classification → fine-tune regression)
2. **Validation set:** Validate on GDPa1 folds only (Tessier val set for transfer learning diagnostics)
3. **Embedding model:** Stick with ESM-1v + p-IgGen (already working, don't change too many variables)
4. **Compute budget:** Run ANARCI overnight if needed (3-4 hours expected)
5. **Storage:** Keep embeddings local (3.3 GB, regenerate if needed)

---

**Last Updated:** 2025-11-13 08:12 EST
**Author:** Ray + Claude
**Status:** ✅ APPROVED - IMPLEMENTING
