# Shehata Dataset Preprocessing Implementation Plan

**Issue:** #3 - Shehata dataset preprocessing
**Author:** Ray (ray/learning branch)
**Date:** 2025-10-31
**Status:** Phase 1 Complete - Phase 2 In Progress

---

## Executive Summary

### What is the Shehata Dataset?
The Shehata dataset (Shehata et al. 2019, Cell Reports) contains **398 antibodies** from human B cell subsets (naïve, IgG memory, IgM memory, and long-lived plasma cells) evaluated for non-specificity using the **Polyspecific Reagent (PSR) assay**.

### Role in the Pipeline
**CRITICAL CLARIFICATION:** The Shehata dataset is an **EXTERNAL TEST SET ONLY** - it is NOT used for training.

**Training vs Testing:**
- **Training data:** Boughter dataset (>1000 antibodies, ELISA-based labels)
- **Test sets:**
  - Jain (137 clinical antibodies, ELISA)
  - **Shehata (398 antibodies, PSR assay)** ← We are here
  - Harvey (140K nanobodies, PSR assay)

**Why test on Shehata?**
1. **External validation** of models trained on Boughter
2. **Cross-assay generalization**: PSR vs ELISA (different non-specificity measurements)
3. **Biological diversity**: Different B cell subsets vs just antigen-specific antibodies
4. **Clinical relevance**: Human antibodies from various maturation stages

From the paper (Section 2.1):
> "the most balanced dataset (i.e. Boughter one) was selected for training of ML models, while the remaining three (i.e. Jain, Shehata and Harvey, which consists exclusively of VHH sequences) were used for testing."

---

## Paper's Preprocessing Procedure (Methods Section 4.3)

### Step-by-Step from Sakhnini et al. 2025:

1. **CDR Annotation** (Line 240):
   > "sequences were annotated in the CDRs using **ANARCI** following the **IMGT numbering scheme**"

2. **Fragment Assembly**:
   - Created **16 different antibody fragment sequences**
   - Fragments include: VH, VL, H-CDR1, H-CDR2, H-CDR3, L-CDR1, L-CDR2, L-CDR3, H-CDRs (combined), L-CDRs (combined), etc.

3. **Embedding** (Lines 240-241):
   - Used **ESM-1v** (top performer: 71% CV accuracy)
   - **Mean pooling**: "average of all token vectors"

4. **Label Preparation**:
   - **Binary classification**: Specific (class 0) vs Non-specific (class 1)
   - For Boughter training: 0 flags = specific, >3 flags = non-specific (excluded 1-3 flags)
   - For Shehata testing: PSR score-based labeling

### Key Finding for Shehata:
From Section 2.1 (Line 51):
> "the Shehata dataset is unbalanced, with **7 out of 398 antibodies characterised as non-specific only**"

**IMPORTANT:** The paper used a **strict threshold** for binary classification. Our analysis shows:
- PSR = 0: 251 antibodies (specific)
- PSR > 0: 147 antibodies (have some non-specificity)
- But only **~7 considered truly "non-specific"** by paper's threshold

We need to determine this threshold from the paper or their methods.

---

## Current Repository Architecture

### Existing Code Analysis:

**1. Preprocessing Pattern (`preprocessing/process_boughter.py`):**
- Uses `riot_na` (ANARCI wrapper) for CDR annotation
- Processes FASTA files with nucleotide sequences
- Outputs: Full sequence + individual CDR/FWR regions (amino acids)
- Format: CSV with columns like `cdr1_aa_H`, `fwr1_aa_H`, etc.

**2. Data Loading (`data.py`):**
- `load_local_data()`: Reads CSV files
- `load_hf_dataset()`: Reads from HuggingFace
- `preprocess_raw_data()`: Embeds sequences using ESM
- Standard format: `sequence, label` columns

**3. Existing Datasets:**
- `train_datasets/VH_only_training_ready.csv` → Boughter training data
- `test_datasets/jain.csv` → Jain test data (already processed)

**4. Expected Output Format (from `jain.csv`):**
```csv
id,heavy_seq,light_seq,label,source,smp,ova
abituzumab,QVQLQQSGGE...,DIQMTQSPSS...,0,jain2017,0.125938041,0.044483421
```

---

## Implementation Complexity Assessment

### Complexity Level: **MEDIUM** (2-phase implementation recommended)

**Why Medium?**
1. ✓ **Simple aspects:**
   - Raw data already available (mmc2.xlsx)
   - CDR annotations already in Excel file
   - Clear binary labels from PSR scores
   - CSV output format well-defined

2. ⚠️ **Complex aspects:**
   - Need to determine correct PSR threshold for binary labels
   - Must match paper's exact 16 fragment types
   - Integration with existing embedding pipeline
   - ANARCI re-annotation vs using provided annotations
   - Handling unbalanced dataset (7/398 non-specific)

---

## Proposed Implementation: 2-Phase Approach

### Phase 1: Basic Preprocessing Script (Minimal Viable Product)
**Goal:** Create test-ready CSV matching existing format
**Effort:** ~2-3 hours
**Deliverable:** `test_datasets/shehata.csv`

**Tasks:**
1. ✅ Read mmc2.xlsx
2. ✅ Extract VH and VL protein sequences
3. ✅ Determine PSR threshold for binary labels (investigate paper/supplement)
4. ✅ Create CSV with columns: `id, heavy_seq, light_seq, label, psr_score, b_cell_subset, source`
5. ✅ Basic validation (sequence quality, label distribution)
6. ✅ Document threshold decision in docstring

**Output Example:**
```csv
id,heavy_seq,light_seq,label,psr_score,b_cell_subset,source
ADI-38502,EVQLLESGGGLVKPGG...,DIVMTQSPSTLSASVG...,0,0.0,IgG memory,shehata2019
ADI-38501,EVQLLESGGGLVQPGG...,DIVMTQSPATLSLSPG...,0,0.023184,IgG memory,shehata2019
```

**Integration Point:**
- Works with existing `data.load_local_data()`
- Can be used by `test.py` for model evaluation
- Enables reproduction of Figure 3C-D from paper

---

### Phase 2: Full Fragment Extraction (Paper-Complete)
**Goal:** Generate all 16 antibody fragments for comprehensive testing
**Effort:** ~4-6 hours
**Deliverable:** `preprocessing/process_shehata.py` + multiple output CSVs

**Tasks:**
1. ✅ Re-annotate sequences using ANARCI (IMGT scheme) for consistency
2. ✅ Extract all 16 fragments:
   - VH, VL (full variable domains)
   - H-CDR1, H-CDR2, H-CDR3
   - L-CDR1, L-CDR2, L-CDR3
   - H-CDRs (concatenated)
   - L-CDRs (concatenated)
   - H-FWRs, L-FWRs
   - VH+VL
   - Other combinations tested in paper
3. ✅ Create separate CSV for each fragment type (like training data)
4. ✅ Match exact preprocessing from `process_boughter.py` style
5. ✅ Comprehensive validation against paper's reported statistics

**Output Structure:**
```
test_datasets/
├── shehata/
│   ├── VH_only_shehata.csv
│   ├── VL_only_shehata.csv
│   ├── H-CDR3_shehata.csv
│   ├── H-CDRs_shehata.csv
│   └── ... (16 total files)
```

**Integration Point:**
- Enables testing all 16 fragment-specific models
- Reproduces Table 4 results from paper
- Allows fragment-level performance comparison

---

## Decision Points & Open Questions

### 1. PSR Score Threshold for Binary Labels
**Question:** What PSR score constitutes "non-specific"?

**Options:**
- **A. PSR > 0** (147 non-specific) - Any polyreactivity
- **B. Paper's threshold** (~7 non-specific) - Need to find this value
- **C. Quartile-based** (PSR > 75th percentile ≈ 0.017)

**Action Required:**
- Search paper supplement for threshold
- Check original Shehata 2019 paper
- Email authors if unclear
- Document decision rationale

**Current Data:**
```
PSR Score Distribution:
Mean: 0.0346
Median: 0.0 (50% have PSR = 0)
75th percentile: 0.0169
Max: 0.711
```

### 2. Use Provided CDR Annotations vs Re-annotate with ANARCI?
**Question:** mmc2.xlsx already has CDR annotations. Re-annotate or use as-is?

**Options:**
- **A. Use provided annotations** - Faster, author-verified
- **B. Re-annotate with ANARCI** - Ensures consistency with Boughter processing

**Recommendation:**
- **Phase 1:** Use provided annotations (faster MVP)
- **Phase 2:** Re-annotate for consistency, compare results

### 3. Handle VH-only vs VH+VL sequences?
**Current state:** mmc2.xlsx has both VH and VL sequences

**From paper:** VH domain is primary predictor (71% accuracy)

**Recommendation:**
- Generate both VH-only and VH+VL versions
- Primary testing on VH-only (matches training data format)
- VL available for future bispecific/full-length analysis

---

## File Structure Plan

```
antibody_training_pipeline_ESM/
├── preprocessing/
│   ├── process_boughter.py (existing)
│   └── process_shehata.py (NEW - Phase 2)
│
├── scripts/ (NEW directory for utilities)
│   └── prepare_shehata_basic.py (NEW - Phase 1)
│
├── test_datasets/
│   ├── mmc2.xlsx (raw data - gitignored)
│   ├── mmc3.xlsx (clinical subset - gitignored)
│   ├── mmc4.xlsx (gitignored)
│   ├── mmc5.xlsx (gitignored)
│   ├── jain.csv (existing)
│   ├── shehata.csv (NEW - Phase 1 output)
│   └── shehata/ (NEW - Phase 2 outputs)
│       ├── VH_only_shehata.csv
│       ├── H-CDR3_shehata.csv
│       └── ... (other fragments)
│
└── docs/
    ├── shehata_preprocessing_implementation_plan.md (THIS FILE)
    └── shehata_preprocessing_validation.md (NEW - post-implementation)
```

---

## Integration with Existing Codebase

### Compatibility Checklist:

✅ **Data format:** CSV with `sequence, label` columns
✅ **Loading:** Works with `data.load_local_data(file_path, sequence_column, label_column)`
✅ **Testing:** Compatible with `test.py --data test_datasets/shehata.csv`
✅ **Embedding:** Sequences ready for ESM-1v processing
✅ **Naming:** Follows `{fragment}_{dataset}.csv` pattern

### Code Reuse Opportunities:

1. **From `process_boughter.py`:**
   - `annotate()` function → ANARCI annotation pattern
   - CDR/FWR extraction logic
   - Output formatting

2. **From `data.py`:**
   - `load_local_data()` → No changes needed
   - `preprocess_raw_data()` → Works with any sequence list

3. **From existing test datasets:**
   - `jain.csv` → Template for column names and format

---

## Validation Strategy

### Post-Implementation Validation:

1. **Data Integrity:**
   - ✓ 398 antibodies (or 402 if including extras)
   - ✓ No missing VH sequences
   - ✓ Valid amino acid sequences only
   - ✓ Label distribution matches paper

2. **Paper Reproduction:**
   - ✓ Can reproduce Figure 3C-D (Shehata PSR predictions)
   - ✓ Distribution of predicted probabilities matches paper
   - ✓ ~7 non-specific antibodies in binary classification

3. **Integration Testing:**
   - ✓ Loads successfully with `load_local_data()`
   - ✓ Embeddings generate without errors
   - ✓ Test script runs end-to-end
   - ✓ Metrics calculated correctly

4. **Cross-dataset Comparison:**
   - ✓ Format matches `jain.csv` structure
   - ✓ Sequence lengths in reasonable range
   - ✓ Label distribution documented

---

## Dependencies & Requirements

### Python Packages:
```python
pandas>=1.5.0          # Excel reading, CSV writing
openpyxl>=3.0.0        # Excel file support
numpy>=1.24.0          # Array operations
biopython>=1.79        # Sequence handling (if re-annotating)
anarci or riot_na      # CDR annotation (Phase 2 only)
```

### External Tools:
- **ANARCI** (Phase 2): CDR numbering tool
  - Installation: `pip install riot-na` or compile ANARCI from source
  - Alternative: Use provided annotations from mmc2.xlsx

---

## Timeline Estimate

### Phase 1 (Basic CSV):
- **Planning:** 1 hour (reading paper, this doc)
- **Coding:** 1-2 hours
- **Testing/Validation:** 1 hour
- **Total:** ~3-4 hours

### Phase 2 (Full fragments):
- **ANARCI setup:** 1 hour
- **Coding:** 3-4 hours
- **Testing/Validation:** 2 hours
- **Total:** ~6-7 hours

### Grand Total: 10-11 hours for complete implementation

---

## Success Criteria

### Phase 1 Complete When:
- [x] `test_datasets/shehata.csv` exists
- [x] 398 rows (one per antibody)
- [x] Columns: `id, heavy_seq, light_seq, label, psr_score, b_cell_subset, source`
- [x] Binary labels defined with documented threshold (0.31002 = 98.24%ile)
- [x] No missing VH/VL sequences
- [x] Loads successfully with existing data pipeline (validation passed)
- [x] README/documentation updated

### Phase 2 Complete When:
- [x] `preprocessing/process_shehata.py` script exists
- [x] 16 fragment-specific CSV files generated in test_datasets/shehata/
- [x] ANARCI re-annotation using IMGT scheme (riot_na v4.0.5)
- [x] All fragments validated for sequence quality (lengths match expected ranges)
- [x] Fragment CSVs have standardized format (id, sequence, label, psr_score, b_cell_subset, source)
- [ ] Can reproduce paper's Figure 3 results (requires model training/inference)
- [x] Comprehensive documentation of preprocessing choices

---

## Risks & Mitigation

### Risk 1: PSR Threshold Uncertainty
**Impact:** Wrong threshold → incorrect labels → invalid test results
**Mitigation:**
- Document assumption clearly
- Test with multiple thresholds
- Compare to paper's reported statistics (7/398)

### Risk 2: ANARCI Installation Issues
**Impact:** Can't complete Phase 2
**Mitigation:**
- Phase 1 uses provided annotations
- Document ANARCI as optional dependency
- Provide containerized environment (Docker)

### Risk 3: Format Incompatibility
**Impact:** Preprocessed data doesn't work with existing pipeline
**Mitigation:**
- Follow `jain.csv` as exact template
- Integration test early
- Ask maintainer (@ludocomito) for format validation

---

## Next Steps

### Before Implementation:
1. ✅ **READ THIS PLAN** - Understand scope and approach
2. 🔲 **Clarify PSR threshold** - Search paper supplement or ask maintainer
3. 🔲 **Choose Phase 1 or both** - Decide scope based on timeline
4. 🔲 **Set up test environment** - Install dependencies, validate Excel files

### During Implementation:
1. 🔲 Start with Phase 1 (get quick win)
2. 🔲 Test with existing pipeline immediately
3. 🔲 Document decisions as you go
4. 🔲 Validate against paper statistics continuously

### After Implementation:
1. 🔲 Create PR with preprocessed CSV(s)
2. 🔲 Write validation report (new doc)
3. 🔲 Update main README with Shehata dataset info
4. 🔲 Close Issue #3
5. 🔲 Consider contributing to Issues #1 (HuggingFace migration) and #2 (Jain preprocessing)

---

## References

1. **Sakhnini et al. 2025** - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"
   - bioRxiv: 10.1101/2025.04.28.650927
   - Local: `literature/markdown/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md`

2. **Shehata et al. 2019** - "Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability"
   - Cell Reports: 10.1016/j.celrep.2019.08.056
   - Supplementary: mmc2.xlsx (main dataset)

3. **Repository Issues:**
   - Issue #3: Shehata preprocessing (this task)
   - Issue #2: Jain preprocessing (reference)
   - Issue #1: HuggingFace migration (future)

---

## Appendix: Raw Data Inventory

### mmc2.xlsx (Main Dataset):
- **Rows:** 402 antibodies (398 reported + 4 extras)
- **Columns:** 25 total
  - Clone name
  - B cell subset (Naïve, IgG memory, IgM memory, LLPCs)
  - Biophysical: TmApp, HIC retention time
  - PSR Score (non-specificity label)
  - Sequences: VH/VL Protein + DNA
  - CDR/FWR regions: Pre-annotated (H-CDR1-3, L-CDR1-3, FR1-4)
  - Germline info

### Other Files:
- **mmc3.xlsx:** 42 clinical antibodies (subset)
- **mmc4.xlsx:** 10 antibodies with mutation data
- **mmc5.xlsx:** 26 antibodies with HIC focus

**Decision:** Use mmc2.xlsx as primary source for all preprocessing.

---

**END OF IMPLEMENTATION PLAN**

**Status:** Ready for review and decision on Phase 1 vs Phase 1+2 implementation.
