# Harvey Dataset – Preprocessing Implementation Plan

**Date:** 2025-11-01 (Updated: 2025-11-06)
**Issue:** #4 – Harvey dataset preprocessing
**Status:** ✅ **COMPLETE - Implementation validated and pipeline operational**

---

## Implementation Status (2025-11-06)

**UPDATE:** Implementation is complete and fully validated. Data source confirmed as the official Harvey Lab repository (`debbiemarkslab/nanobody-polyreactivity`), NOT the HuggingFace ZYMScott/polyreaction dataset.

**Pipeline status:** All preprocessing scripts operational and verified. See `test_datasets/harvey/README.md` for current SSOT.

---

## Objective

Extract nanobody (VHH) fragments from the Harvey dataset following **Sakhnini et al. 2025 methodology** to enable testing of ESM-1v based polyreactivity prediction models.

---

## Methodology Reference: Sakhnini et al. 2025

### Model Architecture (Section 2.3, line 71)

From `literature/markdown/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md`:

> "To show which antibody fragment contributed most to non-specificity, we annotated the CDRs using ANARCI in the IMGT numbering scheme and trained 12 different antibody fragment-specific binary classification models (see **Table 4**). Overall, all of the protein language models (PLMs) performed well with 66-71% 10-fold CV accuracy."

**Table 4 Fragment Types (line 380-388):**
```
1. VH (full heavy variable domain)
2. VL (full light variable domain)
3. H-CDRs (concatenated H-CDR1+2+3)
4. L-CDRs (concatenated L-CDR1+2+3)
5. H-FWRs (concatenated H-FWR1+2+3+4)
6. L-FWRs (concatenated L-FWR1+2+3+4)
7. VH+VL (paired variable domains)
8. All-CDRs (H-CDRs + L-CDRs)
9. All-FWRs (H-FWRs + L-FWRs)
10. Full (VH + VL)
11. VH H-CDR3
12. VL L-CDR3
```

### Top Performing Model

From Section 2.3 (line 73):
> "**The highest PLM-based predictability is achieved by encoding the VH domain**. Across different validation procedures, the **VH-based classifier** demonstrated the best performance with a mean accuracy of 71% in 10-fold CV."

**Critical Finding for Harvey:**
Harvey dataset consists of **nanobodies (VHH)** - single heavy-chain variable domains with NO light chain.

Therefore, for Harvey we extract **VHH-specific fragments only**:
1. VHH (full nanobody = equivalent to VH)
2. H-CDR1
3. H-CDR2
4. H-CDR3
5. H-CDRs (concatenated CDR1+2+3)
6. H-FWRs (concatenated FWR1+2+3+4)

**NO light chain fragments** (L-CDR1/2/3, L-FWRs, VL, etc.)

---

## Harvey Dataset Testing in Sakhnini

### Usage Context (Section 2.7, line 131)

> "To find out whether our ESM 1v mean-mode **VH-based LogisticReg model** can extend its applicability further to the non-specificity scored by the PSR assay, the Shehata dataset and the **VH-based Nb dataset by Harvey and co-authors** [\[45\]], here referred to as the Harvey dataset, were tested."

**Key Points:**
- Harvey used as **test set only** (NOT for training)
- Model was **trained on Boughter dataset** (~1000 antibodies)
- Testing evaluates **generalization to nanobodies**

### Performance Results (Section 2.7, line 131-132)

> "A similar forecast was observed for the Harvey dataset; all the specific PSR-scored Nbs resulted in a broad probability distribution, while the non-specific PSR-scored ones resulted in a narrower probability distribution towards higher non-specificity (**Figure 3E,F**)."

**Interpretation:**
- Model predicts **high polyreactivity** nanobodies better than low
- PSR assay spectrum differs from ELISA (used for Boughter training)
- Still provides useful signal for nanobody polyreactivity prediction

---

## Input Data

### Source File
**Path:** `test_datasets/harvey/processed/harvey.csv`
**Rows:** 141,474 nanobodies (141,475 with header)
**Downloaded:** 2025-11-01 from HuggingFace `ZYMScott/polyreaction`

### Current Column Structure
```
seq                 : Full nanobody VHH sequence (52-137 aa)
CDR1_nogaps         : H-CDR1 sequence (pre-extracted, no gaps)
CDR2_nogaps         : H-CDR2 sequence (pre-extracted, no gaps)
CDR3_nogaps         : H-CDR3 sequence (pre-extracted, no gaps)
label               : Binary polyreactivity (0=low, 1=high)
```

### ⚠️ CRITICAL: NO CDR LENGTH FILTERING

**Harvey's original filter** (from Harvey et al. 2022 line 142):
- CDR1 length == 8
- CDR2 length == 8 OR 9
- CDR3 length between 6-22
- **Result:** 134,302 sequences (65,147 low + 69,155 high)

**Novo Nordisk's approach** (from Sakhnini et al. 2025):
- **NO CDR length filtering**
- Used **all 141,474 sequences** from HuggingFace
- Cited as ">140,000 naïve nanobodies" in Table 4

**Our Implementation:**
- ✅ Process **ALL 141,474 sequences** with NO filtering
- ✅ Match Novo Nordisk methodology exactly
- ✅ Broader coverage than Harvey's original training set
- ⚠️ Some sequences may have CDR lengths outside Harvey's constraints

### Data Quality Notes

**From HuggingFace dataset inspection:**
- Label distribution: 69,702 low (49.3%), 71,772 high (50.7%) - **balanced**
- Sequence length: 52-137 aa (typical nanobody VHH range: 110-130 aa)
- No missing values observed
- CDRs already extracted but **numbering scheme unknown**

**Question:** Are HuggingFace CDRs IMGT-numbered?
**Answer:** Unknown - **safer to re-extract using ANARCI** for consistency

---

## Processing Pipeline

### Step 1: ANARCI Annotation (IMGT Scheme)

**Tool:** `riot_na.create_riot_aa()`
**Numbering:** IMGT (consistent with Jain/Shehata preprocessing)

For each nanobody sequence in `harvey.csv`:
1. Run `annotator.run_on_sequence(seq_id, sequence)`
2. Extract VHH fragments:
   - `fwr1_aa_H`: Framework 1
   - `cdr1_aa_H`: CDR1
   - `fwr2_aa_H`: Framework 2
   - `cdr2_aa_H`: CDR2
   - `fwr3_aa_H`: Framework 3
   - `cdr3_aa_H`: CDR3
   - `fwr4_aa_H`: Framework 4
   - `sequence_alignment_aa`: Full aligned VHH

3. Create concatenated fragments:
   - `H-CDRs`: CDR1 + CDR2 + CDR3 (no separators)
   - `H-FWRs`: FWR1 + FWR2 + FWR3 + FWR4 (no separators)

### Step 2: Fragment CSV Generation

Create directory: `test_datasets/harvey/fragments/`

Generate 6 fragment CSV files:

1. **VHH_only_harvey.csv**
   - Columns: `id, sequence, label, source, sequence_length`
   - Sequence: Full VHH from ANARCI alignment
   - 141,474 rows

2. **H-CDR1_harvey.csv**
   - Columns: `id, sequence, label, source, sequence_length`
   - Sequence: H-CDR1 only
   - 141,474 rows

3. **H-CDR2_harvey.csv**
   - Columns: `id, sequence, label, source, sequence_length`
   - Sequence: H-CDR2 only
   - 141,474 rows

4. **H-CDR3_harvey.csv**
   - Columns: `id, sequence, label, source, sequence_length`
   - Sequence: H-CDR3 only
   - 141,474 rows

5. **H-CDRs_harvey.csv**
   - Columns: `id, sequence, label, source, sequence_length`
   - Sequence: CDR1+CDR2+CDR3 concatenated
   - 141,474 rows

6. **H-FWRs_harvey.csv**
   - Columns: `id, sequence, label, source, sequence_length`
   - Sequence: FWR1+FWR2+FWR3+FWR4 concatenated
   - 141,474 rows

**Column Definitions:**
- `id`: Unique identifier (row index or nanobody name if available)
- `sequence`: Extracted fragment sequence
- `label`: Binary polyreactivity (0=low, 1=high) from input
- `source`: "harvey2022" (dataset provenance)
- `sequence_length`: Length of fragment in amino acids

### Step 3: Validation

**Validation Script:** `scripts/validate_harvey_processing.py`

**Checks:**
1. **Row count consistency:** All 6 fragment files have 141,474 rows
2. **No missing sequences:** No empty/null sequences
3. **Sequence composition:** Only valid amino acids (ACDEFGHIKLMNPQRSTVWY)
4. **Label preservation:** Binary labels (0/1) match input `harvey.csv`
5. **Fragment relationships:**
   - CDR1+CDR2+CDR3 concatenated = H-CDRs
   - FWR1+FWR2+FWR3+FWR4 concatenated = H-FWRs
   - All fragments extracted from same VHH sequence
6. **Length distributions:**
   - VHH: ~110-130 aa (nanobody typical range)
   - CDR1: ~8-12 aa (IMGT H-CDR1 range)
   - CDR2: ~7-10 aa (IMGT H-CDR2 range)
   - CDR3: ~10-20 aa (IMGT H-CDR3 range, longer in nanobodies)

---

## Implementation Files

### Primary Script

**File:** `preprocessing/harvey/step2_extract_fragments.py`
**Purpose:** Extract VHH fragments using ANARCI (IMGT numbering)
**Dependencies:**
- `pandas`: DataFrame manipulation
- `riot_na`: ANARCI wrapper for antibody annotation
- `tqdm`: Progress bar for 141K sequences

**Usage:**
```bash
python3 preprocessing/harvey/step2_extract_fragments.py
```

**Expected Runtime:** ~10-30 minutes (141K sequences × ANARCI annotation)

### Validation Script

**File:** `scripts/validate_harvey_processing.py`
**Purpose:** Verify fragment extraction correctness
**Checks:** Row counts, sequence composition, label preservation, fragment relationships

**Usage:**
```bash
python3 scripts/validate_harvey_processing.py
```

---

## Output Structure

```
test_datasets/harvey/fragments/
├── VHH_only_harvey.csv       (141,474 rows)
├── H-CDR1_harvey.csv          (141,474 rows)
├── H-CDR2_harvey.csv          (141,474 rows)
├── H-CDR3_harvey.csv          (141,474 rows)
├── H-CDRs_harvey.csv          (141,474 rows)
└── H-FWRs_harvey.csv          (141,474 rows)
```

**Total:** 6 fragment files (nanobody-specific, no light chain)

---

## Comparison with Jain/Shehata Preprocessing

### Similarities

1. **ANARCI annotation** with IMGT numbering scheme
2. **Fragment extraction** for ESM-1v embedding
3. **CSV output** with standardized columns
4. **Validation** of row counts and sequence composition

### Differences

| Aspect | Jain/Shehata | Harvey |
|--------|--------------|--------|
| **Antibody Type** | Full IgG (VH+VL) | Nanobody (VHH only) |
| **Chain Types** | Heavy + Light (2 chains) | Heavy only (1 chain) |
| **Fragment Count** | 16 files (H+L combinations) | 6 files (H only) |
| **Dataset Size** | 137-398 sequences | 141,474 sequences |
| **Label Type** | Multi-flag ELISA (0-6) | Binary PSR (0/1) |
| **Source** | Excel (SD01-SD03) | HuggingFace CSV |
| **Use Case** | Test set | Test set |

### File Mapping

**Jain/Shehata fragments NOT applicable to Harvey:**
- ❌ VL_only (no light chain)
- ❌ L-CDR1/2/3 (no light chain)
- ❌ L-CDRs, L-FWRs (no light chain)
- ❌ VH+VL (nanobody = VHH only)
- ❌ All-CDRs, All-FWRs (would just be H-CDRs, H-FWRs)
- ❌ Full (same as VHH)

**Harvey-specific fragments (VHH analogs):**
- ✅ VHH_only ≈ VH_only (full variable domain)
- ✅ H-CDR1/2/3 (same as Jain/Shehata)
- ✅ H-CDRs (same as Jain/Shehata)
- ✅ H-FWRs (same as Jain/Shehata)

---

## Edge Cases & Error Handling

### ANARCI Annotation Failures

**Possible causes:**
- Invalid sequence (non-amino acid characters)
- Sequence too short/long for nanobody
- ANARCI cannot assign numbering

**Handling:**
1. Log failed sequences with IDs
2. Skip failed sequences (continue processing)
3. Report failure count and IDs in summary
4. **Do NOT halt** entire pipeline for individual failures

**Expected failure rate:** <1% (based on Jain/Shehata experience; actual run on 2025-11-01 saw 453 failures = 0.32%)

### Sequence Length Outliers

**Nanobody expected range:** 110-130 aa
**Dataset actual range:** 52-137 aa (from HuggingFace inspection)

**Short sequences (< 100 aa):**
- Possibly truncated or incomplete nanobodies
- ANARCI may fail or produce incomplete annotation
- **Action:** Annotate anyway, log if ANARCI fails

**Long sequences (> 140 aa):**
- Possibly includes linkers or tags
- ANARCI should handle (will annotate VHH domain)
- **Action:** Annotate normally

### Label Verification

**Input labels:** 0 (low polyreactivity), 1 (high polyreactivity)
**Expected distribution:** ~50/50 (balanced dataset)

**Validation:**
- Ensure labels are binary (0 or 1 only)
- Check distribution remains balanced after processing
- Preserve original labels (no transformation)

---

## Documentation Deliverables

After implementation:

1. **harvey_data_cleaning_log.md**
   - ANARCI failures and resolutions
   - Sequence outliers and handling decisions
   - Label distribution verification

2. **harvey_preprocessing_verification_report.md**
   - Validation results (row counts, composition, etc.)
   - Fragment length distributions
   - Comparison with HuggingFace CDRs (if available)
   - SHA256 hashes for reproducibility

3. **Update README.md**
   - Add Harvey dataset to preprocessing section
   - Document fragment file structure
   - Link to Sakhnini et al. 2025 methodology

---

## Testing Strategy

### Unit Tests (Optional but Recommended)

**Test file:** `tests/test_harvey_processing.py`

1. **Test ANARCI annotation:**
   - Known nanobody sequence → expected CDR/FWR split
   - Invalid sequence → graceful error handling

2. **Test fragment concatenation:**
   - CDR1+CDR2+CDR3 = H-CDRs
   - FWR1+FWR2+FWR3+FWR4 = H-FWRs

3. **Test CSV generation:**
   - Column names match specification
   - No missing values
   - Labels preserved

### Integration Test

**Process sample:** First 1000 sequences from `harvey.csv`
**Verify:**
- All 6 fragment files created
- 1000 rows in each file
- No ANARCI failures (or <1%)
- Labels match input

### Full Pipeline Test

**Process:** All 141,474 sequences
**Verify:** All validation checks pass (validate_harvey_processing.py)
**Benchmark:** Runtime, memory usage, failure rate

---

## Approval Required Before Implementation

**Questions to confirm:**

1. ✅ Use **combined harvey.csv** (all 141K) or **splits** (train/val/test)?
   - **Recommendation:** Combined (matches Novo Nordisk test set usage)

2. ✅ Re-extract CDRs with **ANARCI** or trust **HuggingFace CDRs**?
   - **Recommendation:** Re-extract (ensures IMGT consistency with Jain/Shehata)

3. ✅ Generate **6 fragment files** (VHH-specific) or more?
   - **Recommendation:** 6 files (matches nanobody structure)

4. ✅ Handle ANARCI failures by **skipping** or **halting**?
   - **Recommendation:** Skip and log (avoid pipeline failure)

5. ✅ Expected runtime: **10-30 minutes** acceptable?
   - **Recommendation:** Yes (141K sequences is large but manageable)

---

## Implementation Timeline

**Phase 1: Script Development** (30-60 min)
- Write `preprocessing/harvey/step2_extract_fragments.py`
- Adapt from `preprocess_jain_p5e_s2.py` (remove light chain logic)

**Phase 2: Test Run** (10-30 min)
- Process first 1000 sequences
- Verify output format
- Check ANARCI failure rate

**Phase 3: Full Processing** (10-30 min)
- Process all 141,474 sequences
- Generate 6 fragment CSV files

**Phase 4: Validation** (5-10 min)
- Run `validate_harvey_processing.py`
- Verify all checks pass

**Phase 5: Documentation** (30-60 min)
- Write cleaning log
- Write verification report
- Update README

**Total estimated time:** 1.5-3 hours

---

## References

- [Sakhnini et al. 2025] Sakhnini LI, et al. Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv* (2025). https://doi.org/10.1101/2025.04.28.650927

- [Harvey et al. 2022] Harvey EP, et al. An in silico method to assess antibody fragment polyreactivity. *Nat Commun* 13, 7554 (2022). https://doi.org/10.1038/s41467-022-35276-4

- [ANARCI] Dunbar J, Deane CM. ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics* (2016). https://doi.org/10.1093/bioinformatics/btv552

- [IMGT] Lefranc MP, et al. IMGT unique numbering for immunoglobulin and T cell receptor variable domains. *Dev Comp Immunol* (2003).
