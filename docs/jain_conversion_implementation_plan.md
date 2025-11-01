# Jain Dataset Excel-to-CSV Conversion Implementation Plan

**Issue:** #2 - Jain dataset preprocessing
**Author:** Ray (ray/learning branch)
**Date:** 2025-11-01
**Status:** Phase 1 (Excel-to-CSV conversion) - Planning

---

## Executive Summary

### What is the Jain Dataset?
The Jain dataset (Jain et al. 2017, PNAS) contains **137 clinical-stage IgG1-formatted antibodies** evaluated for non-specificity using **ELISA with a panel of common antigens** (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH).

### Role in the Pipeline
The Jain dataset is an **EXTERNAL TEST SET ONLY** - it is NOT used for training.

**Training vs Testing:**
- **Training data:** Boughter dataset (>1000 antibodies, ELISA-based labels)
- **Test sets:**
  - **Jain (137 clinical antibodies, ELISA)** ← We are here
  - Shehata (398 antibodies, PSR assay)
  - Harvey (140K nanobodies, PSR assay)

**Why test on Jain?**
1. **External validation** of models trained on Boughter
2. **Clinical relevance**: Therapeutic antibodies at various clinical stages
3. **Same assay as training data**: ELISA (vs PSR for Shehata/Harvey)
4. **Balanced dataset**: Similar distribution to Boughter (specific/mildly non-specific/non-specific)

From Sakhnini paper (Section 2.1):
> "the most balanced dataset (i.e. Boughter one) was selected for training of ML models, while the remaining three (i.e. Jain, Shehata and Harvey, which consists exclusively of VHH sequences) were used for testing."

---

## Data Provenance Issue Discovered

### The Problem:
The existing `test_datasets/jain.csv` in the repository:
- Contains only **80 antibodies** (should be 137)
- Has different SMP values than PNAS supplementary files:
  - Example: abituzumab `smp` in `jain.csv`: 0.126
  - PNAS SD03 `PSR SMP Score`: 0.167
- 29 antibodies in `jain.csv` are NOT in PNAS files
- Unknown origin - may have been incorrectly converted or from different source

### The Solution:
Convert the **authoritative PNAS supplementary files** directly:
1. `pnas.1616408114.sd01.xlsx` - Metadata (137 antibodies)
2. `pnas.1616408114.sd02.xlsx` - VH/VL sequences (137 antibodies)
3. `pnas.1616408114.sd03.xlsx` - Biophysical measurements (139 entries, 137 match)

Create `jain_v2.csv` (or replace `jain.csv`) with correctly sourced data.

---

## Paper's Preprocessing Procedure (from Sakhnini et al. 2025)

### Labeling System (Methods Section 4.3, Line 236):

**ELISA Flag-Based Labels:**
From Section 2.2 (Line 55):
> "the Boughter dataset was first parsed into two groups: specific (0 flags) and non-specific group (>3 flags), leaving out the mildly non-specific antibodies (1-3 flags)"

**For Jain dataset** (same labeling logic):
- **Class 0 (Specific)**: 0 ELISA flags
- **Class 1 (Non-specific)**: >3 ELISA flags (≥4 flags)
- **Excluded from training**: 1-3 flags (mildly non-specific)

**ELISA Panel (6 ligands):**
- ssDNA (single-stranded DNA)
- dsDNA (double-stranded DNA)
- Insulin
- LPS (lipopolysaccharide)
- Cardiolipin
- KLH (keyhole limpet hemocyanin)

**Flag Counting:**
- Each antibody tested against all 6 ligands
- Each positive result = 1 flag
- Total flags range: 0 (perfectly specific) to 6 (highly non-specific)

### Required Output Format:

Based on existing `jain.csv` structure:
```csv
id,heavy_seq,light_seq,label,source,smp,ova
abituzumab,QVQLQQSGGE...,DIQMTQSPSS...,0,jain2017,0.166666,1.137375
```

**Column Definitions:**
- `id`: Antibody name (from SD01/SD02/SD03 'Name' column)
- `heavy_seq`: VH sequence (from SD02 'VH' column)
- `light_seq`: VL sequence (from SD02 'VL' column)
- `label`: Binary classification
  - 0 = Specific (0 ELISA flags)
  - 1 = Non-specific (≥4 ELISA flags)
  - **Note:** Mildly non-specific (1-3 flags) should be INCLUDED but may be filtered later during preprocessing
- `source`: "jain2017"
- `smp`: PSR SMP Score from SD03 'Poly-Specificity Reagent (PSR) SMP Score (0-1)' column
- `ova`: **TBD** - Need to determine which SD03 column maps to 'ova'
  - Possible candidates: 'ELISA', 'BVP ELISA'
  - Existing `jain.csv` has ova range: -0.002 to 1.274
  - PNAS 'ELISA' range: 0.889 to 14.459
  - PNAS 'BVP ELISA' range: 1.028 to 22.746
  - **Hypothesis**: May need log transformation or normalization

### ✅ RESOLVED: ELISA Flag Derivation

**BREAKTHROUGH (2025-11-01):** Successfully determined flag derivation methodology from Jain et al. 2017 PNAS paper!

#### Flag Derivation Methodology (Table 1)

**Threshold Calculation:**
- Thresholds are set at **90th percentile** of **APPROVED antibodies only** (48 antibodies)
- Represents "10% worst values among approved drugs" (Lipinski approach)
- Each assay group contributes **0 or 1 flag** to total flag count

**Assay Groups (4 groups = max 4 flags):**

1. **Polyreactivity Group (ELISA, BVP):**
   - ELISA threshold: **1.9** (calculated: 1.883 from approved 90th percentile)
   - BVP threshold: **4.3** (calculated: 4.278 from approved 90th percentile)
   - **Flag = 1 if ELISA > 1.9 OR BVP > 4.3**

2. **Self-Interaction Group (PSR, CSI, AC-SINS, CIC):**
   - PSR SMP threshold: **0.26** (calculated: 0.263 from approved 90th percentile)
   - **Flag = 1 if any assay in group exceeds threshold**

3. **Chromatography Group (SGAC100, SMAC, HIC):**
   - Thresholds TBD (need full Table 1 from main paper)
   - **Flag = 1 if any assay in group exceeds threshold**

4. **Stability Group (AS):**
   - AS threshold TBD
   - **Flag = 1 if AS exceeds threshold**

**Total Flags per Antibody:**
- Range: 0 to 4 flags
- 0 flags = Specific
- 1-3 flags = Mildly non-specific
- ≥4 flags = Non-specific (for binary classification)

**Source References:**
- Jain et al. 2017 PNAS, Table 1 (main paper)
- PNAS Supporting Information PDF, Page 2 (assay grouping methodology)
- Web search confirmation of ELISA/BVP thresholds

**Status:** ELISA/BVP/PSR thresholds confirmed. Need to extract remaining 9 assay thresholds from main paper Table 1.

---

## Implementation Plan

### Phase 1: Excel-to-CSV Conversion

**Goal:** Convert PNAS supplementary files into `test_datasets/jain_v2.csv`

**Input files:**
- `test_datasets/pnas.1616408114.sd01.xlsx` (metadata)
- `test_datasets/pnas.1616408114.sd02.xlsx` (sequences)
- `test_datasets/pnas.1616408114.sd03.xlsx` (biophysical properties)

**Output file:**
- `test_datasets/jain_v2.csv` (or `jain.csv` replacement)

**Script:** `scripts/convert_jain_excel_to_csv.py`

#### Step-by-Step Process:

1. **Load all three Excel files**
   ```python
   sd01 = pd.read_excel('test_datasets/pnas.1616408114.sd01.xlsx')  # Metadata
   sd02 = pd.read_excel('test_datasets/pnas.1616408114.sd02.xlsx')  # Sequences
   sd03 = pd.read_excel('test_datasets/pnas.1616408114.sd03.xlsx')  # Properties
   ```

2. **Merge on 'Name' column**
   - SD01 ⋈ SD02 on 'Name' → Get sequences with metadata
   - Result ⋈ SD03 on 'Name' → Add biophysical properties
   - **Expected result:** 137 antibodies (SD03 has 139 entries, 2 are metadata rows)

3. **Extract required columns**
   ```python
   jain_df = pd.DataFrame({
       'id': merged['Name'],
       'heavy_seq': merged['VH'],
       'light_seq': merged['VL'],
       'label': None,  # TBD - need ELISA flag derivation logic
       'source': 'jain2017',
       'smp': merged['Poly-Specificity Reagent (PSR) SMP Score (0-1)'],
       'ova': merged['ELISA']  # Or 'BVP ELISA' or transformed version - TBD
   })
   ```

4. **Handle label derivation**
   - **Option A:** Leave as NaN and document as requiring manual annotation
   - **Option B:** Use threshold on ELISA score (need to determine threshold)
   - **Option C:** Check if Jain paper supplementary has flag data
   - **Recommendation:** Start with Option A, document issue

5. **Data validation**
   - Check for missing sequences
   - Verify all sequences are valid amino acids
   - Check for duplicates
   - Validate lengths (reasonable VH/VL lengths)

6. **Save to CSV**
   ```python
   jain_df.to_csv('test_datasets/jain_v2.csv', index=False)
   ```

7. **Generate conversion report**
   - Document antibody count
   - Document any missing data
   - Document label derivation approach
   - Compare with existing `jain.csv`

### Phase 2: Validation

**Script:** `scripts/validate_jain_conversion.py`

**Validation checks:**
1. **Count validation:** Confirm 137 antibodies
2. **Sequence validation:**
   - All VH/VL sequences present
   - No invalid amino acids
   - Reasonable length distributions
3. **Merge validation:** All SD01/SD02/SD03 names matched
4. **Comparison with existing `jain.csv`:**
   - Which antibodies overlap?
   - Which antibodies are new?
   - Which antibodies are missing?
   - How do SMP values differ?

### Phase 3: Documentation

**Create:**
1. `docs/jain_conversion_verification_report.md` - Validation results
2. `docs/jain_data_cleaning_log.md` - Any manual fixes applied
3. Update `docs/jain_data_sources.md` with final resolution

---

## Open Questions / Blockers

### Critical:
1. **How to derive ELISA flags from PNAS data?**
   - PNAS SD03 has continuous ELISA scores, not discrete flags
   - Sakhnini paper requires flag-based labeling (0, 1-3, ≥4)
   - Need to check original Jain 2017 paper for flag data

2. **What is the 'ova' column?**
   - Existing `jain.csv` has 'ova' but origin unclear
   - Possible mappings: ELISA, BVP ELISA, or derived metric
   - Ranges don't match directly - may need transformation

### Non-critical:
3. Why does existing `jain.csv` have only 80 antibodies instead of 137?
4. Why do SMP values differ between `jain.csv` and PNAS files?

---

## Success Criteria

**Phase 1 Complete when:**
- ✅ `scripts/convert_jain_excel_to_csv.py` successfully runs
- ✅ Generates `test_datasets/jain_v2.csv` with 137 antibodies
- ✅ All sequences validated (no missing, no invalid amino acids)
- ✅ Conversion report documents any assumptions/limitations
- ✅ Validation script confirms data quality

**Phase 2 (Fragment Extraction) will follow** using existing `preprocessing/process_jain.py` (already implemented in PR #17).

---

## References

1. **Jain et al. 2017** - "Biophysical properties of the clinical-stage antibody landscape" PNAS 114(5):944-949. DOI: 10.1073/pnas.1616408114
2. **Sakhnini et al. 2025** - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"
3. **Boughter et al. 2020** - eLife 9:e61393 (original ELISA flag-based labeling approach)
