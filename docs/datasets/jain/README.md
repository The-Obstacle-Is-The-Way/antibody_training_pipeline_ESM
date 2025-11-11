# Jain Dataset Documentation

**Purpose:** Reference documentation for the Jain 2017 clinical antibody dataset processing and methodology.

**Dataset:** Jain et al. (2017) PNAS - 137 clinical-stage antibodies with biophysical measurements and polyreactivity data

---

## Quick Start

### For Implementation
**→ See [`preprocessing/jain/README.md`](../../../preprocessing/jain/README.md)** - SINGLE SOURCE OF TRUTH for pipeline implementation, scripts, and usage

### For Understanding the Dataset
Start here with this directory's documentation ↓

---

## Documentation Structure

### Core Reference Documents

#### 1. label_discrepancy_findings.md ✅ **ACCURATE**
**Purpose:** Critical investigation report documenting the 38.7% label error bug and fix
- Discovered mismatch between ELISA-based SSOT and paper-based flags_total
- Root cause: fragments derived from wrong labeling system
- Fix: step3_extract_fragments.py regeneration with ELISA-based labels
- Before: 67/27/43 (wrong) → After: 94/22/21 (correct)

**Read this if:** You want to understand why labels changed or the quality control process

---

#### 2. complete_guide.md ⚠️ **PARTIALLY OUTDATED**
**Purpose:** Comprehensive guide to Jain dataset processing
- **WARNING:** Contains sections describing RETIRED 94→86 methodology
- Current sections on P5e-S2 method ARE accurate
- Needs major rewrite to remove legacy VH length outlier references

**Read this if:** You want comprehensive context (but verify against preprocessing code)

---

#### 3. complete_history.md ⚠️ **PARTIALLY OUTDATED**
**Purpose:** Historical reference showing evolution of methodologies
- **WARNING:** Describes multiple methodologies including RETIRED approaches
- Contains valuable historical context
- Distribution counts need updating (shows 67/27/40 instead of 94/22/21)

**Read this if:** You need historical context on why we changed methodologies

---

#### 4. reorganization_complete.md ⚠️ **PARTIALLY OUTDATED**
**Purpose:** File reorganization effort from 2025-11-05
- Directory structure (raw/processed/canonical/fragments) is accurate
- **NOTE:** Fragments were regenerated on 2025-11-06 (AFTER this reorg doc)
- Verification section tests OLD methodology

**Read this if:** You want to understand the file organization

---

#### 5. data_sources.md ⚠️ **PARTIALLY OUTDATED**
**Purpose:** Source file descriptions and data provenance
- Source file info (SD01, SD02, SD03) is correct
- **WARNING:** Claims we're missing private ELISA data (we HAVE it!)
- **WARNING:** Describes flags_total methodology (RETIRED)
- Needs major update to describe ELISA-only methodology

**Read this if:** You need to know where the raw data comes from (but verify methodology against code)

---

### Archived Documentation

See [`archive/README.md`](archive/README.md) for:
- JAIN_QC_REMOVALS_COMPLETE.md (describes non-existent 94→86 VH length filtering)
- JAIN_REPLICATION_PLAN.md (describes wrong 10-flag methodology)
- jain_conversion_verification_report.md (describes retired flags_total system)

---

## Current Pipeline (SSOT)

**Implemented in:** `preprocessing/jain/` scripts

```
Step 1: Excel → CSV Conversion (ELISA-only)
  Raw: test_datasets/jain/raw/*.xlsx
  ↓ preprocessing/jain/step1_convert_excel_to_csv.py
  Output: test_datasets/jain/processed/jain_with_private_elisa_FULL.csv (137 antibodies)
          test_datasets/jain/processed/jain_ELISA_ONLY_116.csv (116 antibodies)

Step 2: P5e-S2 Novo Parity Method
  Input: jain_ELISA_ONLY_116.csv (116)
  ↓ preprocessing/jain/step2_preprocess_p5e_s2.py
    - Reclassify 5 specific→non-specific (PSR/Tm/clinical)
    - Remove 30 specific by PSR/AC-SINS sorting
  Output: test_datasets/jain/canonical/jain_86_novo_parity.csv (86 antibodies)

Step 3: Fragment Extraction (ANARCI/IMGT)
  Input: jain_with_private_elisa_FULL.csv (137)
  ↓ preprocessing/jain/step3_extract_fragments.py
  Output: test_datasets/jain/fragments/*.csv (16 fragment types)
```

---

## Key Facts (Current Implementation)

### Pipeline Statistics
- **Raw input:** 137 antibodies with private ELISA data
- **ELISA filtering:** 116 antibodies (excludes ELISA 1-3 flags)
- **P5e-S2 parity:** 86 antibodies (59 specific / 27 non-specific)
- **Novo parity:** 66.28% accuracy, confusion matrix [[40,19],[10,17]] - **EXACT MATCH**

### Label Distribution (ELISA-based SSOT)
- **Specific (0):** 94 antibodies
- **Non-specific (1):** 22 antibodies
- **Mild (NaN):** 21 antibodies (excluded from training)

### 16 Fragment Types
1. VH_only, VL_only (full variable domains)
2. H-CDR1, H-CDR2, H-CDR3 (heavy chain CDRs)
3. L-CDR1, L-CDR2, L-CDR3 (light chain CDRs)
4. H-CDRs, L-CDRs (concatenated CDRs)
5. H-FWRs, L-FWRs (concatenated frameworks)
6. VH+VL (paired variable domains)
7. All-CDRs, All-FWRs (all concatenated)
8. Full (alias for VH+VL)

---

## References

### Primary Papers
- **Jain et al. (2017)** - "Biophysical properties of the clinical-stage antibody landscape." *PNAS* 114(5):944-949. DOI: https://doi.org/10.1073/pnas.1616408114

- **Sakhnini et al. (2025)** - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters." *Cell* 188(1):112-123. DOI: https://doi.org/10.1016/j.cell.2024.12.025

---

## For Contributors

### When to Update Documentation

1. **Implementation changes** → Update `preprocessing/jain/README.md` (SSOT)
2. **Methodology insights** → Update relevant docs/jain/*.md files
3. **Bug fixes** → Update label_discrepancy_findings.md or create new report
4. **Historical/debugging docs** → Move to archive/ with explanation

### Documentation Principles

- **preprocessing/jain/README.md** is the SINGLE SOURCE OF TRUTH for implementation
- **docs/jain/*.md** provide context, rationale, and technical justification
- **docs/jain/archive/** contains retired methodologies and historical analysis
- Keep docs DRY - reference other docs instead of duplicating
- Mark outdated sections with clear ⚠️ **WARNING** banners

---

## Known Documentation Issues

**High Priority (Needs Rewrite):**
- `complete_guide.md` - Remove sections on 94→86 VH length filtering (doesn't exist)
- `data_sources.md` - Remove "ELISA Data Limitation" section (we have private data!)

**Medium Priority (Needs Updates):**
- `complete_history.md` - Add warning banner about historical methodologies
- `reorganization_complete.md` - Note that fragments were regenerated after reorg

**Low Priority (Keep As-Is):**
- `label_discrepancy_findings.md` - Accurate historical bug report ✅

---

**Last Updated:** 2025-11-06
**Documentation Version:** 2.0 (post-code-drift-cleanup)
**Status:** ⚠️ Partially outdated - major rewrites needed for complete_guide and data_sources
