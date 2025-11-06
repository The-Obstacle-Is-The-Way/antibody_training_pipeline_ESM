# Jain Dataset Documentation Archive

**Purpose:** Historical documentation describing RETIRED methodologies that are NO LONGER USED

**⚠️ WARNING:** All documents in this directory describe approaches that were superseded by the current ELISA-only P5e-S2 pipeline

---

## Archived Documents

### 1. JAIN_QC_REMOVALS_COMPLETE.md
**Status:** ❌ **COMPLETELY OUTDATED** (0% accurate)
**Archived:** 2025-11-06
**Reason:** Describes 94→86 QC pipeline that DOES NOT EXIST in code

**What it describes:**
- Pipeline: 94→91 (remove VH length outliers)→86 (remove borderline antibodies)
- Removes: crenezumab, fletikumab, secukinumab (VH length)
- Removes: muromonab, cetuximab, girentuximab, tabalumab, abituzumab (clinical QC)
- Claims this achieves Novo parity

**Why it's wrong:**
- Current code (step2_preprocess_p5e_s2.py) uses completely different method
- No VH length filtering in current pipeline
- No clinical QC removals in current pipeline
- Current method: 116→89 (reclassify 5)→86 (remove 30 by PSR/AC-SINS)

**Historical context:**
This was an early hypothesis about how Novo Nordisk achieved their 86-antibody test set. It was superseded when we discovered the P5e-S2 method (PSR reclassification + PSR/AC-SINS removal) that achieves EXACT parity (confusion matrix [[40,19],[10,17]]).

---

### 2. JAIN_REPLICATION_PLAN.md
**Status:** ❌ **COMPLETELY OUTDATED** (0% accurate)
**Archived:** 2025-11-06
**Reason:** Describes wrong Novo methodology we never implemented

**What it describes:**
- Claims Novo used 10-flag system (0-10 range)
- Describes "buggy 0-7 range" that needs fixing
- States we need to add BVP flag to match Novo
- Entire flag calculation methodology is wrong

**Why it's wrong:**
- Novo used ELISA-ONLY flags (0-6 range from 6 ELISA antigens)
- We have private ELISA data (Private_Jain2017_ELISA_indiv.xlsx)
- step1_convert_excel_to_csv.py implements ELISA-only (NOT 10-flag total)
- Threshold is >=4 ELISA flags, NOT >=4 total flags
- BVP flag is calculated but NOT used for labeling

**Historical context:**
This was a planning document from when we thought Novo used all 10 assay flags. Evidence from Figure S13 (x-axis "ELISA flag" singular, range 0-6) and Table 2 ("panel of 6 ligands") proved this was incorrect.

---

### 3. jain_conversion_verification_report.md
**Status:** ⚠️ **PARTIALLY OUTDATED** (30% accurate)
**Archived:** 2025-11-06
**Reason:** Describes flags_total methodology that's RETIRED

**What it describes:**
- Excel→CSV conversion process (still relevant)
- flags_total calculation (0-4 range) - **RETIRED**
- Distribution: 67/67/3 (specific/mild/non-specific) - **WRONG**
- Threshold >=3 vs >=4 bug discussion - **ONLY RELEVANT TO flags_total**

**Why it's partially outdated:**
- Current system uses ELISA-only flags (0-6 range), NOT flags_total
- Current distribution: 94/22/21 (specific/non-specific/mild)
- Correct threshold: >=4 ELISA flags
- Column names changed: heavy_seq→vh_sequence, light_seq→vl_sequence

**What's still correct:**
- General conversion process (Excel→CSV)
- Source file references (SD01, SD02, SD03, Private ELISA)
- Sequence sanitization discussion

**Historical context:**
This report validated the original paper-based conversion that used flags_total. This methodology was retired when we discovered the ELISA-only SSOT (94/22/21 distribution) documented in label_discrepancy_findings.md.

---

## Why These Were Archived

All three documents describe methodologies that:
1. ❌ Do NOT match the current preprocessing code
2. ❌ Would produce INCORRECT results if followed
3. ❌ Reference files and pipelines that don't exist

**Current SSOT:** See `preprocessing/jain/README.md` and parent `docs/jain/README.md`

---

## What Replaced Them

### Retired Approach 1: flags_total (0-4 or 0-10 range)
**Replaced by:** ELISA-only flags (0-6 range)
**Evidence:** Figure S13, Table 2, step1_convert_excel_to_csv.py
**Distribution:** 67/27/43 → **94/22/21**

### Retired Approach 2: 94→86 VH length + clinical QC
**Replaced by:** P5e-S2 (116→86 PSR reclassification + removal)
**Evidence:** step2_preprocess_p5e_s2.py achieves EXACT Novo parity
**Pipeline:** 94→91→86 → **116→89→86**

---

## How to Use Archived Docs

**Don't:**
- ❌ Follow these methodologies for new work
- ❌ Update code to match these docs
- ❌ Cite these as current approaches

**Do:**
- ✅ Understand historical evolution of methodology
- ✅ Learn from debugging process
- ✅ Reference when explaining why we changed approaches
- ✅ Recover specific details from git history if needed

---

**Archived:** 2025-11-06
**Reason:** Code drift cleanup - align documentation with implemented SSOT
**Recovery:** All documents remain in git history (pre-archive commit)
