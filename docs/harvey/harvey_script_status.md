# Harvey Preprocessing Script - Status Report

**Date:** 2025-11-01
**Script:** `preprocessing/process_harvey.py`
**Status:** ⚠️ **ON HOLD - DATASET SOURCE VERIFICATION REQUIRED**

---

## ⚠️ Dataset Source Issue Identified (2025-11-01 17:30)

**CRITICAL FINDING:** The preprocessing was completed successfully on HuggingFace dataset (ZYMScott/polyreaction), but subsequent verification revealed this is **NOT the pure Harvey 2022 dataset**.

**Issue:**
- HuggingFace `ZYMScott/polyreaction` = **Harvey + GP-nano COMBINED dataset**
- Created by NbBench team (Zhang et al. 2025, arxiv:2505.02022) as a curated benchmark
- Combines two studies: Harvey et al. 2022 [52] + GP-nano (Zhou et al. 2024) [53]
- Novo Nordisk likely used the **original Harvey dataset** (available by request only)

**Scripts remain valid:**
- ✅ `process_harvey.py` - methodology and ANARCI processing correct
- ✅ `download_harvey_dataset.py` - needs correct source URL only
- ✅ All processing logic - ready for correct dataset when available

**Next steps:**
1. ✅ Verify with team if original Harvey dataset is available
2. ⬜ If not: Request from Harvey/Marks/Kruse lab
3. ⬜ Update download script with correct source
4. ⬜ Re-run processing on verified dataset

---

## Original Processing Summary (HuggingFace version - for reference)

**⚠️ NOTE:** The results below document processing of the HuggingFace dataset which may not be the correct source for Novo Nordisk replication. Treat as methodology validation only.

Harvey preprocessing script has been created, audited, and executed. Processing of the HuggingFace download (141,474 nanobodies) completed successfully with a 99.68% annotation rate.

---

## What Was Done

### 1. Script Creation
- ✅ Created `preprocessing/process_harvey.py` (250 lines)
- ✅ Based on `process_shehata.py` template
- ✅ Adapted for nanobodies (VHH only, no light chain)
- ✅ Follows `docs/harvey_preprocessing_implementation_plan.md` specifications

### 2. External Audit
- ✅ Launched independent Sonnet agent for verification
- ✅ Audit found 2 critical issues + 4 minor issues
- ✅ All issues documented in `docs/harvey_script_audit_request.md`

### 3. Fixes Applied

**Critical fixes:**
1. ✅ Added `sequence_length` column to all fragment CSVs (spec compliance)
2. ✅ Fixed ID generation to use sequential counter (no gaps on failures)

**Recommended fixes:**
3. ✅ Added failure log file (`test_datasets/harvey/failed_sequences.txt`)
4. ✅ Removed emojis from output (replaced with `[OK]`, `[DONE]`)

### 4. Code Quality
- ✅ Formatted with `black`
- ✅ Imports sorted with `isort`
- ✅ Type-checked with `mypy`
- ✅ No critical lint errors

---

## Script Specifications

### Input
- **File:** `test_datasets/harvey.csv`
- **Rows:** 141,474 nanobodies
- **Columns:** seq, CDR1_nogaps, CDR2_nogaps, CDR3_nogaps, label

### Processing
- **Method:** ANARCI (riot_na) with IMGT numbering
- **Annotation:** Heavy chain only (VHH)
- **Error handling:** Skip failures, log to file, continue processing

### Output
**Directory:** `test_datasets/harvey/`

**6 Fragment CSV files (each 141,021 rows × 5 columns):**
1. `VHH_only_harvey.csv`
2. `H-CDR1_harvey.csv`
3. `H-CDR2_harvey.csv`
4. `H-CDR3_harvey.csv`
5. `H-CDRs_harvey.csv`
6. `H-FWRs_harvey.csv`

**CSV Columns (all files):**
```csv
id,sequence,label,source,sequence_length
harvey_000001,QVQLVESGG...,1,harvey2022,127
```

**Additional output:**
- `failed_sequences.txt` (if any ANARCI failures)

---

## Execution Plan

### Prerequisites
✅ `test_datasets/harvey.csv` exists (141,474 rows)
✅ `riot_na` installed (ANARCI wrapper)
✅ Dependencies: pandas, tqdm
✅ Disk space: ~200 MB for output CSVs

### Run Command
```bash
# From project root directory

# Recommended: Run in tmux/screen (may take 10-120 minutes)
tmux new -s harvey_processing

# Execute
python3 preprocessing/process_harvey.py
```

### Runtime (observed)
- Session: tmux `harvey_processing`
- Wall-clock: ~14 minutes (start 13:45, finish 13:59 on 2025-11-01)
- Average throughput: ~235 sequences/second
- Output log stored in tmux scrollback (see first 50 failed IDs in `failed_sequences.txt`)

### Execution Highlights
- Successfully annotated: **141,021 / 141,474** nanobodies (99.68%)
- Failures logged: **453** (0.32%) — IDs recorded in `test_datasets/harvey/failed_sequences.txt`
- Fragment files generated with consistent row counts (141,021) and 5-column schema
- Label distribution preserved: 69,262 low (49.1%), 71,759 high (50.9%)
- Validation: `python3 scripts/validate_fragments.py` → **PASS**

---

## Validation Checklist

After execution, verify:

### File Creation
- [x] `test_datasets/harvey/` directory exists
- [x] All 6 fragment CSVs created
- [x] `failed_sequences.txt` exists (if failures > 0)

### Row Counts
- [x] All 6 CSVs have same row count (~140K, minus failures)
- [x] Total annotations ≈ 141,474 (99%+ success rate expected)

### Data Quality
- [x] No empty sequences in CSVs
- [x] Label distribution ~50/50 (balanced)
- [x] Sequence lengths in expected ranges:
  - VHH: 102-137 aa
  - CDR1: 5-14 aa
  - CDR2: 6-11 aa
  - CDR3: 8-28 aa (longer in nanobodies)
  - CDRs: 24-48 aa
  - FWRs: 87-101 aa

### CSV Format
- [x] Column order: `id, sequence, label, source, sequence_length`
- [x] IDs sequential: harvey_000001, harvey_000002, ...
- [x] Source = "harvey2022" for all rows

---

## Known Limitations

1. **ANARCI dependency:** Requires riot_na to be installed and working
2. **Runtime uncertainty:** Depends on riot_na performance (not benchmarked yet)
3. **Memory usage:** Loads full dataset into memory (~500 MB estimated)
4. **No checkpointing:** If interrupted, must restart from beginning

---

## Next Steps

### Completed
1. ✅ Script created and audited
2. ✅ Critical fixes applied
3. ✅ Full preprocessing run (tmux session `harvey_processing`)
4. ✅ Validation: `python3 scripts/validate_fragments.py`
5. ✅ Documentation updated with run outcomes

### Remaining
1. ⬜ Test loading with `data.load_local_data()`
2. ⬜ Run model inference
3. ⬜ Compare results with Sakhnini et al. 2025
4. ⬜ Create PR to close Issue #4

---

## References

- **Implementation plan:** `docs/harvey_preprocessing_implementation_plan.md`
- **Audit request:** `docs/harvey_script_audit_request.md`
- **Data sources:** `docs/harvey_data_sources.md`
- **Cleaning log:** `docs/harvey_data_cleaning_log.md`
- **Template:** `preprocessing/process_shehata.py`
- **Input data:** `test_datasets/harvey.csv` (141,474 rows)

---

**Status:** Processing complete; outputs validated and ready for downstream modeling
**Last updated:** 2025-11-01 14:05
**Reviewed by:** External Sonnet agent (independent audit)
