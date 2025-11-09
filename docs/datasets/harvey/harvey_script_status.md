# Harvey Preprocessing Script - Status Report

**Date:** 2025-11-01 (Updated: 2025-11-06)
**Script:** `preprocessing/harvey/step2_extract_fragments.py`
**Status:** ✅ **COMPLETE - All scripts validated and operational**

---

## Script Status (2025-11-06)

**CONFIRMED:** Data source verified as official Harvey Lab repository (`debbiemarkslab/nanobody-polyreactivity`). All preprocessing scripts operational and validated:

- ✅ `preprocessing/harvey/step1_convert_raw_csvs.py` - Converts raw CSVs to processed format
- ✅ `preprocessing/harvey/step2_extract_fragments.py` - ANARCI annotation and fragment extraction
- ✅ All validation tests passing
- ✅ P0 blocker resolved (gap characters removed)
- ✅ Benchmark parity achieved (61.5% vs Novo's 61.7%)

**Pipeline fully operational.** See `test_datasets/harvey/README.md` for current SSOT.

---

## Processing Summary (Validated)

Harvey preprocessing script has been created, audited, and executed. Processing of the HuggingFace download (141,474 nanobodies) completed successfully with a 99.68% annotation rate.

---

## What Was Done

### 1. Script Creation
- ✅ Created `preprocessing/harvey/step2_extract_fragments.py` (250 lines)
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
3. ✅ Added failure log file (`test_datasets/harvey/fragments/failed_sequences.txt`)
4. ✅ Removed emojis from output (replaced with `[OK]`, `[DONE]`)

### 4. Code Quality
- ✅ Formatted with `black`
- ✅ Imports sorted with `isort`
- ✅ Type-checked with `mypy`
- ✅ No critical lint errors

---

## Script Specifications

### Input
- **File:** `test_datasets/harvey/processed/harvey.csv`
- **Rows:** 141,474 nanobodies
- **Columns:** seq, CDR1_nogaps, CDR2_nogaps, CDR3_nogaps, label

### Processing
- **Method:** ANARCI (riot_na) with IMGT numbering
- **Annotation:** Heavy chain only (VHH)
- **Error handling:** Skip failures, log to file, continue processing

### Output
**Directory:** `test_datasets/harvey/fragments/`

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
✅ `test_datasets/harvey/processed/harvey.csv` exists (141,474 rows)
✅ `riot_na` installed (ANARCI wrapper)
✅ Dependencies: pandas, tqdm
✅ Disk space: ~200 MB for output CSVs

### Run Command
```bash
# From project root directory

# Recommended: Run in tmux/screen (may take 10-120 minutes)
tmux new -s harvey_processing

# Execute
python3 preprocessing/harvey/step2_extract_fragments.py
```

### Runtime (observed)
- Session: tmux `harvey_processing`
- Wall-clock: ~14 minutes (start 13:45, finish 13:59 on 2025-11-01)
- Average throughput: ~235 sequences/second
- Output log stored in tmux scrollback (see first 50 failed IDs in `failed_sequences.txt`)

### Execution Highlights
- Successfully annotated: **141,021 / 141,474** nanobodies (99.68%)
- Failures logged: **453** (0.32%) — IDs recorded in `test_datasets/harvey/fragments/failed_sequences.txt`
- Fragment files generated with consistent row counts (141,021) and 5-column schema
- Label distribution preserved: 69,262 low (49.1%), 71,759 high (50.9%)
- Validation: `python3 scripts/validation/validate_fragments.py` → **PASS**

---

## Validation Checklist

After execution, verify:

### File Creation
- [x] `test_datasets/harvey/fragments/` directory exists
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
4. ✅ Validation: `python3 scripts/validation/validate_fragments.py`
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
- **Template:** `preprocessing/shehata/step2_extract_fragments.py`
- **Input data:** `test_datasets/harvey/processed/harvey.csv` (141,474 rows)

---

**Status:** Processing complete; outputs validated and ready for downstream modeling
**Last updated:** 2025-11-01 14:05
**Reviewed by:** External Sonnet agent (independent audit)
