# Harvey Preprocessing Script - Audit Request

**Date:** 2025-11-01
**Script:** `preprocessing/process_harvey.py`
**Purpose:** Extract VHH (nanobody) fragments from Harvey dataset for ESM-1v testing

---

## ⚠️ Dataset Source Verification Issue (2025-11-01 17:30)

**NOTE:** Audit findings remain valid for processing methodology. Dataset source subsequently identified as requiring verification (HuggingFace ZYMScott/polyreaction = Harvey+GP-nano combined dataset, not pure Harvey 2022).

**Script validity:** Processing logic is correct and ready for use once correct dataset source is verified.

---

## Request

Please audit `preprocessing/process_harvey.py` against the following specifications:

1. **Does it follow the methodology in `docs/harvey_preprocessing_implementation_plan.md`?**
2. **Does it match the codebase style of `preprocessing/process_shehata.py`?**
3. **Are there any bugs, edge cases, or issues?**
4. **Will it correctly process 141,474 nanobody sequences?**

---

## Specifications (from harvey_preprocessing_implementation_plan.md)

### Input File
- **Path:** `test_datasets/harvey/processed/harvey.csv`
- **Rows:** 141,474 nanobodies (plus header)
- **Columns:** `seq, CDR1_nogaps, CDR2_nogaps, CDR3_nogaps, label`
- **Labels:** Binary (0=low polyreactivity, 1=high polyreactivity)

### Processing Method
- **Tool:** ANARCI (riot_na) with IMGT numbering scheme
- **Sequence type:** VHH (nanobody) - **heavy chain only, NO light chain**
- **Expected failures:** <1% (based on Shehata experience)

### Output

**Directory:** `test_datasets/harvey/fragments/`

**6 Fragment CSV files:**
1. `VHH_only_harvey.csv` - Full nanobody variable domain
2. `H-CDR1_harvey.csv` - Heavy CDR1
3. `H-CDR2_harvey.csv` - Heavy CDR2
4. `H-CDR3_harvey.csv` - Heavy CDR3
5. `H-CDRs_harvey.csv` - Concatenated H-CDR1+2+3
6. `H-FWRs_harvey.csv` - Concatenated H-FWR1+2+3+4

**CSV Format (all files):**
```csv
id,sequence,label,source,sequence_length
harvey_000001,QVQLVESGG...,1,harvey2022,127
```

**Columns:**
- `id`: Generated ID (harvey_000001, harvey_000002, etc.)
- `sequence`: Fragment sequence (CDR, FWR, or full VHH)
- `label`: Binary polyreactivity (0=low, 1=high)
- `source`: Dataset provenance ("harvey2022")
- `sequence_length`: Fragment length in amino acids

### Expected Performance
- **Processing time:** 10-30 minutes (141K sequences)
- **Success rate:** >99% (ANARCI annotation)
- **Label distribution:** ~50/50 (balanced dataset)

---

## Comparison with Shehata Script

Harvey script should be **similar to** `preprocessing/process_shehata.py` BUT:

### Same as Shehata:
- ✅ Uses `riot_na.create_riot_aa()` for ANARCI
- ✅ Same `annotate_sequence()` pattern
- ✅ Same CSV output format (id, sequence, label, source)
- ✅ Same error handling (skip failures, log warnings)
- ✅ Same progress bar (tqdm)

### Different from Shehata:
- ❌ NO light chain processing (VHH only)
- ❌ 6 fragments (not 16)
- ❌ Different input format (harvey.csv has "seq" column, not "heavy_seq"/"light_seq")
- ❌ Different metadata (no psr_score, no b_cell_subset)
- ❌ IDs generated (harvey_XXXXXX) instead of loaded from CSV
- ❌ Much larger dataset (141K vs 398 sequences)

---

## Key Verification Points

### 1. Input Handling
- [ ] Reads `test_datasets/harvey/processed/harvey.csv` correctly
- [ ] Accesses `row["seq"]` (not `row["heavy_seq"]`)
- [ ] Handles 141,474 rows without memory issues
- [ ] Generates IDs as `harvey_000001`, `harvey_000002`, etc.

### 2. ANARCI Annotation
- [ ] Uses heavy chain annotation only (no light chain calls)
- [ ] Extracts 8 raw fragments: full_seq_H, fwr1-4_aa_H, cdr1-3_aa_H
- [ ] Creates 2 concatenated fragments: cdrs_H, fwrs_H
- [ ] Handles annotation failures gracefully (skip, log, continue)

### 3. Fragment CSV Generation
- [ ] Creates exactly 6 CSV files (not 16)
- [ ] Correct fragment mapping:
  - `VHH_only` → `full_seq_H`
  - `H-CDR1` → `cdr1_aa_H`
  - `H-CDR2` → `cdr2_aa_H`
  - `H-CDR3` → `cdr3_aa_H`
  - `H-CDRs` → `cdrs_H`
  - `H-FWRs` → `fwrs_H`
- [ ] All files have 5 columns: id, sequence, label, source, sequence_length
- [ ] Source is "harvey2022" (not "harvey2025" or other)

### 4. Error Handling
- [ ] Checks if `test_datasets/harvey/processed/harvey.csv` exists
- [ ] Try/except around ANARCI calls
- [ ] Logs failed sequences to stderr
- [ ] Continues processing after failures (doesn't halt)
- [ ] Reports failure count in summary

### 5. Output Validation
- [ ] All 6 fragment files created
- [ ] Row count: ~141K (minus ANARCI failures)
- [ ] Label distribution preserved from input
- [ ] No missing values in output CSVs

---

## Specific Code Review Questions

1. **Line 92:** Does `seq_id = f"harvey_{idx+1:06d}"` correctly generate `harvey_000001`, `harvey_000002`, etc.?
   - Note: `idx` is 0-indexed, so `idx+1` is needed

2. **Line 94:** Is `row["seq"]` the correct column name?
   - Verify against test_datasets/harvey/processed/harvey.csv header

3. **Line 99:** Is metadata complete?
   - Should it include anything from HuggingFace besides label?

4. **Line 130-141:** Are fragment names and mappings correct?
   - VHH_only → full_seq_H ✓
   - H-CDR1 → cdr1_aa_H ✓
   - H-CDRs → cdrs_H ✓
   - H-FWRs → fwrs_H ✓

5. **Line 198:** Is output directory path correct?
   - Should be `test_datasets/harvey/fragments/` (new structure after cleanup)

6. **Memory usage:** Will loading 141K sequences into DataFrame cause issues?
   - Estimate: ~20-30 MB for CSV, 100-200 MB for processed data (acceptable)

---

## Expected Runtime

**Conservative estimates:**
- ANARCI annotation: ~0.1-0.2s per sequence
- Total: 141,474 × 0.15s ≈ 6 hours (worst case)

**Optimistic estimates:**
- ANARCI annotation: ~0.01-0.05s per sequence (if riot_na is fast)
- Total: 141,474 × 0.03s ≈ 70 minutes

**Recommendation:** Run in tmux/screen for safety

---

## Audit Checklist

### Code Quality
- [ ] Follows PEP 8 (black formatted)
- [ ] Type hints on all functions
- [ ] Docstrings complete
- [ ] Error messages informative
- [ ] Progress tracking with tqdm

### Correctness
- [ ] Matches harvey_preprocessing_implementation_plan.md
- [ ] Consistent with process_shehata.py pattern
- [ ] No light chain code (common bug)
- [ ] Correct column names for harvey.csv

### Performance
- [ ] Will handle 141K sequences
- [ ] No obvious bottlenecks
- [ ] Memory usage reasonable

### Edge Cases
- [ ] Handles ANARCI failures
- [ ] Handles missing input file
- [ ] Handles malformed sequences
- [ ] Reports errors clearly

---

## References

- **Implementation plan:** `docs/harvey_preprocessing_implementation_plan.md`
- **Data sources:** `docs/harvey_data_sources.md`
- **Cleaning log:** `docs/harvey_data_cleaning_log.md`
- **Template script:** `preprocessing/process_shehata.py`
- **Sakhnini et al. 2025:** Table 4 (Harvey dataset specs)

---

**Auditor:** Please verify the script is correct before running on 141K sequences.
