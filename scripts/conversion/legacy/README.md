# Legacy Conversion Scripts

**Status**: INCORRECT - Archived for historical reference

---

## Why These Scripts Are Here

These scripts represent incorrect approaches from early reverse engineering attempts. They are preserved for:

1. **Provenance** - Understanding the evolution of our methodology
2. **Hard lessons** - Documenting what didn't work
3. **Historical context** - Showing the path to the correct solution

**DO NOT USE THESE SCRIPTS** - They contain bugs and incorrect logic.

---

## Scripts

### `convert_jain_excel_to_csv_OLD_BACKUP.py`

**Problem**: Early backup before we understood the correct methodology

**Status**: Superseded by current `convert_jain_excel_to_csv.py`

---

### `convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py`

**Problem**: Used incorrect `total_flags` methodology

**Bug**: The `total_flags` approach did not properly account for the different flag types and severities in the Jain dataset

**Status**: Superseded by P5e-S2 methodology which uses:
- ELISA flag filtering (remove ELISA 1-3)
- PSR-based reclassification (PSR>0.4)
- PSR/AC-SINS removal strategy

---

## Correct Current Scripts

**For Jain dataset conversion**: Use `scripts/conversion/convert_jain_excel_to_csv.py`

**For Jain preprocessing**: Use `preprocessing/preprocess_jain_p5e_s2.py` (implements P5e-S2 methodology)

---

**Archived**: November 4, 2025
**Branch**: `ray/novo-parity-experiments`
