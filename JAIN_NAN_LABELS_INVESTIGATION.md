# Jain Dataset NaN Labels – Investigation (2025-11-21)

Purpose: clarify whether NaN labels in the Jain dataset are expected, and identify any code paths that currently mishandle them.

What the data shows:
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv` has 137 rows with 94 specific, 22 non-specific, and **21 rows with `label = NaN`** (the mild ELISA 1–3 antibodies).
- This aligns with the documented Step 1 pipeline (`preprocessing/jain/step1_convert_excel_to_csv.py`): mild aggregators are intentionally left unlabeled so they can be excluded in Step 2.
- When loaded with `JainDataset.load_data(stage="parity", sd03_csv_path=...)`, the pipeline removes ELISA 1–3 → reclassifies 5 → removes 30 by PSR/AC-SINS, yielding **86 rows (59 specific / 27 non-specific)**, matching the canonical Novo parity files.
- `stage="ssot"` yields the expected **116 rows (94 / 22)** after ELISA 1–3 removal.

Resolution (2025-11-22):
- `JainDataset.load_data` now validates `stage` eagerly and selects schemas contextually:
  - `stage="full"` → `get_jain_preprocessing_schema()` (nullable labels allowed).
  - `stage="ssot"` / `stage="parity"` → strict `get_jain_schema()` (no nulls).
- Integration test `tests/integration/test_jain_stage_filtering.py` now targets the real files and asserts the canonical counts: 137 (full), 116 (ssot), 86 (parity), plus the Novo class split (59/27) and invalid-stage error handling.

Implications (post-fix):
- Training/inference paths remain strict and safe.
- Exploratory/full-stage loads succeed without Pandera crashes while preserving NaN labels for the mild ELISA rows.

Status: Fixed. The NaN-bearing full stage is now handled explicitly; filtered stages remain strict. Keep using `ssot` or `parity` for modeling; use `full` only when you intentionally need the mild ELISA rows present as NaN.
