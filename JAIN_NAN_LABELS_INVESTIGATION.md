# Jain Dataset NaN Labels – Investigation (2025-11-21)

Purpose: clarify whether NaN labels in the Jain dataset are expected, and identify any code paths that currently mishandle them.

What the data shows:
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv` has 137 rows with 94 specific, 22 non-specific, and **21 rows with `label = NaN`** (the mild ELISA 1–3 antibodies).
- This aligns with the documented Step 1 pipeline (`preprocessing/jain/step1_convert_excel_to_csv.py`): mild aggregators are intentionally left unlabeled so they can be excluded in Step 2.
- When loaded with `JainDataset.load_data(stage="parity", sd03_csv_path=...)`, the pipeline removes ELISA 1–3 → reclassifies 5 → removes 30 by PSR/AC-SINS, yielding **86 rows (59 specific / 27 non-specific)**, matching the canonical Novo parity files.
- `stage="ssot"` yields the expected **116 rows (94 / 22)** after ELISA 1–3 removal.

Current mismatch:
- `JainDataset.get_schema()` returns the strict production schema (`get_sequence_dataset_schema`) which forbids null labels. Calling `load_data()` with the default `stage="full"` on the 137-row file triggers Pandera validation errors because of the intentional NaNs.
- The new integration test draft (`tests/integration/test_jain_stage_filtering.py`) will currently **skip** most checks because it points at non-existent file names (`Therapeutics_*.csv`). If those paths are corrected, the default `stage="full"` call will hit the Pandera error unless the schema handling is adjusted.

Implications:
- Training/inference paths that use `stage="ssot"` or `stage="parity"` are fine today (labels are non-null post-filtering).
- Developer workflows that naively load the 137-row file with the default stage will fail validation, even though the NaNs are intentional and documented.

Recommended fixes (for the coding pass, not applied yet):
1) Treat `stage="full"` as a preprocessing view: validate with a nullable schema (e.g., `get_preprocessing_schema()` plus Jain-specific columns) or skip validation until after ELISA filtering. Alternatively, change the default stage to `"ssot"` to avoid accidental validation on the NaN-bearing file.
2) Align the integration test paths to the actual files (`jain_with_private_elisa_FULL.csv`, `jain_sd03.csv`, `VH_only_jain_86_p5e_s2.csv`) and explicitly set `stage` per expectation. This will surface the schema issue in CI and prevent silent skips.
3) Document in the Jain dataset README that `stage="full"` contains NaNs by design and is not a training set; consumers should use `ssot` or `parity` unless they intentionally need the mild-flag rows.

Status: No code changes made. NaNs are expected for ELISA 1–3. The only gap is the strict schema validation when loading the 137-row file with `stage="full"`.
