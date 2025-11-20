# Remaining Technical Debt (Source of Truth)

**Last Updated:** 2025-11-20  
**Status:** Ready to kick off Phase A  
**Goal:** Execute Phases A–E to clear remaining architectural debt.

---

## Current Baseline (validated 2025-11-20)

- **Hardcoded paths:** 106 matches in `preprocessing/*.py` (`rg "data/(train|test)" preprocessing --no-heading | wc -l`) plus additional references in tests/e2e.
- **Large files (>500 lines):** `core/trainer.py` (961), `datasets/base.py` (627), `boughter/stage1_dna_translation.py` (598), `boughter/stage2_stage3_annotation_qc.py` (519).
- **`type: ignore` usages (5):**
  - `src/antibody_training_esm/core/embeddings.py:60`
  - `src/antibody_training_esm/core/classifier_factory.py:138`
  - `src/antibody_training_esm/data/loaders.py:16`
  - `tests/unit/datasets/test_base.py:265`
  - `tests/unit/core/strategies/test_logistic_regression.py:344`
- **Executable permissions:** 6 scripts are `755` (`train_hyperparameter_sweep.py`, `validate_stages2_3.py`, `step2_preprocess_p5e_s2.py`, `test_novo_parity.py`, `step2_extract_fragments.py`, `scripts/validation/validate_fragments.py`); others are `644`.
- **Duplicate preprocessing logic:** ~1.6k LOC overlap across validation/fragment scripts (`preprocessing/boughter/validate_stages2_3.py`, `preprocessing/jain/validate_conversion.py`, `preprocessing/harvey/step1_convert_raw_csvs.py`, `preprocessing/shehata/validate_conversion.py`, `preprocessing/jain/step3_extract_fragments.py`, `preprocessing/harvey/step2_extract_fragments.py`, `preprocessing/shehata/step2_extract_fragments.py`).
- **Config duplication:** `configs/testing/jain_p5e_s2.yaml` plus package configs under `src/antibody_training_esm/conf/`.
- **Print/logging gaps:** ~22 `print()` calls in `preprocessing/`, ~36 in `src/` (excluding READMEs) that should be converted or documented.
- **TODO/bug references:** 1 TODO in `tests/integration/test_dataset_pipeline.py`; `CLI_OVERRIDE_BUG` references in `config_schema.py` and `tests/unit/core/test_structured_configs.py` without a backing doc.
- **Pytest config:** Single source in `pyproject.toml`; no `pytest.ini` (keep it that way).

---

## What’s Already in Good Shape

- No active `sys.path` hacks (only comments remain; see `tests/conftest.py`).
- CLI testing code already modularized under `src/antibody_training_esm/cli/testing/`.
- Preprocessing scripts already carry `#!/usr/bin/env python3` shebangs.
- Logging configuration exists for preprocessing (`preprocessing/logging_config.py`).
- Pytest config consolidated in `pyproject.toml`.

---

## Plan (Phases A–E)

- **Phase A – Quick Wins** ([PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md))  
  Standardize permissions, replace 2 bare `except Exception:`, reduce `type: ignore` from 5 → ≤2 with justification, delete empty `utils/`, merge `configs/` into package `conf/testing/`.

- **Phase B – Path Centralization** ([PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md))  
  Create `preprocessing/paths.py`; migrate all preprocessing scripts plus tests/e2e to path constants; eliminate inline `"data/...`" strings.

- **Phase C – File Splitting** ([PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md))  
  Split `core/trainer.py`, `datasets/base.py`, `boughter/stage1_dna_translation.py`, `boughter/stage2_stage3_annotation_qc.py` into focused modules/packages.

- **Phase D – Code Deduplication** ([PHASE_D_CODE_DEDUPLICATION.md](./PHASE_D_CODE_DEDUPLICATION.md))  
  Extract shared validation and fragment logic into `preprocessing/validation_utils.py` and `preprocessing/fragment_utils.py`; verify outputs byte-for-byte.

- **Phase E – Polish** ([PHASE_E_POLISH.md](./PHASE_E_POLISH.md))  
  Document PSR thresholds, clean/justify remaining `print()` calls, clear TODO/bug references, and add docstrings to new modules after splitting/deduplication.

---

## Verification Checklist (run as phases complete)

- **Hardcoded paths cleared**
  - `rg "data/(train|test)" preprocessing tests --glob "*.py" | grep -v paths.py`

- **No files >500 lines**
  - `find src preprocessing -name "*.py" -exec wc -l {} \; | awk '$1>500{print $2": "$1" lines"}'`

- **`type: ignore` count**
  - `rg "type: ignore" src tests`

- **Permissions consistent**
  - `find preprocessing -name "*.py" ! -name "__init__.py" -exec stat -f "%Sp %N" {} \; | awk '{print $1}' | sort | uniq -c`

- **Configs centralized**
  - `ls configs/testing/` (should be empty post-Phase A)
  - `ls src/antibody_training_esm/conf/testing/` (should contain `jain_p5e_s2.yaml`)

- **Print/logging cleanup**
  - `rg "print\\(" src preprocessing | grep -v README`

- **TODO/bug references**
  - `rg "TODO|FIXME|CLI_OVERRIDE_BUG" src preprocessing tests`

- **Quality gates (every phase)**
  - `make all`
  - `uv run pytest`
  - `uv run mypy src/ preprocessing/ --strict`
  - `uv run ruff check src/ preprocessing/`
  - `uv run bandit -r src/ preprocessing/`

---

## Branch & Workflow

- Use one branch per phase (e.g., `claude/refactor-phase-a` … `-e`).
- Commit message format: `<type>: <summary>` (e.g., `refactor: Phase B - path centralization`).
- Record verification commands/results in commit message body (see phase docs for templates).
- Pause between phases for review; do not carry changes forward without quality gates.

---

## Quick Commands Reference

- **Paths scan:** `rg "data/(train|test)" preprocessing tests --glob "*.py" | grep -v paths.py`
- **Large file scan:** `find src preprocessing -name "*.py" -exec wc -l {} \; | awk '$1>500{print $2": "$1" lines"}'`
- **Permissions scan:** `find preprocessing -name "*.py" ! -name "__init__.py" -exec stat -f "%Sp %N" {} \; | awk '{print $1}' | sort | uniq -c`
- **Type ignores:** `rg "type: ignore" src tests`
- **Print calls:** `rg "print\\(" src preprocessing | grep -v README`

---

## Links

- [REFACTOR_PHASES_OVERVIEW.md](./REFACTOR_PHASES_OVERVIEW.md)
- [PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md)
- [PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md)
- [PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md)
- [PHASE_D_CODE_DEDUPLICATION.md](./PHASE_D_CODE_DEDUPLICATION.md)
- [PHASE_E_POLISH.md](./PHASE_E_POLISH.md)
