# Remaining Technical Debt (Source of Truth)

**Last Updated:** 2025-11-20  
**Status:** Phase A completed; proceed with Phases B–E  
**Goal:** Execute remaining phases to clear architectural debt.

---

## Current Baseline (post-Phase B, validated 2025-11-20)

- **Hardcoded paths:** 0 matches (paths centralized in `preprocessing/paths.py`).
- **Large files (>500 lines):** `core/trainer.py` (961), `datasets/base.py` (627), `boughter/stage1_dna_translation.py` (598), `boughter/stage2_stage3_annotation_qc.py` (519).
- **`type: ignore` usages (2):**
  - `src/antibody_training_esm/core/embeddings.py:60` (HF tokenizer stubs)
  - `src/antibody_training_esm/data/loaders.py:16` (datasets attr-defined)
- **Executable permissions:** 17 preprocessing scripts are `755` (consistent policy).
- **Duplicate preprocessing logic:** ~1.6k LOC overlap across validation/fragment scripts (`preprocessing/boughter/validate_stages2_3.py`, `preprocessing/jain/validate_conversion.py`, `preprocessing/harvey/step1_convert_raw_csvs.py`, `preprocessing/shehata/validate_conversion.py`, `preprocessing/jain/step3_extract_fragments.py`, `preprocessing/harvey/step2_extract_fragments.py`, `preprocessing/shehata/step2_extract_fragments.py`).
- **Config duplication:** Resolved; single source under `src/antibody_training_esm/conf/` (`configs/` removed).
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
- **Paths Centralized:** All preprocessing scripts and tests use `preprocessing/paths.py`.

---

## Plan (Phases C–E)

- **Phase A – Quick Wins** ([PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md)) - **DONE**
- **Phase B – Path Centralization** ([PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md)) - **DONE**

- **Phase C – File Splitting** ([PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md))

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
