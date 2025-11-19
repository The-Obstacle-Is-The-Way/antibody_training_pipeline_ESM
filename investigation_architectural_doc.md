# Architectural Investigation Report: Status & Recommendations

**Date:** 2025-11-18
**Author:** Jules (AI Engineer)
**Status:** Mixed ("Jekyll & Hyde")
**Objective:** Audit codebase for Design Patterns, SOLID principles, and Architectural consistency.

---

## 1. Executive Summary

The codebase exhibits a "Jekyll & Hyde" character.
- **`src/antibody_training_esm/`**: High quality, production-ready code with Type Hints, Hydra integration, and good modularity.
- **`preprocessing/`**: Research-grade scripts, monolithic files, duplicate code, and architectural "slop".

A comprehensive **Architectural Fixes Plan** (`docs/needs_integration/ARCHITECTURAL_FIXES_PLAN.md`) exists and is partially executed. My investigation confirms the plan's findings and adds detail on Design Pattern usage and SOLID violations.

**Verdict:** The codebase is **not** 100% "Gucci banger" yet. The core library is strong, but the supporting infrastructure (preprocessing, utils) lags behind.

---

## 2. Current Status Audit

I have verified the status of the items in the `ARCHITECTURAL_FIXES_PLAN.md` and performed additional checks.

### ✅ Completed / Good
*   **Phase 1 (P0) Fixes:** Critical path repairs (sys.path hacks, pytest markers) are done.
*   **Logging Migration (Fix #5):** Mostly complete. The remaining ~22 `print()` statements in `preprocessing/` are legitimate final reports/tables.
*   **CLI Refactoring:** `src/antibody_training_esm/cli/test.py` has been successfully split into `src/antibody_training_esm/cli/testing/`.
*   **Core Library Quality:** `src/` uses proper dependency injection (via Hydra), strict typing (mypy), and modular components.

### ⚠️ In Progress / Partial
*   **File Splitting (Fix #6):**
    *   `cli/test.py`: **DONE**.
    *   `core/trainer.py`: **PENDING** (961 lines). This is a "God Class" violation.
    *   `preprocessing/boughter/stage1_dna_translation.py`: **PENDING** (598 lines).
    *   `preprocessing/boughter/stage2_stage3_annotation_qc.py`: **PENDING** (519 lines).

### ❌ Issues / Slop (Pending Fixes)
*   **SOLID Violations:**
    *   **Single Responsibility Principle (SRP):** `trainer.py` handles training, cross-validation, logging, caching, *and* serialization. It needs to be decomposed.
    *   **Don't Repeat Yourself (DRY):** `preprocessing/` scripts contain 200+ lines of duplicated fragment extraction logic (ANARCI) and validation logic.
*   **Architectural Inconsistencies:**
    *   **Config Duplication:** `configs/` (root) and `src/antibody_training_esm/conf/` both exist.
    *   **Utils Vacuity:** `src/antibody_training_esm/utils/` is empty/useless.
    *   **Hardcoded Paths:** 50+ instances in `preprocessing/`.
    *   **Permissions:** Random executable bits on scripts.
    *   **Error Handling:** Bare `except Exception` blocks in `trainer.py` swallow errors and make debugging hard.

---

## 3. Design Pattern Audit

The user specifically asked about Design Patterns.

*   **Strategy Pattern:**
    *   **Status:** **Present but implicitly.** `BinaryClassifier` wraps different backends (LogisticRegression, XGBoost) but relies on `if/else` or duck typing rather than explicit Strategy classes.
    *   **Recommendation:** Formalize the Strategy pattern for Classifiers if complexity grows.
*   **Factory Pattern:**
    *   **Status:** **Implicit.** Hydra acts as the Factory for instantiating models and configurations. This is modern and good.
*   **Template Method:**
    *   **Status:** **Missing.** Preprocessing scripts copy-paste the same structure (Load -> Process -> Save).
    *   **Recommendation:** A `PreprocessingPipeline` base class with `run_step()` methods would enforce consistency.
*   **Singleton:**
    *   **Status:** **Avoided (Good).** Global state is managed via Hydra configuration, not Singletons.

---

## 4. Recommendations (The "Pristine" Path)

To reach "Highest Quality Engineering Standards":

1.  **Finish Phase 2 of the existing Plan:**
    *   **CRITICAL:** Refactor `core/trainer.py`. Split it into `training/cache.py`, `training/metrics.py`, `training/serialization.py`.
    *   **HIGH:** Centralize paths in `preprocessing/paths.py` and remove hardcoded strings.
    *   **HIGH:** Create `preprocessing/fragment_utils.py` to DRY up the ANARCI logic.

2.  **Architectural Cleanups:**
    *   Delete `configs/` (move contents to `src/.../conf/`).
    *   Delete or populate `src/.../utils/`.
    *   Standardize shebangs and permissions.

3.  **Documentation:**
    *   Clarify Assay Thresholds (0.4 vs 0.5495) as planned.

---

**Next Steps:**
The existing `ARCHITECTURAL_FIXES_PLAN.md` is solid. I recommend executing **Fix #6 (Split trainer.py)** and **Fix #7 (Centralize Paths)** immediately as they are the biggest contributors to "slop".
