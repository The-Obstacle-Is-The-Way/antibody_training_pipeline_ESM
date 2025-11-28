# Technical Debt Cleanup - Phases Overview

> **Note:** This document references `leroy-jenkins/full-send` which was renamed to `main` on 2025-11-28.

**Last Updated:** 2025-11-20
**Status:** All Phases Completed (A-E)
**Total Effort Remaining:** 0h

---

## Quick Reference

| Phase | Focus | Effort | Risk | Scope | Document |
|-------|-------|--------|------|-------|----------|
| **A** | Quick Wins | ✅ DONE | LOW | Permissions, bare excepts, type ignores, utils/, configs/ | [PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md) |
| **B** | Path Centralization | ✅ DONE | MEDIUM | 20+ files, 106 hardcoded paths | [PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md) |
| **C** | File Splitting | ✅ DONE | HIGH | 4 files >500 lines | [PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md) |
| **D** | Code Deduplication | ✅ DONE | HIGH | 6 preprocessing scripts (~1.6k LOC overlap) | [PHASE_D_CODE_DEDUPLICATION.md](./PHASE_D_CODE_DEDUPLICATION.md) |
| **E** | Polish & Docs | ✅ DONE | LOW | Docs/comments/bug refs after refactors | [PHASE_E_POLISH.md](./PHASE_E_POLISH.md) |

---

## Phase A: Quick Wins (Completed)

**Outcome:** All five tasks completed on commit `claude/refactor-phase-a` (see PHASE_A_QUICK_WINS.md for details and verification logs).

**State after completion:**
- Permissions: 17 preprocessing scripts set to 755 (consistent policy).
- Bare excepts: Two handlers in `core/trainer.py` now log specific exceptions and re-raise on unexpected errors (no silent bare excepts).
- `type: ignore`: Reduced to 2 (embeddings HF tokenizer stubs; datasets import attr-defined) with inline justification.
- `src/antibody_training_esm/utils/`: Removed.
- `configs/`: Removed; `configs/testing/jain_p5e_s2.yaml` moved to `src/antibody_training_esm/conf/testing/`.

**Next:** Proceed to Phase B.

---

## Phase B: Path Centralization (Completed)

**Goal:** Eliminate 100+ hardcoded paths (106 matches in preprocessing alone) and give tests a single source of truth.

**Outcome:**
- Created `preprocessing/paths.py`
- Migrated all preprocessing scripts + supporting tests to use centralized constants
- Zero hardcoded paths in preprocessing scripts

**Read:** [PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md)

---

## Phase C: File Splitting (Completed)

**Goal:** Split 4 oversized files (>500 lines) into focused modules

**Outcome:**
- Split `core/trainer.py` into `core/training/{cache,metrics,serialization}.py` + main orchestrator.
- Split `datasets/base.py` into `datasets/mixins/{annotation,fragment}_mixin.py` + base class.
- Split `boughter/stage1_dna_translation.py` into `translation/{readers,translator}.py`.
- Split `boughter/stage2_stage3_annotation_qc.py` into `annotation/{annotator,qc}.py`.
- All tests passing (513 unit/integration + E2E).

**Read:** [PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md)

---

## Phase D: Code Deduplication (Completed)

**Goal:** Extract duplicated validation/fragment code into shared modules (~1.6k LOC overlapping today)

**Outcome:**
- Created `preprocessing/validation_utils.py` (shared validation logic).
- Created `preprocessing/fragment_utils.py` (shared ANARCI/fragment logic).
- Refactored 7 scripts across Boughter, Jain, Harvey, and Shehata datasets to use shared utils.
- Removed ~600 lines of duplicate code.
- Verified identical behavior and full test suite pass.

**Read:** [PHASE_D_CODE_DEDUPLICATION.md](./PHASE_D_CODE_DEDUPLICATION.md)

---

## Phase E: Polish & Documentation (Completed)

**Goal:** Final touches for pristine codebase

**Outcome:**
- Clarified `CLI_OVERRIDE_BUG` references (replaced missing link with explanation).
- Added missing module docstrings to `core/training/*.py` and `datasets/mixins/*.py`.
- Suppressed `bandit` security false positives for `pickle` in development tools.
- Verified code quality gates (mypy, ruff, bandit, tests).

**Read:** [PHASE_E_POLISH.md](./PHASE_E_POLISH.md)

---

## Execution Strategy

### Recommended Approach: Sequential with Quality Gates

```
Phase A → [Quality Gate] → Phase B → [Quality Gate] → Phase C → [Quality Gate] → Phase D → [Quality Gate] → Phase E → [Final Gate]
```

### Quality Gate Checklist

**After each phase:**
1. ✅ Run `make all` (format, lint, typecheck, test)
2. ✅ Run full test suite: `uv run pytest`
3. ✅ Run security scan: `uv run bandit -r src/ preprocessing/`
4. ✅ Spot check: Manually run 2-3 scripts
5. ✅ Commit with detailed message
6. ✅ Get senior approval before next phase

### Branch Strategy (Option 2 - RECOMMENDED)

Create one branch per phase:

```bash
# Phase A
git checkout -b claude/refactor-phase-a
# ... complete Phase A ...
# Merge to dev, get approval

# Phase B
git checkout -b claude/refactor-phase-b
# ... complete Phase B ...
# Merge to dev, get approval

# (Continue for C, D, E)

# Final merge
git checkout leroy-jenkins/full-send
git merge dev
```

**Why separate branches?**
- Smaller, reviewable PRs
- Can pause for senior review between phases
- Easier rollback if issues found
- Incremental progress visible

---

## Success Metrics

### Current State (Post-Phase A, 2025-11-20)

| Metric | Current | Notes |
|--------|---------|-------|
| Hardcoded paths | 106 matches in `preprocessing/` (rg) + test suite references | No `preprocessing/paths.py` yet |
| Files >500 lines | 4 files: `core/trainer.py` (961), `datasets/base.py` (627), `boughter/stage1_dna_translation.py` (598), `boughter/stage2_stage3_annotation_qc.py` (519) | Single-responsibility violations |
| `type: ignore` usages | 2 (embeddings HF tokenizer stubs; datasets attr-defined) | Documented inline |
| Executable permissions | 17 preprocessing scripts are `755` | Consistent policy applied |
| Duplicate preprocessing logic | ~1.6k LOC overlap across 6 validation/fragment scripts | Needs shared utils |
| Config locations | Single source under `src/antibody_training_esm/conf/` | Root `configs/` removed |
| TODO/bug references | 1 TODO (`tests/integration/test_dataset_pipeline.py`); CLI override bug doc referenced but missing | Clarify/remove in Phase E |
| Print/logging gaps | 22 `print()` calls in `preprocessing/`, 36 in `src/` (excluding READMEs) | Convert/document in Phase E |

### After All Phases (Target State)

| Metric | Target | Notes |
|--------|--------|-------|
| Hardcoded paths | Centralized via `preprocessing/paths.py` with overrides for tests | Zero inline `"data/..."` strings |
| Files >500 lines | 0 | All four large files split into focused modules |
| `type: ignore` usages | ≤2 with inline justification (external stubs only) | Documented in code |
| Executable permissions | Consistent policy applied to preprocessing/scripts | Either all executable or none, documented |
| Duplicate preprocessing logic | Shared validation/fragment utils, noop diffs on outputs | Byte-for-byte verification |
| Config locations | Single source under `src/antibody_training_esm/conf/` | Tests updated accordingly |
| TODO/bug references | Remaining references link to live docs or are removed | No stale breadcrumbs |

---

## Time Budget

| Phase | Estimated | Buffer | Total |
|-------|-----------|--------|-------|
| Phase A | ✅ DONE | — | — |
| Phase B | 2-3h | +1h | 4h max |
| Phase C | 4-5h | +1h | 6h max |
| Phase D | 5-7h | +2h | 9h max |
| Phase E | 2-3h | +1h | 4h max |
| **REMAINING** | **13-18h** | **+4h** | **22h max** |

**Recommended schedule:**
- Week 1: Phases A + B (complete path centralization foundation)
- Week 2: Phase C (file splitting - high risk, needs focus)
- Week 3: Phases D + E (deduplication + polish)

---

## How to Use These Documents

1. **Phase A complete** (see PHASE_A_QUICK_WINS.md for proof/commits)
2. **Phase B complete** (see PHASE_B_PATH_CENTRALIZATION.md)
3. **Start at Phase C** — read PHASE_C_FILE_SPLITTING.md
4. Complete all tasks in each phase
5. Run quality gates
5. Get review/approval
6. Continue through Phase E

Each phase document contains:
- ✅ Overview and goals
- ✅ Detailed task specs
- ✅ Code examples and patterns
- ✅ Verification steps
- ✅ Success criteria
- ✅ Git workflow (branch, commit, PR)

---

## Need Help?

**Questions about phasing strategy?**
- Review this overview document
- Check individual phase docs for details

**Ready to start?**
- Begin with [PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md)
- Follow the sequential execution strategy
- Run quality gates between phases

**Got stuck?**
- Check verification steps in phase doc
- Review previous phase completions
- Ensure quality gates passed

---

**Let's clean up this tech debt and ship a pristine codebase! 🚀**
