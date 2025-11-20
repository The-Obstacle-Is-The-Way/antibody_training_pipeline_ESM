# Technical Debt Cleanup - Phases Overview

**Last Updated:** 2025-11-20
**Status:** Ready to execute
**Total Effort:** 14-18 hours across 5 phases

---

## Quick Reference

| Phase | Focus | Effort | Risk | Scope | Document |
|-------|-------|--------|------|-------|----------|
| **A** | Quick Wins | 1-1.5h | LOW | 6 scripts + 4 library/test files | [PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md) |
| **B** | Path Centralization | 2-3h | MEDIUM | 20+ files, 106 hardcoded paths | [PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md) |
| **C** | File Splitting | 4-5h | HIGH | 4 files >500 lines | [PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md) |
| **D** | Code Deduplication | 5-7h | HIGH | 6 preprocessing scripts (~1.6k LOC overlap) | [PHASE_D_CODE_DEDUPLICATION.md](./PHASE_D_CODE_DEDUPLICATION.md) |
| **E** | Polish & Docs | 2-3h | LOW | Docs/comments/bug refs after refactors | [PHASE_E_POLISH.md](./PHASE_E_POLISH.md) |

---

## Phase A: Quick Wins (Start Here!)

**Goal:** Knock out 5 low-risk fixes in ~1 hour and reduce baseline noise.

**Tasks (current state evidence-based):**
- Fix #8: Standardize file permissions (6 executable scripts today; make policy consistent)
- Fix #9: Replace 2 bare `except Exception:` blocks in `core/trainer.py` (lines ~176, ~858)
- Fix #10: Reduce 5 `type: ignore` usages to the minimum set with explanations (embeddings, classifier_factory, data/loaders, 2 tests)
- Fix #11: Delete empty `src/antibody_training_esm/utils/` (no imports reference it)
- Fix #12: Merge `configs/testing/` into `src/antibody_training_esm/conf/testing/`

**Why start here:** Easy wins, build momentum, zero risk

**Read:** [PHASE_A_QUICK_WINS.md](./PHASE_A_QUICK_WINS.md)

---

## Phase B: Path Centralization

**Goal:** Eliminate 100+ hardcoded paths (106 matches in preprocessing alone) and give tests a single source of truth.

**Tasks:**
- Create `preprocessing/paths.py`
- Migrate all preprocessing scripts + supporting tests to use centralized constants

**Why this is important:** Foundation for later phases, makes testing easier

**Dependencies:** Phase A complete

**Read:** [PHASE_B_PATH_CENTRALIZATION.md](./PHASE_B_PATH_CENTRALIZATION.md)

---

## Phase C: File Splitting

**Goal:** Split 4 oversized files (>500 lines) into focused modules

**Tasks:**
- Split `core/trainer.py` (961 → ~350 lines + cache/metrics/serialization modules)
- Split `datasets/base.py` (627 → ~350 lines + mixins for validation/annotation/fragments)
- Split `boughter/stage1_dna_translation.py` (598 → ~250 lines + translation/validation modules)
- Split `boughter/stage2_stage3_annotation_qc.py` (519 → ~250 lines + annotation/qc modules)

**Why this is HIGH risk:** Significant structural changes, extensive testing needed

**Dependencies:** Phases A & B complete

**Read:** [PHASE_C_FILE_SPLITTING.md](./PHASE_C_FILE_SPLITTING.md)

---

## Phase D: Code Deduplication

**Goal:** Extract duplicated validation/fragment code into shared modules (~1.6k LOC overlapping today)

**Tasks:**
- Create `validation_utils.py` (refactor 4 validation-heavy scripts)
- Create `fragment_utils.py` (refactor 3 fragment extractors)
- Verify byte-for-byte identical outputs

**Why this is HIGH risk:** Must produce identical output, extensive verification

**Dependencies:** Phases A, B, C complete

**Read:** [PHASE_D_CODE_DEDUPLICATION.md](./PHASE_D_CODE_DEDUPLICATION.md)

---

## Phase E: Polish & Documentation

**Goal:** Final touches for pristine codebase

**Tasks:**
- Document PSR threshold differences (classifier vs preprocessing)
- Clean up remaining legacy comments (TODO/CLI override bug references) or link to source doc
- Review and convert lingering `print()` diagnostics to logging where appropriate
- Add/align docstrings for new modules created in Phases C & D

**Why this is LOW risk:** Cosmetic changes only

**Dependencies:** Phases A-D complete

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

### Before All Phases (Current State)

| Metric | Current (2025-11-20) | Notes |
|--------|----------------------|-------|
| Hardcoded paths | 106 matches in `preprocessing/` (rg) + test suite references | No `preprocessing/paths.py` yet |
| Files >500 lines | 4 files: `core/trainer.py` (961), `datasets/base.py` (627), `boughter/stage1_dna_translation.py` (598), `boughter/stage2_stage3_annotation_qc.py` (519) | Single-responsibility violations |
| `type: ignore` usages | 5 occurrences (embeddings, classifier_factory, data/loaders, 2 tests) | Exceeds planned target of ≤2 |
| Executable permissions | 6 Python scripts are `755`, remainder `644` | Inconsistent policy |
| Duplicate preprocessing logic | ~1.6k LOC overlap across 6 validation/fragment scripts | Needs shared utils |
| Config locations | Root `configs/testing/` + package `conf/` | Two sources of truth |
| TODO/bug references | 1 TODO (`tests/integration/test_dataset_pipeline.py`); CLI override bug doc referenced but missing | Clarify/remove in Phase E |

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
| Phase A | 1-1.5h | +30 min | 2h max |
| Phase B | 2-3h | +1h | 4h max |
| Phase C | 3-4h | +1h | 5h max |
| Phase D | 5-7h | +2h | 9h max |
| Phase E | 2-3h | +1h | 4h max |
| **TOTAL** | **14-18h** | **+5.5h** | **24h max** |

**Recommended schedule:**
- Week 1: Phases A + B (complete path centralization foundation)
- Week 2: Phase C (file splitting - high risk, needs focus)
- Week 3: Phases D + E (deduplication + polish)

---

## How to Use These Documents

1. **Read PHASE_A_QUICK_WINS.md first**
2. **Complete all tasks in Phase A**
3. **Run quality gates**
4. **Get senior approval**
5. **Move to Phase B**
6. **Repeat until Phase E complete**

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
