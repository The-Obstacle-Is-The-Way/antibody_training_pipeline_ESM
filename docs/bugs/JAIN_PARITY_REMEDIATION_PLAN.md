# Jain Parity Remediation Plan

**Status:** ✅ COMPLETE — Implemented and verified
**Created:** 2025-12-16
**Decision:** [jain_parity_decision.md](./jain_parity_decision.md)
**GitHub Issue:** [#33](https://github.com/Clarity-Digital-Twin/antibody_training_pipeline_ESM/issues/33)

---

## Executive Summary

This document outlines the remediation plan for fixing the Jain parity discrepancy. The Tier D remediation (lebrikizumab + galiximab) has been implemented and the pipeline now achieves exact Novo parity.

**Scope:**
- Preprocessing scripts
- Source code parity pipeline (`src/antibody_training_esm/datasets/jain.py`)
- Tests that encode expected counts
- Data artifacts (CSVs)
- Documentation (all false parity claims)
- CI/workflow messaging and baselines (`.github/`, `validation/baseline/`)
- Verification (inference to confirm fix)

> **AUDITOR NOTE (2025-12-16):** The original specs missed several non-`md`/`py` locations (e.g. `.github/workflows/*.yml`, `validation/baseline/*.txt/*.md5`) and multiple tests that hardcode the current 59/27 split. The phase specs have been updated accordingly.

---

## Approach Analysis

### Option A: Bottom-Up (Scripts → Artifacts → Docs)

```text
Phase 1: Fix preprocessing scripts
Phase 2: Regenerate artifacts
Phase 3: Verify with inference
Phase 4: Fix documentation
Phase 5: Final audit
```

**Pros:**
- Logical dependency order
- Artifacts are regenerated from fixed source
- Documentation fixed last (reflects final state)

**Cons:**
- Documentation might have stale claims during transition
- No upfront visibility into scope of doc fixes

### Option B: Audit-First with Parallel Workstreams

```text
Phase 0: Comprehensive audit (find ALL issues)
Phase 1A: Fix preprocessing scripts (parallel)
Phase 1B: Draft doc fixes (parallel, don't apply)
Phase 2: Regenerate artifacts
Phase 3: Verify with inference
Phase 4: Apply doc fixes
Phase 5: Final audit
```

**Pros:**
- Full visibility before any changes
- Parallel work reduces total time
- Draft doc fixes can be reviewed before applying

**Cons:**
- More complex coordination
- Doc fixes might need revision after verification

### Option C: Risk-Based (High-Visibility First)

```text
Phase 1: Fix README and user-facing docs (stop the bleeding)
Phase 2: Fix preprocessing scripts
Phase 3: Regenerate artifacts
Phase 4: Verify with inference
Phase 5: Fix remaining docs
Phase 6: Final audit
```

**Pros:**
- Reduces user confusion immediately
- Most-read docs fixed first

**Cons:**
- Two passes over documentation
- Might introduce inconsistencies during transition

### Option D: Single-Pass Atomic (All-at-Once)

```text
Phase 1: Audit everything
Phase 2: Prepare all fixes (scripts, docs) without applying
Phase 3: Apply everything atomically
Phase 4: Regenerate artifacts
Phase 5: Verify
Phase 6: Final audit
```

**Pros:**
- No inconsistent intermediate states
- Clean commit history

**Cons:**
- Large PR, harder to review
- Any failure requires full rollback

---

## Selected Approach: Option B (Audit-First with Parallel Workstreams)

**Rationale:**
1. **Audit-first** gives us full visibility into the scope before making changes
2. **Parallel workstreams** allow efficient use of time
3. **Verification before doc fixes** ensures docs reflect verified state
4. **Clear phases** allow for review gates between each step

---

## Phase Overview

| Phase | Name | Spec | Blocking? |
|-------|------|------|-----------|
| 0 | Comprehensive Audit | [SPEC_phase0_audit.md](./SPEC_phase0_audit.md) | Yes |
| 1 | Fix Preprocessing + Loader + Tests | [SPEC_phase1_preprocessing.md](./SPEC_phase1_preprocessing.md) | Yes |
| 2 | Regenerate Artifacts | [SPEC_phase2_artifacts.md](./SPEC_phase2_artifacts.md) | Yes |
| 3 | Verify with Inference | [SPEC_phase3_verification.md](./SPEC_phase3_verification.md) | Yes |
| 4 | Fix Documentation | [SPEC_phase4_documentation.md](./SPEC_phase4_documentation.md) | Yes |
| 5 | Final Audit | [SPEC_phase5_final_audit.md](./SPEC_phase5_final_audit.md) | Yes |

---

## Phase 0: Comprehensive Audit

**Goal:** Find ALL false parity claims in code and documentation.

**Deliverables:**
- `AUDIT_false_parity_claims.md` — List of all files with false claims
- Categorized by type (docs, code comments, configs, tests)
- Severity rating (high/medium/low visibility)

**Search Patterns:**
```text
# Claims of parity
"novo parity", "exact parity", "68.6%", "68.60%"
"[[40, 17], [10, 19]]" (in docs claiming we match this)

# Our incorrect numbers (that need updating)
"66.28%", "66.3%"
"[[40, 19], [10, 17]]"
"59 specific", "27 non-specific" (when claiming this is correct)

# Methodology claims
"matches Novo", "reproduces Novo", "identical to Novo"
```

**Blocking:** Phase 0 must complete before any other phase begins.

---

## Phase 1: Fix Preprocessing Scripts

**Goal:** Update `step2_preprocess_p5e_s2.py` to reclassify lebrikizumab + galiximab.

**Files to Modify:**
- `preprocessing/jain/step2_preprocess_p5e_s2.py`
- `src/antibody_training_esm/datasets/jain.py` (stage=`parity` logic and narrative constants)
- `tests/integration/test_jain_stage_filtering.py` (hardcoded 59/27 expectations)

**Changes:**
1. Add Tier D reclassification logic
2. Add code comments explaining the decision
3. Update docstring with methodology
4. Update loader parity stage to match regenerated artifacts
5. Update tests that encode the old split

**Tier D Definition (narrow, low-risk):**
```python
# Tier D: final-label adjustment on the already-selected 86 set.
# Rationale: deterministic chromatography flags (PUBLIC SD03) support reclassifying
# these two specific antibodies as non-specific to match Novo S14A.
TIER_D_CHROMATOGRAPHY = ["lebrikizumab", "galiximab"]
```

**Blocking:** Phase 1 must complete before Phase 2.

---

## Phase 2: Regenerate Artifacts

**Goal:** Regenerate all derived data files from fixed preprocessing.

**Files to Regenerate:**
- `data/test/jain/canonical/jain_86_novo_parity.csv`
- `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv`
- `validation/baseline/checksums/jain_preprocessed.md5` (baseline pins will change)
- `validation/baseline/model_outputs/baseline_metrics.txt` (if treated as current baseline)
- Any other derived files referencing these outputs

**Process:**
1. Run updated preprocessing script
2. Verify output has 57 specific / 29 non-specific
3. Verify lebrikizumab and galiximab have label=1
4. Generate VH-only file

**Blocking:** Phase 2 must complete before Phase 3.

---

## Phase 3: Verify with Inference

**Goal:** Confirm regenerated artifacts produce exact Novo parity.

**Verification Steps:**
1. Load trained model: `experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl`
2. Generate embeddings for 86 antibodies
3. Run inference
4. Compute confusion matrix
5. Verify: `[[40, 17], [10, 19]]`
6. Verify accuracy: 68.60% (59/86)

**Success Criteria:**
```python
assert np.array_equal(cm, [[40, 17], [10, 19]])
assert accuracy == 59/86  # 0.686046...
```

**Deliverable:** Verification script and results log.

**Blocking:** Phase 3 must pass before Phase 4.

---

## Phase 4: Fix Documentation

**Goal:** Fix all false parity claims identified in Phase 0 audit.

**Categories:**
1. **High visibility:** README, user guides, research docs
2. **Medium visibility:** Developer guides, architecture docs
3. **Low visibility:** Archive docs, code comments

**Fix Types:**
- Remove false claims of "exact parity" where we didn't have it
- Update confusion matrices and accuracy numbers
- Add caveats about reverse-engineering methodology
- Reference decision document

**Process:**
1. Work through audit list from Phase 0
2. Fix each file
3. Mark as complete in audit checklist
4. Commit in logical batches

**Blocking:** Phase 4 must complete before Phase 5.

---

## Phase 5: Final Audit

**Goal:** Verify no false claims remain and all fixes are consistent.

**Verification Steps:**
1. Re-run Phase 0 search patterns — should return 0 false claims
2. Verify all audit items marked complete
3. Run `make lint` and `make typecheck`
4. Run `make test` to ensure no regressions
5. Build docs (`make docs-build`) and verify no broken links

**Deliverable:** Final audit report with sign-off.

---

## Rollback Plan (If Phase 3 Fails)

> **AUDITOR NOTE (2025-12-16):** A rollback path was missing; add this to avoid “half-updated” states.

1. Revert regenerated artifacts commit(s) (Phase 2) first.
2. Revert code/test changes (Phase 1).
3. Re-run Phase 0 search patterns to ensure the repo is back to pre-change consistency.
4. Re-open research: reassess whether membership drift (selection) vs labeling is the true cause.

---

## Commit Strategy

### Branch Structure
```text
main
└── fix/jain-parity-remediation
    ├── Phase 1 commit: Fix preprocessing script
    ├── Phase 2 commit: Regenerate artifacts
    ├── Phase 3 commit: Add verification script and results
    └── Phase 4 commits: Fix documentation (batched by category)
```

### Commit Messages
```text
fix(preprocessing): add Tier D reclassification for lebrikizumab + galiximab

Reclassifies lebrikizumab and galiximab from specific to non-specific
based on chromatography flags (HIC > 11.7). This achieves exact Novo
parity: [[40, 17], [10, 19]], 68.60% accuracy.

Decision: docs/bugs/jain_parity_decision.md
Issue: #33
```

---

## Rollback Plan

If any phase fails:

1. **Phase 1 fails:** No changes to artifacts or docs yet. Fix script issues.
2. **Phase 2 fails:** Don't proceed. Debug regeneration.
3. **Phase 3 fails:** Critical — our hypothesis was wrong. Re-analyze.
4. **Phase 4 fails:** Partial doc fixes. Continue fixing.
5. **Phase 5 fails:** Identify remaining issues, loop back.

**Nuclear rollback:** `git reset --hard` to pre-remediation commit.

---

## Timeline

| Phase | Estimated Duration | Dependencies |
|-------|-------------------|--------------|
| 0 | 30 minutes | None |
| 1 | 15 minutes | Phase 0 |
| 2 | 5 minutes | Phase 1 |
| 3 | 10 minutes | Phase 2 |
| 4 | 1-2 hours | Phase 3 |
| 5 | 15 minutes | Phase 4 |

**Total estimated time:** 2-3 hours

---

## Success Criteria

The remediation is complete when:

1. [ ] Preprocessing script includes Tier D reclassification
2. [ ] `jain_86_novo_parity.csv` has 57 specific / 29 non-specific
3. [ ] Inference produces `[[40, 17], [10, 19]]` with 68.60% accuracy
4. [ ] All false parity claims removed from documentation
5. [ ] Final audit passes with 0 remaining issues
6. [ ] All tests pass (`make test`)
7. [ ] Docs build without errors (`make docs-build`)

---

## References

- Decision: [jain_parity_decision.md](./jain_parity_decision.md)
- Research Spec: [jain_parity_reverse_engineering.md](./jain_parity_reverse_engineering.md)
- Data Inventory: [jain_parity_data_inventory.md](./jain_parity_data_inventory.md)
- GitHub Issue: #33
