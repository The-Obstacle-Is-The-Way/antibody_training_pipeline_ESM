# SPEC: Phase 4 — Fix Documentation

**Status:** DRAFT
**Parent:** [JAIN_PARITY_REMEDIATION_PLAN.md](./JAIN_PARITY_REMEDIATION_PLAN.md)
**Depends On:** Phase 3 (Verification MUST PASS)
**Blocks:** Phase 5 (Final Audit)

---

## Objective

Fix all false parity claims identified in Phase 0 audit. Update documentation to accurately reflect our methodology and results.

---

## Prerequisites

**Phase 3 MUST have passed before starting Phase 4.**

We need verified results before updating documentation:
- Confusion Matrix: `[[40, 17], [10, 19]]`
- Accuracy: 68.60%
- Label Split: 57 specific / 29 non-specific

---

## Fix Categories

### Category 1: Remove False Parity Claims

**Pattern:** Claims that we "achieved" or "match" Novo parity when we didn't (before this fix).

**Action:** Either:
- Remove the claim entirely
- Rewrite to reflect accurate history ("we now achieve parity after Tier D reclassification")

**Example Before:**
```markdown
Our preprocessing achieves exact Novo parity with 68.6% accuracy.
```

**Example After:**
```markdown
Our preprocessing achieves exact Novo parity (68.6% accuracy) after implementing
the full reclassification methodology including Tier D (chromatography-flagged antibodies).
```

### Category 2: Update Confusion Matrix References

**Pattern:** References to our old confusion matrix `[[40, 19], [10, 17]]` as if it were correct.

**Action:** Update to `[[40, 17], [10, 19]]` with context.

**Example Before:**
```markdown
Test results on Jain: [[40, 19], [10, 17]], 66.28% accuracy
```

**Example After:**
```markdown
Test results on Jain: [[40, 17], [10, 19]], 68.60% accuracy
(Matches Novo Figure S14A)
```

### Category 3: Update Accuracy References

**Pattern:** References to 66.28% or 66.3% as our accuracy.

**Action:** Update to 68.60% where appropriate, or provide context.

### Category 4: Update Label Split References

**Pattern:** References to 59/27 split as correct.

**Action:** Update to 57/29 with context.

### Category 5: Add Methodology Caveats

**Pattern:** Documents that claim to explain "Novo's methodology" without caveats.

**Action:** Add caveat that our methodology was reverse-engineered.

**Example Addition:**
```markdown
> **Note:** The Tier D reclassification (lebrikizumab, galiximab) was determined
> through reverse-engineering to achieve Novo parity. Novo does not explicitly
> document this step. See [jain_parity_decision.md](../bugs/jain_parity_decision.md)
> for methodology.
```

---

## Fix Priority Order

Work through the Phase 0 audit list in this order:

### Priority 1: High Visibility (Fix First)

1. **README.md** — First thing users see
2. **docs/index.md** — Landing page
3. **docs/user-guide/training.md** — User-facing training docs
4. **docs/user-guide/testing.md** — User-facing testing docs
5. **docs/research/novo-parity.md** — Research overview

### Priority 2: Medium Visibility

6. **docs/developer-guide/*.md** — Developer docs
7. **docs/datasets/jain/*.md** — Dataset documentation
8. **CLAUDE.md** — AI assistant context
9. **docs/research/methodology.md** — Methodology docs
10. **.github/workflows/benchmark.yml** — CI messaging (must reflect new parity)
11. **validation/baseline/** — Baseline checksums/metrics docs (avoid stale “0.6628”)

### Priority 3: Low Visibility

10. **Code comments** — In preprocessing scripts, test files
11. **Config file comments** — In YAML files
12. **Archive docs** (if any remain)
13. **Checked-in benchmark outputs** — `experiments/benchmarks/**` (keep if historical, but ensure not cited as current)

---

## Standard Fix Templates

### Template A: Methodology Section Update

```markdown
## Jain Dataset Preprocessing

The Jain dataset is preprocessed using the P5e-S2 methodology:

1. **ELISA Filtering:** Remove antibodies with 1-3 ELISA flags (mild)
2. **Reclassification:** Convert specific → non-specific based on biophysical criteria
   - Tier A: PSR > 0.4 (bimagrumab, bavituximab, ganitumab)
   - Tier B: Tm < 60°C (eldelumab)
   - Tier C: Clinical ADA > 60% (infliximab)
   - Tier D: Chromatography-flagged + model-predicted (lebrikizumab, galiximab)
3. **Removal:** Remove 30 specific antibodies by PSR/AC-SINS ranking

**Result:** 57 specific / 29 non-specific = 86 total antibodies

> **Note:** Tier D was determined through reverse-engineering to achieve Novo parity.
> See [jain_parity_decision.md](../bugs/jain_parity_decision.md) for the full rationale.
```

### Template B: Results Section Update

```markdown
## Results on Jain Dataset

| Metric | Value |
|--------|-------|
| Confusion Matrix | `[[40, 17], [10, 19]]` |
| Accuracy | 68.60% |
| Label Split | 57 specific / 29 non-specific |

These results exactly match Novo Figure S14A (ESM-1v VH LogisticReg).
```

### Template C: Caveat Block

```markdown
> **Methodology Note:** Our preprocessing methodology was partially reverse-engineered
> to match Novo's reported results. Specifically, the Tier D reclassification
> (lebrikizumab, galiximab) is inferred, not explicitly stated in Sakhnini et al. (2025).
> See [docs/bugs/jain_parity_decision.md](../bugs/jain_parity_decision.md) for details.
```

---

## File-Specific Guidance

### README.md

**If it claims parity:** Update to reflect accurate state.

**Suggested text:**
```markdown
## Benchmarks

ESM-1v VH LogisticRegression on Jain clinical antibodies:
- **Accuracy:** 68.60% (matches Novo Figure S14A)
- **Confusion Matrix:** `[[40, 17], [10, 19]]`
```

### docs/research/novo-parity.md

This file should be the comprehensive reference. Include:
- Full methodology with all tiers
- Confusion matrix and accuracy
- Caveat about reverse-engineering
- Link to decision document

### CLAUDE.md

Update any claims about parity. Ensure AI assistants get accurate context.

### Code Comments

Update comments in:
- `preprocessing/jain/step2_preprocess_p5e_s2.py` (Phase 1 should have done this)
- Test files that reference expected values
- Benchmark scripts

---

## Commit Strategy

### Batch Commits by Category

```bash
# Commit 1: High visibility docs
git add README.md docs/index.md docs/user-guide/*.md docs/research/novo-parity.md
git commit -m "docs: fix false parity claims in high-visibility docs

Updates README, landing page, and user guides to reflect actual
Novo parity (68.60% accuracy, [[40, 17], [10, 19]] confusion matrix)
achieved after Tier D reclassification.

Issue: #33"

# Commit 2: Medium visibility docs
git add docs/developer-guide/*.md docs/datasets/*.md CLAUDE.md
git commit -m "docs: fix false parity claims in developer and dataset docs

Issue: #33"

# Commit 3: Low visibility (code comments, configs)
git add src/ preprocessing/ tests/
git commit -m "fix: update parity references in code comments

Issue: #33"
```

---

## Verification After Fixes

After all fixes applied:

1. **Re-run Phase 0 search patterns:**
   ```bash
   rg -i "66\.28" --type md  # Should return 0 false claims
   rg "\[40, 19\]" --type md  # Should return 0 (unless in historical context)
   ```

2. **Build docs:**
   ```bash
   make docs-build
   ```
   Should complete without errors.

3. **Check for broken links:**
   Review MkDocs output for warnings.

---

## Checklist

Work through audit list from Phase 0:

### High Visibility
- [ ] README.md
- [ ] docs/index.md
- [ ] docs/user-guide/training.md
- [ ] docs/user-guide/testing.md
- [ ] docs/research/novo-parity.md

### Medium Visibility
- [ ] docs/developer-guide/architecture.md
- [ ] docs/developer-guide/development-workflow.md
- [ ] docs/datasets/jain/preprocessing.md (or equivalent)
- [ ] CLAUDE.md

### Low Visibility
- [ ] Code comments in preprocessing/
- [ ] Code comments in tests/
- [ ] Config file comments

### Verification
- [ ] Re-run search patterns (0 false claims)
- [ ] `make docs-build` passes
- [ ] `make lint` passes
- [ ] `make typecheck` passes

---

## Exit Criteria

Phase 4 is complete when:

1. [ ] All items from Phase 0 audit fixed
2. [ ] Each fix committed with clear message
3. [ ] Re-run of search patterns returns 0 false claims
4. [ ] Docs build without errors
5. [ ] All checklist items marked complete

---

**End of Spec**
