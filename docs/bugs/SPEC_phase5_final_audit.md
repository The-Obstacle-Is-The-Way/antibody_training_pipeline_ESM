# SPEC: Phase 5 — Final Audit

**Status:** DRAFT
**Parent:** [JAIN_PARITY_REMEDIATION_PLAN.md](./JAIN_PARITY_REMEDIATION_PLAN.md)
**Depends On:** Phase 4 (Documentation Fix)
**Blocks:** None (Final Phase)

---

## Objective

Verify that all remediation tasks are complete, no false claims remain, and the codebase is in a consistent, correct state.

---

## Audit Sections

### Section 1: Re-run Phase 0 Search Patterns

Execute all search patterns from Phase 0. Each should return either:
- **0 results** (false claim removed)
- **Results in appropriate context** (e.g., historical reference, decision doc)

```bash
# These should return 0 false claims (may return historical/contextual mentions)
rg -i "novo parity" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "66\\.28%|66\\.28\\b|0\\.6628\\b" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "66\\.3" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "\\[\\[\\s*40\\s*,\\s*19\\s*\\],\\s*\\[\\s*10\\s*,\\s*17\\s*\\]\\]" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "59 specific|27 non-specific|59\\s*/\\s*27" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"

# These should return accurate claims
rg "68\\.6%|68\\.60%|0\\.6860|0\\.686046" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "\\[\\[\\s*40\\s*,\\s*17\\s*\\],\\s*\\[\\s*10\\s*,\\s*19\\s*\\]\\]" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
```

**Pass Criteria:** No false claims found. All matches are in appropriate context.

---

### Section 2: Verify Artifacts

#### 2.1 Data Files

```python
import pandas as pd

# jain_86_novo_parity.csv
df = pd.read_csv("data/test/jain/canonical/jain_86_novo_parity.csv")
assert len(df) == 86
assert (df["label"] == 0).sum() == 57
assert (df["label"] == 1).sum() == 29
assert df[df["id"] == "lebrikizumab"]["label"].values[0] == 1
assert df[df["id"] == "galiximab"]["label"].values[0] == 1
print("✅ jain_86_novo_parity.csv verified")

# VH_only_jain_86_p5e_s2.csv
df_vh = pd.read_csv("data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv")
assert len(df_vh) == 86
assert (df_vh["label"] == 0).sum() == 57
assert (df_vh["label"] == 1).sum() == 29
print("✅ VH_only_jain_86_p5e_s2.csv verified")
```

#### 2.2 Model Checkpoint

```bash
# Verify model exists
ls -la experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

---

### Section 3: Re-run Verification Script

Run the Phase 3 verification script to confirm parity still holds:

```bash
python -m experiments.benchmarks.novo_parity.scripts.verify_parity
```

**Expected Output:**
```
✅ EXACT NOVO PARITY ACHIEVED!
```

---

### Section 4: Code Quality Checks

```bash
# Linting
make lint
# Expected: 0 errors

# Type checking
make typecheck
# Expected: 0 errors

# Tests
make test
# Expected: All tests pass

# E2E tests (if applicable)
make test-e2e
# Expected: All tests pass or expected skips
```

---

### Section 5: Documentation Build

```bash
make docs-build
```

**Check for:**
- [ ] Build completes without errors
- [ ] No warnings about missing files
- [ ] No broken links warnings
- [ ] All pages render correctly

---

### Section 6: Git Status

```bash
git status
```

**Expected:**
- Working tree clean (all changes committed)
- On fix branch: `fix/jain-parity-remediation`

```bash
git log --oneline -10
```

**Verify commits:**
- [ ] Phase 1: Preprocessing fix commit
- [ ] Phase 2: Artifact regeneration commit
- [ ] Phase 3: Verification script commit
- [ ] Phase 4: Documentation fix commits

---

### Section 7: Cross-Reference Check

Verify key documents reference each other correctly:

| Document | Should Reference |
|----------|------------------|
| `docs/bugs/index.md` | jain_parity_decision.md |
| `docs/bugs/jain_parity_decision.md` | jain_parity_reverse_engineering.md |
| `docs/bugs/jain_parity_reverse_engineering.md` | jain_parity_decision.md |
| `docs/research/novo-parity.md` | bugs/jain_parity_decision.md |
| `preprocessing/jain/step2_preprocess_p5e_s2.py` | docs/bugs/jain_parity_decision.md |

---

## Final Checklist

### Preprocessing
- [ ] `step2_preprocess_p5e_s2.py` includes Tier D
- [ ] Code comments explain rationale
- [ ] Reference to decision document included

### Artifacts
- [ ] `jain_86_novo_parity.csv` has 57/29 split
- [ ] `VH_only_jain_86_p5e_s2.csv` has 57/29 split
- [ ] lebrikizumab has label=1
- [ ] galiximab has label=1

### Verification
- [ ] Confusion matrix: `[[40, 17], [10, 19]]`
- [ ] Accuracy: 68.60%
- [ ] Verification script passes

### Documentation
- [ ] All false parity claims removed
- [ ] Accurate results documented
- [ ] Methodology caveats added
- [ ] Cross-references correct

### Code Quality
- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] `make test` passes
- [ ] `make docs-build` passes

### Git
- [ ] All changes committed
- [ ] Clear commit messages
- [ ] Ready for PR

---

## Sign-Off

### Automated Checks

```
[ ] Phase 0 patterns: 0 false claims
[ ] Data artifacts: Verified
[ ] Verification script: PASS
[ ] make lint: PASS
[ ] make typecheck: PASS
[ ] make test: PASS
[ ] make docs-build: PASS
```

### Manual Review

```
[ ] Read through key documentation pages
[ ] Spot-check code comments
[ ] Verify links work
[ ] Review commit history
```

### Final Sign-Off

```
Auditor: _______________
Date: _______________
Status: [ ] APPROVED / [ ] NEEDS WORK
Notes: _______________
```

---

## Post-Remediation Actions

After Phase 5 passes:

1. **Create Pull Request:**
   ```bash
   gh pr create --title "fix: Jain parity remediation (Tier D reclassification)" \
     --body "$(cat <<'EOF'
   ## Summary
   - Adds Tier D reclassification (lebrikizumab + galiximab) to preprocessing
   - Regenerates Jain artifacts with corrected labels (57/29 split)
   - Achieves exact Novo parity: [[40, 17], [10, 19]], 68.60% accuracy
   - Fixes all false parity claims in documentation

   ## Decision
   See docs/bugs/jain_parity_decision.md for the full rationale (triple agent consensus).

   ## Test Plan
   - [ ] Verification script passes
   - [ ] All tests pass
   - [ ] Docs build without errors

   Closes #33
   EOF
   )"
   ```

2. **Update GitHub Issue #33:**
   - Link to PR
   - Mark as resolved pending merge

3. **Notify stakeholders:**
   - Tag relevant reviewers on PR
   - Update any external tracking systems

---

## Rollback Procedure

If issues discovered post-merge:

1. **Revert PR:**
   ```bash
   git revert <merge-commit-sha>
   ```

2. **Investigate:**
   - Identify which phase had the error
   - Create new fix branch

3. **Re-run remediation:**
   - Start from failed phase
   - Re-run all subsequent phases

---

**End of Spec**
