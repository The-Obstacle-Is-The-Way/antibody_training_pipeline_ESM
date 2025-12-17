# Jain Parity Remediation Specs — Adversarial Audit

**Date:** 2025-12-16  
**Scope:** Audit of `docs/bugs/*` remediation specs + codebase reality check (no pipeline/data changes executed).  
**Goal:** Ensure the specs are *complete*, *accurate*, and will not leave inconsistencies after execution.

---

## Executive Summary

**Verdict:** Specs were **not IRONCLAD** as written. Key gaps would have caused missed false-claim references and failing tests after artifact regeneration.  
**Primary blockers found:**
- Phase 0 search patterns miss important file types/locations (`.github/workflows/*.yml`, `validation/baseline/*`, `experiments/benchmarks/*.yaml`, etc.).
- Phase 1 spec does not match the actual structure/constraints of `preprocessing/jain/step2_preprocess_p5e_s2.py`.
- Multiple **tests hardcode the current 59/27 split** and will break after switching to 57/29.

**Action taken in this audit:** Updated the remediation specs to cover these gaps (see “Revised Specs” below).

**Confidence after spec fixes:** **HIGH** (remaining uncertainty is “did we miss a file that references the old numbers?”, mitigated by strengthened Phase 0/Phase 5 searches).

---

## Issues Found

### 1) Phase 0 audit would miss critical false-claim locations
- Current spec searches are mostly `--type md --type py`. That **misses**:
  - `.github/workflows/benchmark.yml` (contains old matrix + old accuracy references and “expected” language)
  - `validation/baseline/model_outputs/baseline_metrics.txt` (contains `0.6628`)
  - `validation/baseline/checksums/jain_preprocessed.md5` (pins old artifact checksums)
  - `experiments/benchmarks/**` result `.yaml` and `.log` files (can contain old results, even if historical)

### 2) Phase 1 spec mismatches current preprocessing script behavior
`preprocessing/jain/step2_preprocess_p5e_s2.py` currently:
- Asserts intermediate counts (`89/27`) and final counts (`59/27`) and prints the “our result” matrix.
- The spec proposed “Tier D inside the reclassification list” without accounting for:
  - Step 4 keep-count logic (currently keeps 59 specifics)
  - Assertion updates
  - Potential membership drift (changing the selection set, not just labels)

### 3) Tests will break after switching to 57/29 unless explicitly updated
Hardcoded expectations exist in:
- `tests/integration/test_jain_stage_filtering.py` (expects parity stage and canonical file to be 59/27)
- `tests/unit/datasets/test_jain.py` (asserts **59/27** for the `remove_30_by_psr_acsins` intermediate output; this only needs changing if Tier D is moved *before* the selection step)

### 4) Derived “baseline” artifacts will become inconsistent
If the canonical Jain labels change, the following become stale and should be updated or explicitly treated as historical:
- `validation/baseline/checksums/jain_preprocessed.md5`
- `validation/baseline/model_outputs/baseline_metrics.txt`

### 5) Criterion phrasing risk (“Chromatography + Model”)
Specs described Tier D as “chromatography + model predicted non-specific”. That reads like **using model output to decide labels**, which is not defensible as a preprocessing rule and invites “trial-and-error fitting” criticism. Model predictions should be described as **explanatory** (why FP→TP happens), not as a labeling criterion.

### 6) Spec command correctness (ripgrep type)
Some draft commands used `rg --type yml`, but ripgrep does not define a `yml` type by default. Use `--type yaml` (which includes `*.yml` and `*.yaml`). The specs were updated so the Phase 0 / Phase 5 commands are runnable as written.

---

## Risks Identified

- **Parity regression risk (membership drift):** Implementing Tier D *before* the removal step likely changes which antibodies survive selection. That could break the “only FP/TP move by 2” property and might not reproduce the Novo matrix without additional work.
- **Undocumented dependency risk:** CI workflow summaries and baseline files may remain stale even if core code/data are fixed, causing ongoing confusion.
- **Test suite breakage:** Failing integration/unit tests will block merge if not updated in the plan.

---

## Recommendations (Now Incorporated Into Specs)

1. **Strengthen Phase 0 / Phase 5 search coverage**
   - Search `.yml/.yaml`, `.txt`, `.json`, selected `.log` locations, and `.github/workflows`.
   - Use robust regex for confusion matrices (optional whitespace) to avoid missing `[[40,19],[10,17]]` formatting.

2. **Implement Tier D as a final-label adjustment on the 86-set (low-risk path)**
   - Keep the 86 membership stable; flip labels for `lebrikizumab` and `galiximab` on the final dataset.
   - This matches the evidence that parity can be achieved by label flips with predictions held fixed.

3. **Update `src/antibody_training_esm/datasets/jain.py` parity stage + tests**
   - Parity stage should return the same 86 membership but **57/29** label distribution.
   - Update integration tests that assert 59/27.

4. **Treat baseline files as derived artifacts**
   - Update checksums and baseline metrics after regeneration (or explicitly label them historical and exclude them from “no false claims remain” gates).

5. **Add rollback guidance**
   - Define how to revert artifacts + code if verification fails (e.g., `git revert` the regeneration commit; restore prior checksums/baselines).

---

## Confirmations (What’s Correct/Complete Already)

- All spec file paths exist under `docs/bugs/`.
- Model checkpoint referenced by specs exists: `experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl`.
- Canonical artifacts exist: `data/test/jain/canonical/jain_86_novo_parity.csv`, `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv`.
- Proof artifact exists and matches the documented 3 matching pairs: `experiments/benchmarks/novo_parity/results/phase2b_results.json`.

---

## Revised Specs (Summary of Changes)

Updated to close the gaps above:
- `docs/bugs/JAIN_PARITY_REMEDIATION_PLAN.md`
- `docs/bugs/SPEC_phase0_audit.md`
- `docs/bugs/SPEC_phase1_preprocessing.md`
- `docs/bugs/SPEC_phase2_artifacts.md`
- `docs/bugs/SPEC_phase4_documentation.md`
- `docs/bugs/SPEC_phase5_final_audit.md`

All edits are marked with **AUDITOR NOTE (2025-12-16)** blocks inside the spec files.
