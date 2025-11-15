# Archive - Novo Parity Experiments

This folder contains **outdated or superseded documentation** from the Novo parity reverse-engineering effort.

---

## Archived Files

### REVERSE_ENGINEERING_SUCCESS_P5_OUTDATED.md
**Date Archived**: 2025-11-08
**Reason**: Superseded by EXACT_MATCH_FOUND.md

**What it claimed:**
- Permutation P5 was the answer
- Confusion Matrix: `[[40, 19], [9, 18]]` - Only 2 cells off from Novo
- Accuracy: 67.44% (better than Novo's 66.28%)
- Recommended using this as "good enough"

**Why it's outdated:**
- Further iteration discovered **P5e-S2** and **P5e-S4** achieve **EXACT MATCH**
- Exact confusion matrix: `[[40, 19], [10, 17]]` matches Novo perfectly
- Exact accuracy: 66.28% matches Novo perfectly
- The key differences:
  1. Swap olaratumab → eldelumab in reclassification (use lowest Tm outlier)
  2. Add tiebreaker for PSR=0 antibodies (AC-SINS or Tm)

**Current source of truth:**
- Documentation: `EXACT_MATCH_FOUND.md` and `MISSION_ACCOMPLISHED.md`
- Canonical dataset: `data/test/jain/canonical/jain_86_novo_parity.csv` (based on P5e-S2)
- Audit trail: `results/permutations/P5e_S2_final_audit.json`

---

## Why Archive Instead of Delete?

Archived docs preserve the **scientific discovery process** and show:
1. How P5 was the initial breakthrough (perfect top row)
2. The iterative refinement process
3. The importance of tiebreakers for PSR=0 antibodies
4. Historical context for understanding the final solution

**Never delete scientific provenance - archive it!**
