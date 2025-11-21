# JAIN DATASET CORRUPTION: ROOT CAUSE ANALYSIS

**Incident Date:** 2025-11-20
**Status:** RESOLVED
**Impact:** High (Invalidated benchmark results for Pertuzumab)

## 1. The Incident
After the Phase D refactoring (Code Deduplication), the VH sequence for **Pertuzumab** (Jain dataset) was found to be truncated by one amino acid in the generated fragment files (`VH_only_jain.csv`, `Full_jain.csv`, etc.).

- **Correct (Canonical):** `...YWGQGTLVTVSS` (ends in `SS`)
- **Corrupted (Fragment):** `...YWGQGTLVTVS` (ends in `S`)

This affected only Pertuzumab among the 137 antibodies in Jain because its sequence extends slightly beyond the strict IMGT FWR4 boundary used by ANARCI.

## 2. Root Cause
The corruption was caused by a change in logic within the new `preprocessing/fragment_utils.py` module, which replaced dataset-specific extraction scripts.

### Old Logic (Correct)
In `preprocessing/jain/step3_extract_fragments.py` (and similar), the full sequence was taken directly from the ANARCI output's `sequence_aa` field:
```python
f"full_seq_{chain}": annotation.sequence_aa  # Uses full input sequence (gap-free)
```

### New Logic (Incorrect)
In `preprocessing/fragment_utils.py` (Phase D), the full sequence was **reconstructed** by concatenating individual fragments:
```python
fragments[f"full_seq_{chain}"] = "".join([
    fwr1, cdr1, fwr2, cdr2, fwr3, cdr3, fwr4
])
```

This change was intended to strip potential "constant region garbage" from the input. However, ANARCI's definition of `fwr4_aa` (Framework Region 4) for Pertuzumab ends at `...VTVS`. The final `S` of the input sequence `...VTVSS` was considered by ANARCI to be outside the strict FWR4 boundary (or part of the constant region start), and thus was excluded from `fwr4_aa`.

By reconstructing the sequence from these strictly defined fragments, the final `S` was lost.

## 3. Verification
We confirmed this behavior with a reproduction script (`debug_pertuzumab.py`):
- `annotation.sequence_aa` -> `...VTVSS` (Correct)
- `annotation.fwr4_aa` -> `...VTVS` (Truncated)

## 4. The Fix
The fix restores the original behavior for the full sequence field while maintaining the shared utility structure.

**File:** `preprocessing/fragment_utils.py`

```python
# Old (Buggy)
# fragments[f"full_seq_{chain}"] = "".join([...])

# New (Fixed)
fragments[f"full_seq_{chain}"] = annotation.sequence_aa
```

This ensures that `VH_only` and `Full` fragments exactly match the input V-domain sequence provided in the source CSV/Excel, preserving residues that may technically fall outside strict IMGT boundaries but are biologically relevant (e.g., J-region termini).

## 5. Impact Analysis
- **Jain:** Fixed. Pertuzumab sequence restored.
- **Boughter:** Unaffected (input is DNA translation, likely ends at FWR4 boundary anyway).
- **Harvey:** Unaffected (VHH input).
- **Shehata:** Unaffected (Phase 2 script used `sequence_aa`, Phase D refactor momentarily broke it but now fixed).

## 6. Prevention
- Added regression test via `grep` in CI/manual verification.
- Documented `sequence_aa` vs `fwr4_aa` distinction in `fragment_utils.py`.
- **Lesson Learned:** "Cleaning" data by reconstruction is dangerous when reference definitions (IMGT) might disagree with canonical inputs. Trust the source sequence for the full domain.
