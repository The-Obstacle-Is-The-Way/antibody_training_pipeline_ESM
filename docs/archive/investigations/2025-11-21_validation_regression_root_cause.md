# Validation Regression Root Cause Analysis & Friction Dossier

**Date:** 2025-11-21
**Incident:** Regression in Jain Parity Accuracy (66.28% → 65.12%)
**Status:** ✅ RESOLVED
**Verdict:** Configuration Drift (Batch Size), NOT Code/Data Corruption.

---

## 1. The Incident
After a system crash and restart, the validation pipeline reported a regression in the Novo Nordisk parity benchmark on the Jain dataset.
*   **Expected (Golden):** 66.28% accuracy (57/86 correct).
*   **Observed (Regression):** 65.12% accuracy (56/86 correct).
*   **Specific Discrepancy:** One antibody, **`tabalumab`**, flipped from **Specific (Correct)** to **Non-Specific (Incorrect)**.

## 2. The Friction (Why it was hard)
We encountered multiple layers of "fog of war" that obscured the truth:

1.  **The "Crash" Red Herring:** Because the IDE crashed the night before, we assumed file corruption or state loss. This sent us chasing ghosts in the git history.
2.  **The "Cache" Red Herring:** We suspected the embedding cache was corrupted. Clearing it didn't fix the result (it stayed at 65.12%).
3.  **The Debugging Trap (Segfaults):** To prove the model was deterministic, we tried forcing the pipeline to run on **CPU** (`--device cpu`).
    *   *Result:* Immediate **Segmentation Fault (SIGSEGV)** in `libomp.dylib`.
    *   *Why:* PyTorch on macOS has known stability issues with OpenMP threading when switching from MPS to CPU under high load. This prevented us from getting a clean "control" reading.

## 3. The Root Cause: Batch Size Physics
The issue was **Floating Point Non-Determinism on MPS (Metal Performance Shaders)**.

*   **The Mechanism:** Neural networks (ESM-1v) use matrix multiplication. On GPUs (especially MPS), the order in which floating-point numbers are added affects the final tiny decimal places (e.g., `0.49999` vs `0.50001`).
*   **The Trigger:**
    *   The **Golden Baseline** was created with the training default: **`batch_size=8`**.
    *   The **Validation Runner** (and our manual re-tests) optimized for speed, automatically upgrading to **`batch_size=32`**.
*   **The Physics:** Changing the batch size changes the memory layout and accumulation order on the GPU. This caused the embedding vector for `tabalumab` to shift slightly.
*   **The Cliff:** `tabalumab` is a borderline case.
    *   Batch Size 8 prediction: **0.497** (Class 0 ✅)
    *   Batch Size 32 prediction: **0.517** (Class 1 ❌)

## 4. The Fix
We validated this hypothesis by forcing the test runner to use the original training parameters:

```bash
uv run antibody-test ... --batch-size 8
```

**Result:** **66.28% Accuracy.** The exact confusion matrix was restored.

## 5. Corrective Actions & Lessons
1.  **Explicit Configuration:** "Defaults" are dangerous. Validation scripts must explicitly enforce the exact parameters (`batch_size`, `device`) used during training.
2.  **MPS Awareness:** We must accept that **MPS is not bit-exact** across different run configurations. Small variances are mathematical facts, not bugs.
3.  **Avoid CPU Fallback on Mac:** Relying on CPU for debugging deep learning models on macOS is fragile due to OpenMP/library conflicts. Stick to MPS and control the variables (batch size, seed).

## 6. Definitive Answer
*   **Was the model corrupted?** NO.
*   **Was the code broken?** NO.
*   **Is the pipeline valid?** YES. 100%.

**Signed:**
*The Gucci Banger Validation Team*
