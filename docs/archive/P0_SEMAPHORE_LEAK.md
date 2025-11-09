# P0 Bug: Semaphore Leak & Double Model Loading

**Status:** 🔴 CRITICAL - Crashes Harvey benchmark (141k sequences)
**Root Cause:** Amateur object lifecycle management - orphaned GPU models
**Discovered:** 2025-11-06 (Harvey crashed at 15% with semaphore leak warning)

---

## Executive Summary

The testing pipeline (`src/antibody_training_esm/cli/test.py`) creates **TWO COPIES** of the 650M parameter ESM-1v model when loading pickled classifiers, causing:
- 2x memory consumption (1.3GB → 2.6GB)
- GPU resource leaks (MPS device on Mac)
- **Semaphore leaks** that crash large dataset benchmarks
- System instability on datasets >1000 sequences

Small datasets (Jain 86, Shehata 398) survive the leak. **Harvey (141k sequences) crashes at ~15% completion** with Python multiprocessing semaphore warnings.

---

## Bug Trace (Execution Flow)

### Step 1: Unpickling Creates First Model
**File:** `src/antibody_training_esm/core/classifier.py:230-239`

```python
def __setstate__(self, state):
    """Custom unpickle method - recreate ESM model with correct config"""
    self.__dict__.update(state)
    batch_size = getattr(self, "batch_size", 32)
    self.embedding_extractor = ESMEmbeddingExtractor(
        self.model_name, self.device, batch_size  # ← self.device = "mps" (from training)
    )
```

**Result:** ESM-1v (650M params) loaded on **MPS device** (Apple Silicon GPU)

**Log Evidence:**
```
Line 16: ESM model facebook/esm1v_t33_650M_UR90S_1 loaded on mps with batch_size=8
```

---

### Step 2: Test CLI Immediately Creates Second Model
**File:** `src/antibody_training_esm/cli/test.py:122-135`

```python
def load_model(self, model_path: str) -> BinaryClassifier:
    # ... (unpickle happens above, triggering __setstate__) ...

    # Update device if different from config
    if (
        hasattr(model, "embedding_extractor")
        and model.embedding_extractor.device != self.config.device  # "mps" != "cpu"
    ):
        self.logger.info(
            f"Updating device from {model.embedding_extractor.device} to {self.config.device}"
        )
        # 🔴 BUG: Creates NEW extractor WITHOUT cleaning up old one
        batch_size = getattr(model, "batch_size", 32)
        model.embedding_extractor = ESMEmbeddingExtractor(
            model.model_name, self.config.device, batch_size  # ← config.device = "cpu"
        )
        model.device = self.config.device
```

**Result:** ESM-1v (650M params) loaded **AGAIN** on **CPU device**

**Log Evidence:**
```
Line 17: Updating device from mps to cpu
Line 20: ESM model facebook/esm1v_t33_650M_1 loaded on cpu with batch_size=8
```

---

### Step 3: First Model Orphaned (LEAK)

The first `ESMEmbeddingExtractor` is **orphaned but not destroyed**:
- Still holds 650M parameters in GPU memory
- GPU resources (MPS semaphores, CUDA contexts) NOT released
- Python garbage collector won't clean up until much later
- Each test run leaks another 650M model

**Crash Evidence (Harvey @ 15%):**
```python
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

---

## Why Small Datasets Survive, Harvey Crashes

| Dataset | Size | Outcome | Reason |
|---------|------|---------|--------|
| Jain | 86 sequences | ✅ Survives | Leak is small, finishes before OOM |
| Shehata | 398 sequences | ✅ Survives | Moderate leak, system has enough RAM |
| Harvey | 141,021 sequences | 🔴 **CRASHES @ 15%** | Leak accumulates, Mac runs out of resources |

**Harvey Timeline:**
- Batch 1/4407: Leak starts (2x model in memory)
- Batch 660/4407 (~15%): Semaphore leak warning appears
- System crash shortly after (Mac exhausts GPU/CPU resources)

---

## Root Cause Analysis

### Amateur Mistake: Object Reassignment ≠ Cleanup

```python
# WRONG (what the n00b code does):
model.embedding_extractor = ESMEmbeddingExtractor(...)  # Old object orphaned

# RIGHT (what pros do):
if hasattr(model, 'embedding_extractor'):
    del model.embedding_extractor  # Explicit cleanup
    torch.cuda.empty_cache()  # For CUDA
    torch.mps.empty_cache()   # For MPS (Apple Silicon)
model.embedding_extractor = ESMEmbeddingExtractor(...)
```

### Why This Matters for GPU Models

Normal Python objects (strings, dicts, lists) can be reassigned freely - garbage collector handles it. **GPU-backed models are different:**

1. Hold external resources (GPU memory, CUDA/MPS contexts, semaphores)
2. Require **explicit cleanup** before reassignment
3. Python GC doesn't know about GPU resources → leaks persist

This is CS 101 resource management. The original developer didn't understand:
- Object lifecycle in Python
- GPU resource management
- RAII (Resource Acquisition Is Initialization) patterns

---

## Device Configuration Mismatch

**Training Config (`configs/config.yaml`):**
```yaml
model:
  device: "mps"  # Apple Silicon GPU
```

**Test Config (`src/antibody_training_esm/cli/test.py:65`):**
```python
@dataclass
class TestConfig:
    device: str = "cpu"  # ← HARDCODED DEFAULT, doesn't match training
```

**Result:** EVERY test run triggers the double-loading bug because `"mps" != "cpu"` is always true.

---

## The Fix (3 Options)

### Option 1: Explicit Cleanup (Safest)
```python
# test.py:122-135
if (
    hasattr(model, "embedding_extractor")
    and model.embedding_extractor.device != self.config.device
):
    # EXPLICIT CLEANUP
    old_extractor = model.embedding_extractor
    del model.embedding_extractor
    del old_extractor

    # Clear device-specific cache
    if self.config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif self.config.device.startswith("mps"):
        torch.mps.empty_cache()

    # NOW create new extractor
    model.embedding_extractor = ESMEmbeddingExtractor(
        model.model_name, self.config.device, batch_size
    )
```

### Option 2: Don't Auto-Create in __setstate__ (Better Architecture)
```python
# classifier.py:230-239
def __setstate__(self, state):
    """Custom unpickle method - restore state WITHOUT creating extractor"""
    self.__dict__.update(state)
    # Don't create embedding_extractor here - let caller handle it
    self.embedding_extractor = None
```

Then explicitly create in `load_model()` ONCE.

### Option 3: Consistent Device Config (Easiest)
```python
# test.py:65
@dataclass
class TestConfig:
    device: str = "mps"  # Match training config - no recreation needed
```

**Downside:** Requires MPS-capable hardware. Not portable to Linux/CUDA.

---

## Professional Standard

This bug reveals fundamental gaps in the original codebase:

❌ **What the n00b did:**
- Reassign objects without cleanup
- Ignore GPU resource lifecycle
- Hardcode configs that don't match training
- No testing on large datasets (Harvey was never validated)

✅ **What pros do:**
- Explicit resource cleanup (`del`, cache clearing)
- Consistent device configs across train/test
- Test on production-scale data BEFORE shipping
- Understand RAII and object lifecycle

---

## Recommended Action

1. **Immediate:** Use Option 1 (explicit cleanup) to fix the leak
2. **Short-term:** Use Option 3 (consistent device="mps") to avoid recreation
3. **Long-term:** Refactor to Option 2 (better architecture - lazy extractor creation)

---

## Validation Plan

After fix, verify on Harvey:
```bash
# Should complete all 4,407 batches without semaphore leaks
tmux new-session -d -s test_harvey_clean
uv run python -m antibody_training_esm.cli.test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/VHH_only_harvey.csv \
  --output-dir test_results \
  --device mps  # Explicit device override
```

Monitor for:
- No "Updating device from X to Y" messages (device should match from start)
- No semaphore leak warnings
- Smooth progress through all 4,407 batches
- ~6 hour completion time (5.17 sec/batch)

---

**Written:** 2025-11-06
**Author:** Root cause analysis after Harvey benchmark crash
**Status:** Bug identified, fix pending validation
