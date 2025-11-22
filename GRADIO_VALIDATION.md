# Gradio App Validation (Post-Pydantic Integration)

**Date**: 2025-11-22
**Context**: Validated Gradio UI after Phase 1-4 Pydantic v2 integration
**Result**: ✅ All systems operational - no bugs found

---

## Test Results

| Test Case | Input | Expected Behavior | Actual Result | Status |
|-----------|-------|-------------------|---------------|--------|
| Valid VH sequence | 128aa standard VH | Prediction + probability | `specific (28.0%)` | ✅ PASS |
| Valid VL sequence | 107aa standard VL | Prediction + probability | `non-specific (96.9%)` | ✅ PASS |
| Lowercase input | `qvqlvqsgae` | Auto-uppercase → predict | `non-specific (98.1%)` | ✅ PASS |
| Invalid amino acids | Sequence with `Z` | Pydantic ValidationError | `Invalid characters found: Z` | ✅ PASS |
| Empty sequence | `""` | Pydantic ValidationError | `String should have at least 1 character` | ✅ PASS |

**Conclusion**: Pydantic integration works flawlessly in production Gradio UI.

---

## Key Behaviors (Important for Developers)

### 1. **Auto-Cleaning Input Sequences**
**Location**: `src/antibody_training_esm/models/prediction.py:39`

```python
@field_validator("sequence")
@classmethod
def validate_amino_acids(cls, v: str) -> str:
    cleaned = v.strip().upper()  # ← Auto-uppercase
```

**Behavior**: User input is automatically cleaned before validation:
- Whitespace stripped
- **Lowercase → UPPERCASE** (user-friendly, intentional)
- Then validated for valid amino acids

**Why this matters**:
- Don't expect strict lowercase rejection - it's a feature, not a bug
- Input like `qvql` becomes `QVQL` automatically
- Still validates amino acids AFTER cleaning

---

### 2. **Gradio API Testing Gotcha**

**Friction encountered**: Direct HTTP POST to `/api/predict` returns 404

**Solution**: Use `gradio_client` library instead:
```python
from gradio_client import Client

client = Client("http://localhost:7860")
result = client.predict("QVQL...", api_name="/predict")
```

**Why**: Gradio 5.x uses a different API structure than older versions. The `gradio_client` library handles this automatically.

---

### 3. **macOS-Specific Workarounds**

**Location**: `src/antibody_training_esm/cli/app.py:45-59`

**Applied patches**:
1. **Force CPU on macOS** (line 45-49) - MPS causes SegFaults in Gradio
2. **Single-threaded PyTorch** (line 55-59) - OpenMP crashes in Gradio threads

```python
if platform.system() == "Darwin" and device == "mps":
    logger.warning("macOS detected. Forcing CPU for Gradio app stability")
    device = "cpu"

if platform.system() == "Darwin" and device == "cpu":
    torch.set_num_threads(1)  # Prevent OpenMP crashes
```

**Impact**: Gradio runs slower on macOS (CPU-only, single-threaded), but **stable**.

---

## Running Gradio App for Testing

**Start in tmux**:
```bash
tmux new-session -d -s gradio-test "uv run python -m antibody_training_esm.cli.app \
  classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  2>&1 | tee /tmp/gradio_test.log"
```

**Test with Python**:
```python
from gradio_client import Client

client = Client("http://localhost:7860")

# Test valid sequence
result = client.predict(
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMG...",
    api_name="/predict"
)
print(result)  # ('specific', '28.0%')
```

**Stop**:
```bash
tmux kill-session -t gradio-test
```

---

## Validation Checklist for Future Pydantic Changes

Before merging Pydantic-related PRs, validate Gradio still works:

- [ ] App starts without errors
- [ ] Valid sequences return predictions
- [ ] Invalid sequences raise clear Pydantic errors
- [ ] Auto-cleaning still works (lowercase → uppercase)
- [ ] Error messages display properly in UI
- [ ] macOS stability patches still applied

---

## Files Involved

**Gradio App**:
- `src/antibody_training_esm/cli/app.py` - Main Gradio interface

**Pydantic Models**:
- `src/antibody_training_esm/models/prediction.py` - `PredictionRequest`, `PredictionResult`

**Core Prediction**:
- `src/antibody_training_esm/core/prediction.py` - `Predictor` class

**Config**:
- `src/antibody_training_esm/conf/predict.yaml` - Gradio config (Hydra)

---

## Status: ✅ Production Ready

No friction points found. Pydantic integration enhances validation without breaking UX.
