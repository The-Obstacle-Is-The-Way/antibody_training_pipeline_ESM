# Gradio App Guide

**Date**: 2025-11-22
**Purpose**: Local development guide and Hugging Face Spaces deployment reference
**Status**: ✅ Validated post-Pydantic v2 integration (Phases 1-4)

---

## Overview

This app runs in **two environments**:

1. **Local Development** (full functionality, full model)
   - Run on your machine for testing/demos
   - Uses full pipeline with all features
   - Hydra config, all models supported

2. **Hugging Face Spaces** (public demo, simplified)
   - Free public hosting at `https://huggingface.co/spaces/yourname/antibody-predictor`
   - Simplified `app.py` (no Hydra, pre-loaded model)
   - Both can coexist - local stays unchanged

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

**Friction encountered**: Direct HTTP POST to `/api/predict` returns 404.

**Root cause (intentional)**: The app is launched with `show_api=False` (`src/antibody_training_esm/cli/app.py:187`), so REST endpoints are hidden for safety. Only the queued Gradio client flow is exposed.

**Solution**: Use `gradio_client` (it reads the app config and hits the queue endpoints):
```python
from gradio_client import Client

client = Client("http://localhost:7860")
result = client.predict("QVQL...", api_name="/predict")
```

**Why**: With `show_api=False` in Gradio 5.x, `/api/*` routes are disabled. `gradio_client` is the supported way to exercise the interface without re-enabling the public API.

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

---

## Deployment Options

### Local Use (Current Setup)

**Perfect for:**
- Development and testing
- Local demos
- Research experiments

**How to run:**
```bash
uv run python -m antibody_training_esm.cli.app \
  classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**Access at:** `http://localhost:7860`

### Hugging Face Spaces (Public Demo)

**Perfect for:**
- Sharing with collaborators worldwide
- Public demos and portfolio
- No server management

**What's different from local:**
- Simplified `app.py` (no Hydra, direct Gradio code)
- Pre-trained model file committed to repo
- Environment variables for secrets
- Free hosting (CPU) or paid GPU

**Setup guide:** See branch `feat/huggingface-spaces-deployment` (coming soon)

### Temporary Public Sharing (ngrok)

**Perfect for:**
- Quick demos without deploying
- Temporary public access

```bash
# Start local app
uv run python -m antibody_training_esm.cli.app classifier.path=...

# In another terminal
ngrok http 7860
# Gives: https://abc123.ngrok.io (share this URL)
```

---

## Hugging Face Spaces Best Practices (2025)

**From official Gradio 5 + HF documentation:**

### Required Files
1. **app.py** - Entry point (HF looks for this file)
2. **requirements.txt** or **pyproject.toml** - Dependencies
3. **README.md** - Space description (displays on HF)
4. **Model files** - Committed to repo or loaded from HF Hub

### Security
- Never hardcode API keys/secrets in code
- Use HF Spaces **Secrets** (Settings → Repository secrets)
- Access via `os.getenv("HF_TOKEN")`

### Performance
- Use `queue=True` for concurrency (handles multiple users)
- Move model loading to global scope (not inside predict function)
- Consider GPU hardware for ESM models (paid tier)

### Discoverability
- Add tags: `gradio`, `antibody`, `protein`, `ESM`
- Upload thumbnail: `thumbnail.png` in repo root
- Set visibility to Public

---

## Status

**Local Development:** ✅ Production ready
- All Pydantic integration validated
- No friction points found
- macOS stability patches applied

**HF Spaces Deployment:** 🚧 In progress
- Branch: `feat/huggingface-spaces-deployment`
- Simplified app.py needed (no Hydra dependency)
- Model file needs to be committed or loaded from HF Hub

---

## Next Steps

1. ✅ Keep local app as-is (full functionality)
2. 🚧 Create HF Spaces branch with simplified `app.py`
3. 🚧 Test deployment on HF Spaces
4. ✅ Both environments coexist independently
