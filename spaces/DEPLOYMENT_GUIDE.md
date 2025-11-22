# Hugging Face Spaces Deployment Guide

**Goal:** Deploy antibody prediction demo to Hugging Face Spaces
**Time:** ~30 minutes
**Cost:** Free (CPU tier)

---

## Prerequisites

1. **Hugging Face Account**
   Sign up at https://huggingface.co/join

2. **Trained Model**
   Ensure you have a trained model file:
   ```bash
   experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
   ```

3. **Git LFS** (for model files)
   ```bash
   brew install git-lfs  # macOS
   git lfs install
   ```

---

## Step 1: Create Space on Hugging Face

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Name:** `antibody-predictor` (or your choice)
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
3. Click **Create Space**

---

## Step 2: Clone Space Repository

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/antibody-predictor
cd antibody-predictor
```

---

## Step 3: Copy Files from This Repo

```bash
# From antibody_training_pipeline_ESM root:

# 1. Copy app.py (HF Spaces entry point)
cp spaces/app.py app.py

# 2. Copy requirements
cp spaces/requirements.txt requirements.txt

# 3. Copy README (HF Spaces metadata)
cp spaces/README.md README.md

# 4. Copy source code
cp -r src .

# 5. Copy pyproject.toml (for local package install)
cp pyproject.toml .

# 6. Copy trained model (use Git LFS)
git lfs track "*.pkl"
mkdir -p experiments/checkpoints/esm1v/logreg
cp experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
   experiments/checkpoints/esm1v/logreg/
```

---

## Step 4: Adjust Model Path (if needed)

Edit `app.py` if your model is at a different path:

```python
# Line ~25
MODEL_PATH = os.getenv(
    "MODEL_PATH", "experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
)
```

---

## Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment: Antibody non-specificity predictor"
git push
```

**HF Spaces will automatically:**
1. Detect `app.py` as entry point
2. Install dependencies from `requirements.txt`
3. Install local package with `pip install -e .`
4. Launch Gradio app on port 7860

---

## Step 6: Monitor Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/antibody-predictor`
2. Check **Logs** tab for build progress
3. Wait 5-10 minutes for first build (downloading ESM model)

**Expected logs:**
```
Installing requirements...
Installing antibody_training_esm package...
Loading model from experiments/checkpoints/...
Warming up model...
Model ready!
Running on http://0.0.0.0:7860
```

---

## Step 7: Test Deployment

1. Open your Space URL
2. Paste test sequence:
   ```
   QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS
   ```
3. Click "🔬 Predict Non-Specificity"
4. Should get: `specific (28.0%)`

✅ **If this works, you're live!**

---

## Optional Enhancements

### Add GPU Support (Paid)

1. Go to Space **Settings**
2. Change Hardware to **T4 small** (~$0.60/hour)
3. Edit `app.py` line ~30:
   ```python
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
   ```

### Add Authentication

1. Go to Space **Settings** → **Secrets**
2. Add secret: `HF_TOKEN` (your HF access token)
3. Enable **Private** visibility
4. Share via direct link only

### Custom Domain

1. Upgrade to HF Pro ($9/month)
2. Add custom domain in Space settings

---

## Troubleshooting

### Build fails with "Out of disk space"

**Solution:** Model file too large for free tier
- Option 1: Upload model to HF Hub, load from there
- Option 2: Use smaller model (ESM2-150M)

### App crashes on startup

**Check logs for:**
```
ModuleNotFoundError: No module named 'antibody_training_esm'
```

**Fix:** Ensure `pyproject.toml` exists and `pip install -e .` runs

### Model inference too slow

**Solutions:**
- Reduce `batch_size` in code
- Upgrade to GPU hardware
- Use quantized model

---

## Differences from Local App

| Feature | Local (`src/antibody_training_esm/cli/app.py`) | HF Spaces (`spaces/app.py`) |
|---------|-----------------------------------------------|----------------------------|
| Config system | Hydra (YAML configs) | Direct Python (no Hydra) |
| Model loading | Dynamic path via CLI args | Hardcoded path or env var |
| Dependencies | Full dev environment | Minimal (requirements.txt) |
| Device support | CPU/CUDA/MPS | CPU (or paid GPU) |
| Deployment | Manual (local or server) | Automatic (git push) |

**Both apps:**
- ✅ Use same Pydantic validation
- ✅ Use same Predictor class
- ✅ Use same ESM model
- ✅ Same auto-cleaning (lowercase → uppercase)

---

## Maintenance

**Update model:**
```bash
# Copy new model file
cp ../antibody_training_pipeline_ESM/experiments/checkpoints/.../new_model.pkl .

# Commit and push
git add new_model.pkl
git commit -m "Update model to v2"
git push
```

**Update code:**
```bash
# Make changes to app.py
git add app.py
git commit -m "Fix: improved error handling"
git push
```

HF Spaces auto-rebuilds on every push.

---

## Success Checklist

- [ ] Space created on HuggingFace.co
- [ ] Files copied (app.py, requirements.txt, README.md, src/, model)
- [ ] Git LFS configured for .pkl files
- [ ] Pushed to HF Spaces repo
- [ ] Build completed successfully (check logs)
- [ ] Test prediction works
- [ ] Space is public and accessible

---

## Next Steps

1. Share your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/antibody-predictor`
2. Add to your portfolio/publications
3. Monitor usage in HF Analytics
4. Consider upgrading to GPU if popular

**Local development stays unchanged** - keep using `uv run python -m antibody_training_esm.cli.app` for full features.
