# CLI-Only Deployment to Hugging Face Spaces

**No browser needed!** Everything via `huggingface-cli`.

---

## Quick Start (Automated)

```bash
# 1. Set your HuggingFace username
export HF_USERNAME=your_username

# 2. Run deployment script
./spaces/deploy_cli.sh
```

**Done!** Your Space will be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/antibody-predictor`

---

## Manual CLI Deployment (Step by Step)

### Step 1: Login

```bash
# Login to HuggingFace (opens browser once for token)
huggingface-cli login

# Check login
huggingface-cli whoami
```

### Step 2: Create Space

```bash
# Create Gradio Space
huggingface-cli repo create \
  --repo-type space \
  --space_sdk gradio \
  your_username/antibody-predictor

# Output: Space created at https://huggingface.co/spaces/your_username/antibody-predictor
```

### Step 3: Clone Space Repository

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/your_username/antibody-predictor
cd antibody-predictor

# Enable Git LFS for model files
git lfs install
git lfs track "*.pkl"
```

### Step 4: Copy Files

```bash
# From antibody_training_pipeline_ESM repo:

# Copy app files
cp ../antibody_training_pipeline_ESM/spaces/app.py app.py
cp ../antibody_training_pipeline_ESM/spaces/requirements.txt requirements.txt
cp ../antibody_training_pipeline_ESM/spaces/README.md README.md

# Copy source code
cp -r ../antibody_training_pipeline_ESM/src .
cp ../antibody_training_pipeline_ESM/pyproject.toml .

# Copy model (Git LFS handles large file)
mkdir -p experiments/checkpoints/esm1v/logreg
cp ../antibody_training_pipeline_ESM/experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
   experiments/checkpoints/esm1v/logreg/
```

### Step 5: Push to HuggingFace

```bash
git add .
git commit -m "Initial deployment: Antibody predictor"
git push
```

**Done!** HF Spaces auto-deploys on push.

---

## Advanced CLI Commands

### Check Space Status

```bash
# Get Space info via API
curl https://huggingface.co/api/spaces/your_username/antibody-predictor

# List files in Space
huggingface-cli repo-files list your_username/antibody-predictor --repo-type space
```

### Update Deployed Space

```bash
# Make changes locally
cd antibody-predictor
# ... edit files ...

# Push updates
git add .
git commit -m "Update: improved error handling"
git push

# HF Spaces auto-rebuilds
```

### Delete Space (if needed)

```bash
# Via API
curl -X DELETE \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  https://huggingface.co/api/repos/delete \
  -d '{"repo": "your_username/antibody-predictor", "type": "space"}'

# Or use web UI: Settings → Delete this space
```

### Upload Single File

```bash
# Upload/update just one file
huggingface-cli upload \
  your_username/antibody-predictor \
  app.py \
  --repo-type space
```

---

## Environment Variables

Set these before running `deploy_cli.sh`:

```bash
# Required
export HF_USERNAME=your_username

# Optional (customize Space name)
export HF_SPACE_NAME=my-antibody-app  # Default: antibody-predictor

# Run deployment
./spaces/deploy_cli.sh
```

---

## Authentication Options

### Option 1: Interactive Login (Recommended)

```bash
huggingface-cli login
# Opens browser, paste token, done
```

### Option 2: Token Environment Variable

```bash
# Get token from: https://huggingface.co/settings/tokens
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx

# CLI will use token automatically
huggingface-cli whoami
```

### Option 3: Token File

```bash
# Save token to file
echo "hf_xxxxxxxxxxxxx" > ~/.huggingface/token

# CLI reads from ~/.huggingface/token
huggingface-cli whoami
```

---

## Deployment Script Features

The automated script (`deploy_cli.sh`) does:

1. ✅ Checks login status
2. ✅ Creates Space (if doesn't exist)
3. ✅ Clones Space repo
4. ✅ Configures Git LFS
5. ✅ Copies all files
6. ✅ Commits and pushes
7. ✅ Prints success info + URLs

**Zero manual steps** - just set `HF_USERNAME` and run!

---

## Troubleshooting CLI

### "Not logged in"

```bash
# Solution: Login first
huggingface-cli login
```

### "Permission denied"

```bash
# Solution: Check token has write access
# Go to: https://huggingface.co/settings/tokens
# Create new token with "write" permission
```

### "Space already exists"

```bash
# Solution: Use --exist-ok flag
huggingface-cli repo create \
  --repo-type space \
  --space_sdk gradio \
  --exist-ok \
  your_username/antibody-predictor
```

### "Git LFS not found"

```bash
# macOS
brew install git-lfs
git lfs install

# Linux
sudo apt install git-lfs
git lfs install
```

---

## Comparison: CLI vs Web

| Task | Web UI | CLI |
|------|--------|-----|
| Create Space | Click buttons | `huggingface-cli repo create` |
| Upload files | Drag & drop | `git push` |
| Update app | Web editor | `git commit && git push` |
| Check status | Refresh page | `curl API` |
| **Automation** | ❌ Manual | ✅ Scriptable |

**CLI wins for:**
- Automation (CI/CD)
- Scripting
- Batch operations
- No browser needed

**Web UI wins for:**
- First-time users
- Visual file browser
- Settings management

---

## CI/CD Integration

Use CLI in GitHub Actions:

```yaml
# .github/workflows/deploy-hf.yml
name: Deploy to HF Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to HF Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
        run: |
          pip install huggingface-hub
          ./spaces/deploy_cli.sh
```

Store `HF_TOKEN` in GitHub Secrets.

---

## Summary

**Can everything be done via CLI?**
✅ **YES!**

- Create Space: `huggingface-cli repo create`
- Upload files: `git push`
- Update Space: `git push`
- Check status: `curl` or `huggingface-cli`
- Delete Space: `curl` API

**No browser needed** (except initial token generation).

Run `./spaces/deploy_cli.sh` and you're live in 5 minutes! 🚀
