#!/bin/bash
# Hugging Face Spaces Deployment - Full CLI Automation
# No browser needed - everything via huggingface-cli

set -e  # Exit on error

echo "🚀 Hugging Face Spaces CLI Deployment"
echo "======================================"
echo ""

# Configuration
SPACE_NAME="${HF_SPACE_NAME:-antibody-predictor}"
HF_USERNAME="${HF_USERNAME}"

if [ -z "$HF_USERNAME" ]; then
    echo "❌ Error: HF_USERNAME environment variable not set"
    echo "   Set it with: export HF_USERNAME=your_username"
    exit 1
fi

SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"
REPO_DIR="/tmp/hf_space_${SPACE_NAME}"

echo "📝 Configuration:"
echo "   Space ID: ${SPACE_ID}"
echo "   Local dir: ${REPO_DIR}"
echo ""

# Step 1: Login check
echo "🔐 Step 1: Checking HuggingFace login..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "   Not logged in. Running login..."
    huggingface-cli login
else
    USERNAME=$(huggingface-cli whoami | head -1)
    echo "   ✅ Logged in as: ${USERNAME}"
fi
echo ""

# Step 2: Create Space
echo "🌟 Step 2: Creating Hugging Face Space..."
if huggingface-cli repo create \
    --repo-type space \
    --space_sdk gradio \
    --exist-ok \
    "${SPACE_ID}"; then
    echo "   ✅ Space created (or already exists): ${SPACE_ID}"
else
    echo "   ❌ Failed to create Space"
    exit 1
fi
echo ""

# Step 3: Clone Space repo
echo "📥 Step 3: Cloning Space repository..."
rm -rf "${REPO_DIR}"
if git clone "https://huggingface.co/spaces/${SPACE_ID}" "${REPO_DIR}"; then
    echo "   ✅ Repository cloned"
else
    echo "   ❌ Failed to clone Space"
    exit 1
fi
cd "${REPO_DIR}"
echo ""

# Step 4: Enable Git LFS for model files
echo "🔧 Step 4: Configuring Git LFS..."
git lfs install
git lfs track "*.pkl"
git add .gitattributes
echo "   ✅ Git LFS configured for .pkl files"
echo ""

# Step 5: Copy files from antibody_training_pipeline_ESM
echo "📦 Step 5: Copying files from main repo..."
MAIN_REPO="/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM"

# Copy app.py (HF Spaces entry point)
cp "${MAIN_REPO}/spaces/app.py" app.py
echo "   ✅ Copied app.py"

# Copy requirements
cp "${MAIN_REPO}/spaces/requirements.txt" requirements.txt
echo "   ✅ Copied requirements.txt"

# Copy README
cp "${MAIN_REPO}/spaces/README.md" README.md
echo "   ✅ Copied README.md"

# Copy source code
cp -r "${MAIN_REPO}/src" .
echo "   ✅ Copied src/"

# Copy pyproject.toml (for package install)
cp "${MAIN_REPO}/pyproject.toml" .
echo "   ✅ Copied pyproject.toml"

# Copy trained model
echo "   📊 Copying trained model (this may take a moment)..."
mkdir -p experiments/checkpoints/esm1v/logreg
cp "${MAIN_REPO}/experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl" \
   experiments/checkpoints/esm1v/logreg/
echo "   ✅ Copied model file"
echo ""

# Step 6: Commit and push
echo "🚢 Step 6: Pushing to Hugging Face Spaces..."
git add .
git commit -m "Initial deployment: Antibody non-specificity predictor

- ESM-1v (650M) + Logistic Regression
- Trained on Boughter dataset
- Pydantic v2 validation
- Gradio 5.x UI
"

if git push; then
    echo "   ✅ Pushed to Hugging Face"
else
    echo "   ❌ Failed to push"
    exit 1
fi
echo ""

# Step 7: Success!
echo "✨ Deployment Complete!"
echo "========================"
echo ""
echo "🌐 Your Space is live at:"
echo "   https://huggingface.co/spaces/${SPACE_ID}"
echo ""
echo "📊 Monitor build logs:"
echo "   huggingface-cli repo-files list ${SPACE_ID} --repo-type space"
echo ""
echo "🔍 Check Space status:"
echo "   curl https://huggingface.co/api/spaces/${SPACE_ID}"
echo ""
echo "🗑️  Clean up local clone:"
echo "   rm -rf ${REPO_DIR}"
echo ""
echo "⏱️  Note: First build takes ~5-10 minutes (downloading ESM model)"
echo "    Visit the Space URL to see build progress"
echo ""
