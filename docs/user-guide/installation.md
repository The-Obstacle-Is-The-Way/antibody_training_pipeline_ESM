# Installation Guide

This guide walks you through setting up the antibody training pipeline on your local machine.

---

## Prerequisites

### System Requirements

- **Operating System:** Linux, macOS, or Windows
- **Python:** 3.12 or later
- **Git:** For cloning the repository
- **Disk Space:** ~10 GB for dependencies and cached embeddings
- **Memory:** 8 GB RAM minimum (16 GB recommended for training)
- **GPU (Optional):** CUDA-compatible GPU or Apple Silicon (MPS) for faster embedding extraction

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM.git
cd antibody_training_pipeline_ESM
```

### 2. Install uv Package Manager

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management with virtual environments.

**Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (using pip):**

```bash
pip install uv
```

### 3. Set Up Python Environment

**Linux / macOS:**

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install all dependencies
uv sync
```

**Windows:**

```bash
# Create virtual environment
uv venv

# Activate virtual environment
.venv\Scripts\activate

# Install all dependencies
uv sync
```

### 4. Verify Installation

Run a quick test to ensure everything is installed correctly:

```bash
# Test imports
uv run python -c "import antibody_training_esm; print('✅ Installation successful!')"

# Check installed commands
uv run antibody-train --help
uv run antibody-test --help
```

You should see the help messages for the training and testing commands.

---

## Development Installation (Optional)

If you plan to contribute code or run tests, install development dependencies:

```bash
# Install with all extras (dev tools, testing, linting)
uv sync --all-extras

# Install pre-commit hooks (auto-run quality checks on commits)
uv run pre-commit install

# Verify development setup
make all  # Runs format, lint, typecheck, test
```

---

## GPU Support

### CUDA (NVIDIA GPUs)

If you have an NVIDIA GPU, install CUDA toolkit (11.8 or later):

```bash
# Install CUDA from https://developer.nvidia.com/cuda-downloads

# Verify CUDA is available
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Apple Silicon (MPS)

If you're on Apple Silicon (M1/M2/M3), PyTorch will automatically use Metal Performance Shaders (MPS):

```bash
# Verify MPS is available
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Note:** If you encounter MPS memory issues, see [Troubleshooting Guide](troubleshooting.md#mps-memory-issues).

---

## Directory Structure After Installation

After installation, your directory structure will look like:

```
antibody_training_pipeline_ESM/
├── .venv/                      # Virtual environment (created by uv venv)
├── src/                        # Source code
│   └── antibody_training_esm/
├── configs/                    # Configuration files
├── preprocessing/              # Dataset preprocessing scripts
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml             # Project dependencies
└── README.md                  # Project overview
```

---

## Common Installation Issues

### Issue: `uv` command not found

**Solution:** Restart your terminal after installing `uv`, or add `~/.cargo/bin` to your `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue: Python version mismatch

**Solution:** This project requires Python 3.12+. Install the correct version:

```bash
# Check current version
python --version

# Install Python 3.12 (example for Ubuntu)
sudo apt update
sudo apt install python3.12

# Or use pyenv for version management
pyenv install 3.12
pyenv local 3.12
```

### Issue: Permission denied on macOS/Linux

**Solution:** Don't use `sudo` with `uv`. If you encounter permission issues, check your Python installation ownership:

```bash
# Fix ownership (replace YOUR_USERNAME)
sudo chown -R YOUR_USERNAME:YOUR_USERNAME ~/.local
```

---

## Next Steps

After installation:

1. **Quick Start:** Follow the [Getting Started Guide](getting-started.md) for a 5-minute quickstart
2. **Training:** See [Training Guide](training.md) to train your first model
3. **Testing:** See [Testing Guide](testing.md) to evaluate models on test sets

---

## Uninstallation

To remove the pipeline:

```bash
# Deactivate virtual environment
deactivate

# Remove the repository
cd ..
rm -rf antibody_training_pipeline_ESM
```

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
