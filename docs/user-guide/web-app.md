# Web Application Guide

This guide explains how to use the **Antibody Non-Specificity Predictor** web interface, powered by Gradio. This local web app provides a user-friendly way to screen antibody sequences interactively.

## 1. Quick Start

To launch the app, you need a trained model checkpoint (see [Training Guide](training.md)).

```bash
# Using a standard Pickle checkpoint
uv run antibody-app classifier.path=experiments/checkpoints/esm1v/logreg/model.pkl
```

Once launched, open your browser to the URL displayed in the terminal (usually `http://127.0.0.1:7860`).

---

## 2. Features

- **Interactive Interface**: Paste single sequences and get instant feedback.
- **Real-time Validation**: Checks for invalid characters before running the heavy model.
- **Probability Score**: Displays the exact confidence score (0-100%).
- **Examples included**: Click on pre-loaded examples to test the system immediately.

---

## 3. Supported Model Formats

The app supports both development and production model formats:

### Option A: Development (`.pkl`)

Use this for quick iteration during research.

```bash
uv run antibody-app classifier.path=path/to/model.pkl
```

### Option B: Production (`.npz` + `.json`)

Use this for secure, pickle-free deployment.

```bash
uv run antibody-app \
    classifier.path=experiments/checkpoints/esm1v/logreg/model.npz \
    classifier.config_path=experiments/checkpoints/esm1v/logreg/model_config.json
```

*Note: If `config_path` is omitted, the app will look for a JSON file with the same name as the NPZ file (e.g., `model_config.json`).*

---

## 4. macOS Optimization (Important)

Running heavy ML models on macOS can be tricky due to conflicts between PyTorch's MPS (Metal Performance Shaders) and Gradio's threading model.

**We have solved this for you automatically.**

When you run `antibody-app` on a Mac:
1. **Auto-Downgrade**: It forces the model to run on **CPU** (even if you request MPS).
2. **Single-Threading**: It restricts PyTorch to a **single thread** (`torch.set_num_threads(1)`).

**Why?**
This prevents a known "Segmentation Fault" crash caused by the OpenMP library when used inside Gradio's multi-threaded web server.

**Performance Impact:**
Inference will take ~1-2 seconds per sequence instead of <0.5s. For interactive use, this is negligible. If you need high-throughput batch processing on Mac, use the CLI tool (`antibody-predict`) instead, which fully supports MPS acceleration.

---

## 5. Advanced Configuration

Since the app is built on Hydra, you can override any configuration parameter from the command line:

```bash
# Change the sequence length limit or model architecture
uv run antibody-app \
    classifier.path=model.pkl \
    model.name=facebook/esm2_t33_650M_UR50D \
    model.max_length=1024
```

## 6. Troubleshooting

- **App crashes on startup**: Ensure the `classifier.path` is correct and the file exists.
- **"Invalid characters" error**: Ensure your sequence contains only standard amino acids (ACDEFGHIKLMNPQRSTVWY). Gap characters (`-`) and stop codons (`*`) are not allowed.
- **Port already in use**: Gradio will automatically find the next available port (e.g., 7861). Check the terminal output.
