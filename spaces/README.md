---
title: Antibody Non-Specificity Predictor
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
tags:
  - antibody
  - protein
  - ESM
  - gradio
  - polyreactivity
  - machine-learning
---

# 🧬 Antibody Non-Specificity Predictor

Predict antibody polyreactivity (non-specificity) from Variable Heavy (VH) or Variable Light (VL) sequences using ESM-1v protein language models.

## Model

- **Architecture:** ESM-1v (650M parameters) + Logistic Regression
- **Training Data:** Boughter dataset (914 antibodies, ELISA polyreactivity)
- **Methodology:** Sakhnini et al. (2025) - Prediction of Antibody Non-Specificity using PLMs

## Usage

1. Paste your antibody VH or VL amino acid sequence
2. Click "🔬 Predict Non-Specificity"
3. Get prediction (specific vs non-specific) + probability

## Supported Input

- **Valid characters:** Standard amino acids (ACDEFGHIKLMNPQRSTVWY)
- **Max length:** 2000 amino acids
- **Auto-cleaning:** Lowercase automatically converted to uppercase

## Examples

The app includes example sequences:
- Standard VH (128aa)
- Standard VL (107aa)
- Short VH (Herceptin-like)

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{sakhnini2025antibody,
  title={Prediction of Antibody Non-Specificity using Protein Language Models},
  author={Sakhnini, et al.},
  year={2025}
}
```

## Repository

Full source code: [antibody_training_pipeline_ESM](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM)

## License

MIT License - See repository for details
