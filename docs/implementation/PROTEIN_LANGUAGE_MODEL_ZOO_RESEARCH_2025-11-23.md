# Protein Language Model Zoo Research: The "Gold Standard" Implementation Plan (November 2025)

**Date**: 2025-11-23
**Status**: **RESTORED & ENHANCED** (Combines Deep Research Findings + Detailed Implementation Plan)
**Purpose**: Define the definitive strategy to outperform ESM-1v for antibody non-specificity prediction using a scientifically rigorous "Model Zoo" of backbones and simple classifier heads.

---

## 1. Executive Summary: The "Alpha" Findings

To beat Novo Nordisk's **71% accuracy** (Jain dataset), we cannot simply swap one general PLM for another. We must leverage distinct **inductive biases** that target specific limitations of the baseline.

### 1.1 The "Banger" Backbones (November 2025)
| Model | Inductive Bias | Why it could win | Status |
|-------|----------------|------------------|--------|
| **ESM-1v** | **Evolution** | **Baseline**. Captures fitness via masked evolutionary modeling. | *Current* |
| **IgBERT** | **Scale** | **2B+ Antibody Sequences**. Massive domain-specific coverage (OAS) vs ESM's general protein data. | *Tier 1* |
| **AMPLIFY** | **Data Quality** | **Curated Quality**. "Is Scaling Necessary?" paper shows 350M params can beat larger models if data is clean. | *Tier 1* |
| **SaProt** | **Structure** | **Surface Aware**. Encodes 3D structure (Foldseek tokens). Specificity is a surface property; this is the orthogonal "breaker". | *Tier 1* |
| **ESM-2 650M**| **Modern Arch**| **Efficiency**. Modern transformer architecture; potential "sweet spot" for transfer learning. | *Tier 1* |

### 1.2 The "Banger" Heads
Novo Nordisk tested LogisticReg, RandomForest, GaussianProcess, GradientBoosting, and **SVM** (Methods Section 4.3) but only published Logistic Regression results. We will test what they didn't publish:
*   **Linear SVM**: **Hypothesis** - Max-margin optimization may generalize better than probability fitting (LogReg) on limited data (914 samples) in high-dimensional embedding space (1280d+). This is an **empirical question** we will answer.
*   **Ridge Classifier**: L2 regularization explicitly handles multicollinearity in correlated PLM embeddings, potentially improving robustness over standard LogReg.

---

## 2. Detailed Model Analysis

### 2.1 IgBERT (2024) - The "Big Data" Bet
*   **Source**: [PLOS Comp Biol 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012646)
*   **Specs**: ~650M params, trained on 2.1 Billion OAS sequences.
*   **Hypothesis**: Captures rare CDR patterns and antibody-specific liabilities that general PLMs miss.
*   **Risk Alert (Nov 2025)**: A March 2025 preprint ("A curriculum learning approach...") noted IgBERT/IgT5 might perform *poorly* on specificity tasks despite their size. This makes benchmarking it against ESM-1v critical to verify if "scale is enough."
*   **Integration**: Initialized from ProtBERT; compatible with Hugging Face `transformers`.

### 2.2 AMPLIFY (350M) - The "Efficiency" Bet
*   **Source**: [MarkTechPost / Nature 2024](https://www.marktechpost.com/2024/09/30/amplify-leveraging-data-quality-over-scale-for-efficient-protein-language-model-development/)
*   **Specs**: 350M params, trained on curated UR50 + OAS.
*   **Hypothesis**: Matches or beats ESM-1v performance at ~50% inference cost due to superior data curation.

### 2.3 SaProt (650M) - The "Structure" Bet
*   **Source**: [ICLR 2024 Spotlight](https://openreview.net/forum?id=6MRm3G4NiU)
*   **Specs**: 650M params, input = Amino Acids + Foldseek 3Di tokens.
*   **Requirement**: Requires generating structure tokens.
    *   *Pipeline*: Sequence -> ESMFold (fast) -> Foldseek -> SaProt Tokens -> Embeddings.
*   **Hypothesis**: **The strongest theoretical candidate.** Recent research (Oct 2024) confirms polyreactivity is driven by **heavy-chain CDR surface charge and hydrophobicity**. Sequence models infer this indirectly; SaProt's structure tokens explicitly encode these surface geometries.
*   **Verdict**: The highest "high risk, high reward" candidate.

### 2.4 ESM-2 (650M)
*   **Source**: Meta AI / Science 2022.
*   **Specs**: 650M params (Layer 33).
*   **Hypothesis**: Standard modern baseline. If it fails, it confirms Novo's finding that "Evolutionary training (ESM-1v) > General Sequence training (ESM-2)".

---

## 3. Technical Implementation Plan

We will refactor `src/antibody_training_esm` to support a plug-and-play **Model Zoo**.

### 3.1 Directory Structure
```
src/antibody_training_esm/
├── models/
│   ├── __init__.py
│   ├── base.py           # AbstractBaseClass for PLMs
│   ├── esm1v.py          # Existing, refactored
│   ├── igbert.py         # New
│   ├── amplify.py        # New
│   ├── saprot.py         # New (Includes Foldseek preprocessing hooks)
│   ├── esm2.py           # New
│   └── registry.py       # Factory: get_model("igbert", device="cuda")
└── heads/
    ├── __init__.py
    ├── logistic.py       # Wraps sklearn LogisticRegression
    ├── svm.py            # Wraps sklearn LinearSVC
    └── ridge.py          # Wraps sklearn RidgeClassifier
```

### 3.2 Unified Interface (`models/base.py`)
```python
from abc import ABC, abstractmethod
import torch
import numpy as np

class ProteinLanguageModel(ABC):
    @abstractmethod
    def load_model(self, device: str):
        """Load model weights to device."""
        pass

    @abstractmethod
    def embed(self, sequences: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate frozen embeddings.
        Args:
            sequences: List of amino acid sequences.
            batch_size: Inference batch size.
        Returns:
            np.ndarray: Shape (n_samples, hidden_dim)
        """
        pass
    
    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Return embedding dimension."""
        pass
        
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return unique identifier."""
        pass
```

### 3.3 SaProt Special Handling (`models/saprot.py`)
SaProt requires a structure tokenization step.
```python
class SaProtModel(ProteinLanguageModel):
    def embed(self, sequences, batch_size=32):
        # 1. Fold: Run ESMFold (or lightweight folder) if structures missing
        # 2. Tokenize: Run Foldseek to get 3Di tokens
        # 3. Embed: Pass (Seq + 3Di) to SaProt
        pass
```
*Note: For the MVP, we may simply use amino-acid only mode if 3Di generation is too heavy, but the "structure bet" requires the tokens.*

---

## 4. Benchmarking Framework

We will create `scripts/benchmark_zoo.py` to rigorously compare all combinations.

### 4.1 The Loop
1.  **Data**: Load Boughter (Train) and Jain (Test).
2.  **Outer Loop (Backbones)**: `[ESM-1v, IgBERT, AMPLIFY, SaProt, ESM-2]`
    *   Generate/Cache embeddings for all sequences.
3.  **Inner Loop (Heads)**: `[LogReg, SVM, Ridge]`
    *   Train on Boughter embeddings.
    *   Predict on Jain embeddings.
    *   Compute Metrics: Accuracy, AUC, Precision, Recall, MCC.
4.  **Output**: `results/benchmark_zoo_20251123.csv`

### 4.2 Success Matrix
We are looking for a model that exceeds the **Parity Threshold**.

| Backbone | Head | Jain Accuracy (Target > 71%) | AUC (Target > 0.80) | Note |
|----------|------|------------------------------|---------------------|------|
| ESM-1v | LogReg | **71.0%** | ~0.79 | **Baseline** |
| ESM-1v | SVM | ? | ? | Quick Win? |
| IgBERT | LogReg | ? | ? | Scale Win? |
| AMPLIFY | LogReg | ? | ? | Efficiency Win? |
| SaProt | SVM | ? | ? | **Singularity Win?** |

---

## 5. Configuration & Dependencies

### 5.1 `pyproject.toml` Additions
We need to ensure we have the libraries for the new models.
*   `transformers>=4.39.0` (For recent models)
*   `flash-attn` (For efficient inference, optional but recommended)
*   `foldseek` (Binary dependency for SaProt, typically external or via conda)

### 5.2 Hydra Config (`conf/config.yaml`)
```yaml
defaults:
  - model: esm1v  # Default
  - head: logreg  # Default

model:
  name: esm1v # Choices: igbert, amplify, saprot, esm2
  
head:
  type: logreg # Choices: svm, ridge
  params:
    C: 1.0
```

---

## 6. References

1.  **ESM-1v (Novo Baseline)**: Sakhnini et al., "Prediction of Antibody Non-Specificity using Protein Language Models...", bioRxiv 2025.
2.  **IgBERT**: "Large scale paired antibody language models", PLOS Comp Biol 2024.
3.  **AMPLIFY**: "AMPLIFY: Leveraging Data Quality...", Nature/MarkTechPost 2024.
4.  **SaProt**: "SaProt: Protein Language Modeling with Structure-Aware Vocabulary", ICLR 2024.
5.  **ESM-2**: "Evolutionary-scale prediction of atomic-level protein structure", Science 2022.
6.  **Benchmarking**: "Ab-VS: Evaluating LLMs for Virtual Antibody Screening", bioRxiv 2025.

---

## 7. Next Steps (Actionable)

1.  **Refactor**: Create `src/antibody_training_esm/models/` and move `esm1v` logic there.
2.  **Implement**: Add `IgBERT` and `AMPLIFY` wrappers first (easiest integration).
3.  **Benchmark**: Run `ESM-1v` vs `IgBERT` vs `AMPLIFY` on Jain dataset.
4.  **Implement SaProt**: Add the structural folding pipeline (Phase 2).
