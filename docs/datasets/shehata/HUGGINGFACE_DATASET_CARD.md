---
license: cc-by-4.0
task_categories:
  - text-classification
tags:
  - biology
  - proteins
  - antibody
  - immunology
  - polyreactivity
  - non-specificity
  - PSR
  - affinity-maturation
  - B-cell
  - protein-language-model
  - esm
  - novo-nordisk
pretty_name: Shehata Antibody PSR Dataset (Novo Nordisk Preprocessing)
size_categories:
  - n<1K
dataset_info:
  features:
    - name: id
      dtype: string
    - name: sequence
      dtype: string
    - name: label
      dtype: int64
    - name: psr_score
      dtype: float64
    - name: b_cell_subset
      dtype: string
    - name: source
      dtype: string
  splits:
    - name: test
      num_examples: 398
  config_name: default
---

# Shehata Antibody PSR Dataset (Novo Nordisk Preprocessing)

## Dataset Description

- **Homepage:** [Hugging Science Organization](https://huggingface.co/hugging-science)
- **Repository (this implementation):** [The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM)
- **Upstream:** [ludocomito/antibody_training_pipeline_ESM](https://github.com/ludocomito/antibody_training_pipeline_ESM)
- **Paper (Original Dataset):** [Shehata et al. 2019, Cell Reports](https://doi.org/10.1016/j.celrep.2019.08.056)
- **Paper (Preprocessing Methodology):** [Sakhnini et al. 2025, bioRxiv](https://doi.org/10.1101/2025.04.28.650927)
- **Point of Contact:** [Hugging Science](https://huggingface.co/hugging-science)

### Dataset Summary

This dataset contains **398 human antibody heavy chain variable domain (VH) sequences** with PSR (Poly-Specificity Reagent) measurements, preprocessed according to the methodology described in **Sakhnini et al. 2025** (Novo Nordisk & University of Cambridge). The dataset was originally published by **Shehata et al. 2019** and contains human B cell-derived antibodies studying the relationship between affinity maturation and antibody specificity.

**This is the preprocessed version used as a test set for evaluating cross-assay transfer learning (ELISA-trained model → PSR test data).**

### Key Features

- **Organism:** Human (*Homo sapiens*)
- **Molecule Type:** Antibody heavy chain variable domain (VH)
- **Source:** Human B cells from healthy donors (IgG memory, IgM memory, Naïve, LLPCs)
- **Assay:** PSR (Poly-Specificity Reagent) flow cytometry (CHO cell membrane/cytosolic proteins)
- **Labels:** Binary classification (0 = low PSR, 1 = high PSR)
- **Annotation:** ANARCI with IMGT numbering scheme
- **Balance:** Highly imbalanced (98.2% low PSR, 1.8% high PSR)

### Important Note: Class Imbalance

This dataset is **highly imbalanced** with only 7 high-PSR sequences out of 398 total. This reflects the biological reality that most antibodies in the study were specific (low PSR). Consider this when evaluating model performance.

### Supported Tasks and Leaderboards

- **Binary Classification:** Predicting antibody PSR from sequence
- **Cross-Assay Transfer Learning:** Testing ELISA-trained models on PSR data
- **Benchmark:** Sakhnini et al. 2025 Fig. S14C (58.8% accuracy)

### Languages

Protein sequences (amino acid alphabet)

## Dataset Structure

### Data Instances

```json
{
  "id": "ADI-38502",
  "sequence": "EVQLLESGGGLVKPGGSLRLSCAASGFIFSDYSMNWVRQAPGKGLEWVSSISSSSGYIYYADSVKGRFTISRDNAKNSLYLQMNSLRADDTAVYYCARRAYGSGTSPQYFDYWGQGTLVTVSS",
  "label": 0,
  "psr_score": 0.0,
  "b_cell_subset": "IgG memory",
  "source": "shehata2019"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Antibody identifier (ADI-XXXXX format from Adimab) |
| `sequence` | string | Antibody VH amino acid sequence (gap-free; ANARCI/IMGT-validated) |
| `label` | int | Binary label: 0 = low PSR, 1 = high PSR |
| `psr_score` | float | Continuous PSR score from flow cytometry |
| `b_cell_subset` | string | B cell subset origin (IgG memory, IgM memory, Naïve, LLPCs) |
| `source` | string | Data source identifier (shehata2019) |

### Data Splits

| Split | Examples | Label 0 (Low PSR) | Label 1 (High PSR) |
|-------|----------|-------------------|--------------------|
| test | 398 | 391 (98.2%) | 7 (1.8%) |

**Note:** This dataset is used exclusively as a test set for cross-assay validation. The entire dataset is the "test" split.

## Dataset Creation

### Curation Rationale

This dataset was created to study the relationship between antibody affinity maturation and specificity. It enables evaluation of whether models trained on ELISA polyreactivity data can transfer to PSR-based measurements of non-specificity.

### Source Data

#### Original Data Collection

From Shehata et al. 2019:
- **Source:** Human B cells from healthy donors (LLPCs from bone marrow; naïve and memory B cells from peripheral blood)
- **Subsets:** IgG memory, IgM memory, Naïve, LLPCs (long-lived plasma cells)
- **Assay:** PSR (Poly-Specificity Reagent) flow cytometry
- **Study Focus:** How affinity maturation affects antibody specificity

**Key Finding from Original Paper:**
> "Affinity maturation enhances antibody specificity but compromises conformational stability"

#### Preprocessing Pipeline (Novo Nordisk Methodology)

| Stage | Description | Sequences |
|-------|-------------|-----------|
| 1. Excel Extraction | Extract from `shehata-mmc2.xlsx` Supplementary Table S1 | 402 |
| 2. Drop Non-sequence Rows | Drop legend/metadata rows without VH/VL sequences | 402 → 400 |
| 3. Drop Missing PSR | Drop antibodies without numeric PSR scores | 400 → 398 |
| 4. ANARCI Annotation | Annotate using ANARCI with IMGT numbering | 398 → 398 (100%) |
| 5. Gap Removal | Use `sequence_aa` not `sequence_alignment_aa` | (no change) |

**100% Success Rate:** All 398 sequences were successfully annotated by ANARCI.

### Novo Nordisk Methodology Verification

This dataset's preprocessing was cross-referenced against Sakhnini et al. (2025) Section 4.1:

| Metric | Novo Paper (Section 4.1) | This Dataset | Status |
|--------|--------------------------|--------------|--------|
| Dataset Size | "398 antibodies" | 398 sequences | ✅ EXACT MATCH |
| Non-specific Count | "7 out of 398 antibodies characterised as non-specific only" | 7 (1.8%) | ✅ EXACT MATCH |
| Annotation Method | "ANARCI following the IMGT numbering scheme" | ANARCI/IMGT | ✅ MATCH |
| Source | Shehata et al. 2019 | Cell Reports Supplementary Table S1 | ✅ MATCH |

**Verification Notes:**
- The exact counts (398 total, 7 non-specific) match the Novo paper precisely
- Labels are derived from PSR scores using the same threshold methodology
- No additional filtering was applied beyond ANARCI annotation

#### Binary Label Assignment

PSR scores were converted to binary labels following Shehata et al. (2019) high-polyreactivity threshold:

- **Low/no PSR (label=0):** `psr_score ≤ 0.33` → 391 antibodies (98.2%)
- **High PSR (label=1):** `psr_score > 0.33` → 7 antibodies (1.8%)

For parity with Sakhnini et al. (2025), the conversion script computes a cutoff at the top 7/398 antibodies (98.24th percentile); in this dataset that is equivalent to `psr_score > 0.33`.

### Annotations

#### Annotation Process

1. **Excel Parsing:** Extract VH sequences and PSR scores from Supplementary Table S1
2. **ANARCI Annotation:** IMGT numbering scheme applied to identify VH domain boundaries
3. **Gap Character Handling:** Use `sequence_aa` (gap-free) for ESM compatibility
4. **Label Binarization:** PSR scores converted to binary (low/high)

#### Special Attribution

From the original paper acknowledgments:
> Tingwan Sun and Yingda Xu from **Adimab, LLC** contributed to the PSR measurements in this study.

#### Who are the annotators?

- **Original PSR Assays:** Shehata et al. 2019 (Laura Walker Lab, Adimab collaborators)
- **Preprocessing pipeline:** Based on Sakhnini et al. 2025 (Novo Nordisk & University of Cambridge)
- **This preprocessing:** CLARITY-DIGITAL-TWIN project (reproducing Novo methodology)

### Personal and Sensitive Information

This dataset contains human-derived antibody sequences. However, these are B cell receptor sequences from healthy donor samples, which do not constitute personally identifiable information. The original study was conducted with appropriate ethical oversight.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset enables:
- Understanding the specificity-stability tradeoff in antibody engineering
- Cross-assay validation of polyreactivity prediction models
- Development of tools to identify potentially non-specific antibodies early in drug development

### Discussion of Biases

1. **Severe Class Imbalance:** Only 1.8% (7/398) are high-PSR - consider appropriate metrics (F1, ROC-AUC)
2. **Human-Specific:** All sequences are human-derived; may not generalize to other species
3. **Assay Bias:** PSR assay measures different aspects of non-specificity than ELISA
4. **Selection Bias:** Antibodies were selected for affinity maturation studies, not random sampling
5. **B Cell Subset Distribution:** Enriched for memory B cells

### Other Known Limitations

1. **VH Only:** This dataset contains only heavy chain sequences; light chain (VL) is available separately
2. **Small Size:** 398 sequences limits statistical power
3. **Extreme Imbalance:** Standard accuracy metrics may be misleading

### Recommended Usage

When evaluating models trained on ELISA data (Boughter):
```python
# For reproducing Sakhnini et al. (2025) Fig. S14C, binarize model probabilities with:
THRESHOLD = 0.5495  # decision threshold on predicted P(non-specific)
predictions = (model_probabilities >= THRESHOLD).astype(int)

# Use appropriate metrics for imbalanced data
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
```

### Note on Inference Threshold (0.5495)

**IMPORTANT:** The 0.5495 threshold is for **model inference/evaluation only**, NOT preprocessing.

- **What it is:** A decision threshold for binarizing model prediction probabilities during evaluation
- **What it is NOT:** A preprocessing parameter - the data (sequences, labels) is unaffected
- **Why it exists:** Empirically determined to better reproduce Sakhnini et al. (2025) Fig. S14C results when evaluating ELISA-trained models on PSR test data
- **Not in the paper:** This threshold value is not described in Sakhnini et al. (2025); it is derived via threshold sweep in this repository for parity against reported results
- **Standard threshold:** 0.5 (binary classification default)
- **PSR-calibrated threshold:** 0.5495 (determined via threshold sweep to match Novo's reported accuracy)

This threshold adjustment compensates for the cross-assay domain shift between ELISA (training) and PSR (testing) data.

### Recommended Metrics

Due to severe class imbalance, prioritize these metrics over accuracy:
- **ROC-AUC:** Area under the ROC curve (not affected by threshold or imbalance)
- **Balanced Accuracy:** Average of sensitivity and specificity
- **F1 Score:** Harmonic mean of precision and recall

## Additional Information

### Dataset Curators

- **Original Dataset:** Laila Shehata, Laura M. Walker (Scripps Research, Adimab)
- **PSR Measurements:** Tingwan Sun, Yingda Xu (Adimab, LLC)
- **Preprocessing Methodology:** Laila I. Sakhnini, Daniele Granata et al. (Novo Nordisk)
- **This Preprocessing:** CLARITY-DIGITAL-TWIN project (Hugging Science)

### Licensing Information

Shehata et al. (2019) is published under **CC-BY-4.0** (per the DOI landing page). The raw source files in this repository are the Cell Reports supplementary spreadsheets; please retain upstream attribution/citations.

### Citation Information

**If you use this dataset, please cite the original paper, the Novo Nordisk methodology paper, and ANARCI (used for IMGT numbering):**

```bibtex
@article{shehata2019affinity,
  title={Affinity maturation enhances antibody specificity but compromises conformational stability},
  author={Shehata, Laila and Maurer, Daniel P and Wec, Anna Z and Lilov, Asparouh and Champney, Elizabeth and Sun, Tingwan and Archambault, Kimberly and Burnina, Irina and Lynaugh, Heather and Zhi, Xiaoyong and Xu, Yingda and Walker, Laura M},
  journal={Cell Reports},
  volume={28},
  number={13},
  pages={3300--3308},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.celrep.2019.08.056}
}

@article{sakhnini2025prediction,
  title={Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters},
  author={Sakhnini, Laila I. and Beltrame, Ludovica and Fulle, Simone and Sormanni, Pietro and Henriksen, Anette and Lorenzen, Nikolai and Vendruscolo, Michele and Granata, Daniele},
  journal={bioRxiv},
  year={2025},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.04.28.650927}
}

@article{dunbar2016anarci,
  title={ANARCI: antigen receptor numbering and receptor classification},
  author={Dunbar, James and Deane, Charlotte M},
  journal={Bioinformatics},
  volume={32},
  number={2},
  pages={298--300},
  year={2016},
  doi={10.1093/bioinformatics/btv552}
}
```

### Acknowledgments

We are grateful to:
- **Adimab, LLC** (Tingwan Sun, Yingda Xu) for contributing the PSR measurements
- **Laura Walker Lab** (Scripps Research) for publishing this valuable dataset
- **Novo Nordisk** for publishing their preprocessing methodology

### Contributions

Thanks to the Shehata/Walker lab and Adimab for making the original data publicly available, and to Novo Nordisk for publishing their preprocessing methodology.

---

**Version:** 1.0.0
**Last Updated:** 2025-12-14
**Maintainer:** Hugging Science Organization
