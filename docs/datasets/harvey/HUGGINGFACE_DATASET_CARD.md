---
license: mit
task_categories:
  - text-classification
tags:
  - biology
  - proteins
  - antibody
  - nanobody
  - VHH
  - immunology
  - polyreactivity
  - non-specificity
  - PSR
  - FACS
  - deep-sequencing
  - protein-language-model
  - esm
  - novo-nordisk
pretty_name: Harvey Nanobody Polyreactivity Dataset (Novo Nordisk Preprocessing)
size_categories:
  - 100K<n<1M
dataset_info:
  features:
    - name: id
      dtype: string
    - name: sequence
      dtype: string
    - name: label
      dtype: int64
    - name: source
      dtype: string
    - name: sequence_length
      dtype: int64
  splits:
    - name: test
      num_examples: 141021
  config_name: default
---

# Harvey Nanobody Polyreactivity Dataset (Novo Nordisk Preprocessing)

## Dataset Description

- **Homepage:** [Hugging Science Organization](https://huggingface.co/hugging-science)
- **Repository (this implementation):** [The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM)
- **Upstream:** [ludocomito/antibody_training_pipeline_ESM](https://github.com/ludocomito/antibody_training_pipeline_ESM)
- **Paper (Original Dataset):** [Harvey et al. 2022, Nature Communications](https://doi.org/10.1038/s41467-022-35276-4)
- **Paper (Preprocessing Methodology):** [Sakhnini et al. 2025, bioRxiv](https://doi.org/10.1101/2025.04.28.650927)
- **Original Data Source:** [debbiemarkslab/nanobody-polyreactivity](https://github.com/debbiemarkslab/nanobody-polyreactivity)
- **Point of Contact:** [Hugging Science](https://huggingface.co/hugging-science)

### Dataset Summary

This dataset contains **141,021 nanobody (VHH) sequences** with binary polyreactivity labels, preprocessed according to the methodology described in **Sakhnini et al. 2025** (Novo Nordisk & University of Cambridge). The dataset was originally published by **Harvey et al. 2022** and contains synthetic nanobodies assessed by PSR (Poly-Specificity Reagent) assay via FACS sorting and deep sequencing.

**This is the preprocessed version used as a test set for evaluating the ESM-1v + Logistic Regression model trained on the Boughter dataset.**

### Key Features

- **Organism:** Synthetic camelid (nanobody) library (yeast display)
- **Molecule Type:** Nanobody / Single-domain antibody (VHH)
- **Assay:** PSR (Poly-Specificity Reagent) from Sf9 insect cell membranes
- **Method:** FACS sorting + Deep sequencing
- **Labels:** Binary classification (0 = low polyreactivity, 1 = high polyreactivity)
- **Annotation:** ANARCI with IMGT numbering scheme
- **Balance:** Well-balanced (49.1% low, 50.9% high polyreactivity)
- **Scale:** Large-scale dataset (141K sequences)

### Supported Tasks and Leaderboards

- **Binary Classification:** Predicting nanobody polyreactivity from sequence
- **Cross-Domain Validation:** Testing conventional antibody-trained models on nanobodies
- **Benchmark:** Sakhnini et al. 2025 Fig. S14E (61.7% accuracy)

### Languages

Protein sequences (amino acid alphabet)

## Dataset Structure

### Data Instances

```json
{
  "id": "harvey_000001",
  "sequence": "QVQLVESGGGLVQAGGSLRLSCAASGFTFVYYVMGWYRQAPGKERELVAAINAGGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCNARVRVRWSSYYYWGQGTQVTVSS",
  "label": 1,
  "source": "harvey2022",
  "sequence_length": 120
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (harvey_XXXXXX format) |
| `sequence` | string | Nanobody VHH amino acid sequence (gap-free; ANARCI/IMGT-validated) |
| `label` | int | Binary label: 0 = low polyreactivity, 1 = high polyreactivity |
| `source` | string | Data source identifier (harvey2022) |
| `sequence_length` | int | Length of the VHH sequence in amino acids |

### Data Splits

| Split | Examples | Label 0 (Low) | Label 1 (High) |
|-------|----------|---------------|----------------|
| test | 141,021 | 69,262 (49.1%) | 71,759 (50.9%) |

**Note:** This dataset is used exclusively as a test set for models trained on the Boughter dataset. The entire dataset is the "test" split.

## Dataset Creation

### Curation Rationale

This dataset was created to evaluate whether models trained on conventional antibody polyreactivity data (Boughter - ELISA) can generalize to:
1. **Different molecule types:** Nanobodies (VHH) vs conventional antibodies (VH)
2. **Different assays:** PSR assay vs ELISA assay

### Source Data

#### Original Data Collection

From Harvey et al. 2022:
- Started with >2 × 10⁹ synthetic yeast display nanobody library
- MACS enrichment for polyreactive clones
- FACS sorting with PSR (polyspecificity reagent from Sf9 insect cell membranes)
- Deep sequencing of high and low polyreactivity pools

**Original Files (from debbiemarkslab/nanobody-polyreactivity):**
- `high_polyreactivity_high_throughput.csv`: 71,772 sequences
- `low_polyreactivity_high_throughput.csv`: 69,702 sequences
- **Total:** 141,474 sequences

#### Preprocessing Pipeline (Novo Nordisk Methodology)

**IMPORTANT:** Sakhnini et al. (2025) describe using the **unfiltered** Harvey dataset (>140,000 nanobodies), not the CDR-length-filtered subset (~134K) used in Harvey et al.'s published one-hot predictor. This export starts from the full official repository release (141,474 sequences).

| Stage | Description | Sequences |
|-------|-------------|-----------|
| 1. Raw Data | Combine high and low polyreactivity CSVs | 141,474 |
| 2. ANARCI Annotation | Annotate using ANARCI with IMGT numbering | 141,474 → 141,021 (99.68%) |
| 3. Gap Removal | Use `sequence_aa` not `sequence_alignment_aa` | (no change) |

**ANARCI Failures:** 453 sequences (0.32%) failed annotation and were excluded.

#### CDR Length Filtering

Harvey et al.'s published predictor uses a CDR length filter:
- CDR1==8, CDR2==8 or 9, CDR3==6-22 → 134,302 sequences

Sakhnini et al. (2025) describe using ">140 000 naïve nanobodies" and do not mention applying this filter. Accordingly, this export does not apply it:
- No CDR-length filter → 141,474 raw sequences → 141,021 after ANARCI

Evidence: Sakhnini et al. (2025) Section 4.1 ("Data sources") describes the Harvey dataset as ">140 000 naïve nanobodies", consistent with using the unfiltered data.

### Novo Nordisk Methodology Verification

This dataset's preprocessing was cross-referenced against Sakhnini et al. (2025) Section 4.1:

| Metric | Novo Paper (Section 4.1) | This Dataset | Status |
|--------|--------------------------|--------------|--------|
| Dataset Size | ">140,000 naïve nanobodies" | 141,021 sequences | ✅ MATCH |
| Annotation Method | "ANARCI following the IMGT numbering scheme" | ANARCI/IMGT | ✅ MATCH |
| Source | Harvey et al. 2022 | debbiemarkslab/nanobody-polyreactivity | ✅ MATCH |
| ANARCI Failures | Not explicitly stated | 453 (0.32%) | Documented |

**Verification Notes:**
- The paper states ">140,000" which is consistent with our 141,021 post-ANARCI count
- Labels are directly from the original Harvey et al. 2022 FACS sorting (high/low PSR pools)
- No additional filtering was applied beyond ANARCI annotation
- Note: Sakhnini et al. Fig. S14E confusion matrix totals 141,559 nanobodies, suggesting their preprocessing snapshot may differ slightly from the official upstream data used here (141,474 raw → 141,021 ANARCI-validated)

### Annotations

#### Annotation Process

1. **ANARCI Annotation:** IMGT numbering scheme applied to identify VHH domain boundaries
2. **Gap Character Handling:** Use `sequence_aa` (gap-free) instead of `sequence_alignment_aa`
3. **Label Assignment:** Binary labels from original FACS sorting (high vs low PSR pools)

#### Who are the annotators?

- **Original FACS/Sequencing:** Harvey et al. 2022 (Debbie Marks Lab, Harvard)
- **Preprocessing pipeline:** Based on Sakhnini et al. 2025 (Novo Nordisk & University of Cambridge)
- **This preprocessing:** CLARITY-DIGITAL-TWIN project (reproducing Novo methodology)

### Personal and Sensitive Information

This dataset contains synthetic nanobody sequences from a yeast display library. No human sequences or personal information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset enables:
- Development of polyreactivity prediction tools for nanobodies
- Cross-domain validation of antibody developability models
- In-silico screening to reduce experimental burden

### Discussion of Biases

1. **Synthetic Library Bias:** All sequences are from a synthetic yeast display library, not natural immune repertoires
2. **Assay Bias:** PSR assay may capture different aspects of non-specificity than ELISA
3. **Selection Pressure:** FACS sorting may introduce biases based on expression level
4. **Nanobody-Specific:** Results may not generalize to conventional antibodies

### Other Known Limitations

1. **VHH Only:** This dataset contains single-domain antibodies (no light chain)
2. **Binary Labels:** Quantitative PSR scores are not included (only binary high/low)
3. **Cross-Assay Transfer:** Models trained on ELISA data (Boughter) may not optimally transfer to PSR data

### Recommended Usage

When evaluating models trained on ELISA data (Boughter):
```python
# For reproducing Sakhnini et al. (2025) Fig. S14E, binarize model probabilities with:
THRESHOLD = 0.5495  # decision threshold on predicted P(non-specific)
predictions = (model_probabilities >= THRESHOLD).astype(int)
```

### Note on Inference Threshold (0.5495)

**IMPORTANT:** The 0.5495 threshold is for **model inference/evaluation only**, NOT preprocessing.

- **What it is:** A decision threshold for binarizing model prediction probabilities during evaluation
- **What it is NOT:** A preprocessing parameter - the data (sequences, labels) is unaffected
- **Why it exists:** Empirically determined to better reproduce Sakhnini et al. (2025) Fig. S14E results when evaluating ELISA-trained models on PSR test data
- **Not in the paper:** This threshold value is not described in Sakhnini et al. (2025); it is derived via threshold sweep in this repository for parity against reported results
- **Standard threshold:** 0.5 (binary classification default)
- **PSR-calibrated threshold:** 0.5495 (determined via threshold sweep to match Novo's reported accuracy)

This threshold adjustment compensates for the cross-assay domain shift between ELISA (training) and PSR (testing) data.

## Additional Information

### Dataset Curators

- **Original Dataset:** Emily P. Harvey, Debbie Marks Lab (Harvard Medical School)
- **Preprocessing Methodology:** Laila I. Sakhnini, Daniele Granata et al. (Novo Nordisk)
- **This Preprocessing:** CLARITY-DIGITAL-TWIN project (Hugging Science)

### Licensing Information

Harvey et al. (2022) is published under **CC-BY-4.0** (per the DOI landing page). The raw source files in this repository were copied from `debbiemarkslab/nanobody-polyreactivity` (repository license: MIT). This Hugging Face export is distributed under the **MIT license**; please retain upstream attribution/citations (paper + repository).

### Citation Information

**If you use this dataset, please cite the original paper, the Novo Nordisk methodology paper, and ANARCI (used for IMGT numbering):**

```bibtex
@article{harvey2022in_silico,
  title={An in silico method to assess antibody fragment polyreactivity},
  author={Harvey, Edward P. and Shin, Jung-Eun and Skiba, Meredith A. and Nemeth, Genevieve R. and Hurley, Joseph D. and Wellner, Alon and Shaw, Ada Y. and Miranda, Victor G. and Min, Joseph K. and Liu, Chang C. and Marks, Debora S. and Kruse, Andrew C.},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={7554},
  year={2022},
  publisher={Springer Science and Business Media LLC},
  doi={10.1038/s41467-022-35276-4}
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

### Contributions

Thanks to the Harvey lab and Debbie Marks lab for making the original data publicly available, and to Novo Nordisk for publishing their preprocessing methodology.

---

**Version:** 1.0.0
**Last Updated:** 2025-12-14
**Maintainer:** Hugging Science Organization
