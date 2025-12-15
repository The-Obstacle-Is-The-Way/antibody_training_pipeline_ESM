---
license: mit
task_categories:
  - text-classification
tags:
  - biology
  - proteins
  - antibody
  - immunology
  - polyreactivity
  - non-specificity
  - ELISA
  - protein-language-model
  - esm
  - novo-nordisk
pretty_name: Boughter Antibody Polyreactivity Dataset (Novo Nordisk Preprocessing)
size_categories:
  - n<1K
dataset_info:
  features:
    - name: sequence
      dtype: string
    - name: label
      dtype: float64
  splits:
    - name: train
      num_examples: 914
  config_name: default
---

# Boughter Antibody Polyreactivity Dataset (Novo Nordisk Preprocessing)

## Dataset Description

- **Homepage:** [Hugging Science Organization](https://huggingface.co/hugging-science)
- **Repository (this implementation):** [The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM)
- **Upstream:** [ludocomito/antibody_training_pipeline_ESM](https://github.com/ludocomito/antibody_training_pipeline_ESM)
- **Paper (Original Dataset):** [Boughter et al. 2020, eLife](https://doi.org/10.7554/eLife.61393)
- **Paper (Preprocessing Methodology):** [Sakhnini et al. 2025, bioRxiv](https://doi.org/10.1101/2025.04.28.650927)
- **Point of Contact:** [Hugging Science](https://huggingface.co/hugging-science)

### Dataset Summary

This dataset contains **914 antibody heavy chain variable domain (VH) sequences** with binary polyreactivity labels, preprocessed according to the methodology described in **Sakhnini et al. 2025** (Novo Nordisk & University of Cambridge). The dataset was originally published by **Boughter et al. 2020** and contains mouse antibodies with ELISA-based polyreactivity measurements against a panel of 4–7 antigens (commonly described as: DNA, insulin, LPS, flagellin, albumin, cardiolipin, KLH).

**This is the preprocessed version used for training the ESM-1v + Logistic Regression model that predicts antibody non-specificity.**

### Key Features

- **Organism:** Mouse (*Mus musculus*)
- **Molecule Type:** Antibody heavy chain variable domain (VH)
- **Assay:** ELISA polyreactivity panel (4–7 antigens: DNA, insulin, LPS, flagellin, albumin, cardiolipin, KLH)
- **Labels:** Binary classification (0 = specific, 1 = non-specific/polyreactive)
- **Annotation:** ANARCI with IMGT numbering scheme
- **Balance:** Well-balanced (48.5% specific, 51.5% non-specific)

### Supported Tasks and Leaderboards

- **Binary Classification:** Predicting antibody polyreactivity/non-specificity from sequence
- **Benchmark:** Novo Nordisk parity benchmark (71% 10-fold CV accuracy)

### Languages

Protein sequences (amino acid alphabet)

## Dataset Structure

### Data Instances

```json
{
  "sequence": "EVQLVESGGGLVKPGGSLRLSCSASGFTFSSYTMHWVRQAPGKGLEWLSSISSSSAYIYYADSVKGRFTVSRDNAKKSLYLQMDSLRAEDTAIYFCARDGTSLTVAGPLDYWGQGTLVTVSS",
  "label": 0.0
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | string | Antibody VH amino acid sequence (IMGT-annotated) |
| `label` | float | Binary label: 0.0 = specific (0 ELISA flags), 1.0 = non-specific (4+ ELISA flags) |

### Data Splits

| Split | Examples | Label 0 (Specific) | Label 1 (Non-specific) |
|-------|----------|--------------------|-----------------------|
| train | 914 | 443 (48.5%) | 471 (51.5%) |

## Dataset Creation

### Curation Rationale

This dataset was created to enable training of machine learning models for predicting antibody polyreactivity from sequence alone. The preprocessing follows the methodology described in Sakhnini et al. 2025, which demonstrated that ESM-1v embeddings combined with logistic regression can predict antibody non-specificity with ~71% accuracy.

### Source Data

#### Original Data Collection

The original sequences were collected from six mouse antibody subsets by Boughter et al. 2020:
1. **Influenza-reactive** (flu): 379 sequences
2. **HIV NAT** (hiv_nat): 134 sequences
3. **HIV Control** (hiv_cntrl): 50 sequences
4. **HIV PLOS** (hiv_plos): 52 sequences
5. **Gut HIV** (gut_hiv): 75 sequences
6. **Mouse IgA** (mouse_iga): 481 sequences

**Total raw sequences:** 1,171

#### Preprocessing Pipeline (Novo Nordisk Methodology)

The following preprocessing was applied according to Sakhnini et al. 2025:

| Stage | Description | Sequences |
|-------|-------------|-----------|
| 1. DNA Translation | Translate DNA nucleotide sequences to protein | 1,171 → 1,117 (95.4%) |
| 2. ANARCI Annotation | Annotate using ANARCI with IMGT numbering | 1,117 → 1,110 (99.4%) |
| 3. Quality Control | Remove sequences with X in CDRs or empty CDRs | 1,110 → 1,065 (95.9%) |
| 4. Novo Flagging | Keep only 0 flags (specific) and 4+ flags (non-specific) | 1,065 → 914 (85.8%) |

**Critical:** Sequences with 1-3 ELISA flags (mildly polyreactive) were **excluded** from training per Novo Nordisk methodology.

#### ELISA Polyreactivity Panel

Antibodies were tested against a panel of up to 7 biochemically diverse antigens:
- DNA (negatively charged)
- Insulin (negatively charged)
- LPS (lipopolysaccharide - amphipathic)
- Flagellin (large protein)
- Albumin (negatively charged)
- Cardiolipin (amphipathic lipid)
- KLH (keyhole limpet hemocyanin - large, polar)

**Flagging Strategy:**
- **0 flags** → Specific (label=0) - INCLUDE
- **1-3 flags** → Mildly polyreactive - **EXCLUDE**
- **4+ flags** → Non-specific (label=1) - INCLUDE

### Annotations

#### Annotation Process

1. **DNA Translation:** Standard genetic code translation
2. **ANARCI Annotation:** IMGT numbering scheme applied to identify CDR and framework regions
3. **Quality Control:** Based on Boughter's seq_loader.py methodology:
   - Remove sequences with 'X' (ambiguous amino acid) in any CDR
   - Remove sequences with empty CDR regions
4. **Label Assignment:** Binary labels based on ELISA flag count

#### Who are the annotators?

- **Original ELISA assays:** Boughter et al. 2020 (University of Chicago)
- **Preprocessing pipeline:** Based on Sakhnini et al. 2025 (Novo Nordisk & University of Cambridge)
- **This preprocessing:** CLARITY-DIGITAL-TWIN project (reproducing Novo methodology)

### Personal and Sensitive Information

This dataset contains mouse antibody sequences only. No human sequences or personal information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset enables development of computational tools to predict antibody developability, potentially accelerating therapeutic antibody discovery and reducing animal testing by enabling in-silico screening.

### Discussion of Biases

1. **Species Bias:** All sequences are from mouse (*Mus musculus*), which may not generalize to human or humanized antibodies
2. **Assay Bias:** ELISA-based polyreactivity may not capture all forms of non-specificity
3. **Selection Bias:** Sequences with 1-3 flags were excluded, potentially removing informative borderline cases
4. **Panel Bias:** The 7-antigen panel may not represent all potential off-target interactions

### Other Known Limitations

1. **VH Only:** This dataset contains only heavy chain sequences; light chain information is not included in this file
2. **No CDR/Framework Annotations:** The CSV contains full VH sequences without explicit CDR boundaries (use ANARCI for annotation)
3. **Small Size:** 914 sequences is relatively small for deep learning applications

## Additional Information

### Dataset Curators

- **Original Dataset:** Christopher T. Boughter, Erin J. Adams (University of Chicago)
- **Preprocessing Methodology:** Laila I. Sakhnini, Daniele Granata et al. (Novo Nordisk)
- **This Preprocessing:** CLARITY-DIGITAL-TWIN project (Hugging Science)

### Licensing Information

Boughter et al. (2020) is published under **CC-BY-4.0** (per the DOI landing page). The raw source files in this repository were copied from `ctboughter/AIMS_manuscripts` (repository license: MIT). This Hugging Face export is distributed under the **MIT license**; please retain upstream attribution/citations (paper + repository).

### Citation Information

**If you use this dataset, please cite the original paper, the Novo Nordisk methodology paper, and ANARCI (used for IMGT numbering):**

```bibtex
@article{boughter2020biochemical,
  title={Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops},
  author={Boughter, Christopher T and Borowska, Marta T and Guthmiller, Jenna J and Bendelac, Albert and Wilson, Patrick C and Roux, Beno{\^\i}t and Adams, Erin J},
  journal={eLife},
  volume={9},
  pages={e61393},
  year={2020},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.61393}
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

Thanks to the Boughter lab for making the original data publicly available, and to Novo Nordisk for publishing their preprocessing methodology, enabling independent replication.

---

**Version:** 1.0.0
**Last Updated:** 2025-12-14
**Maintainer:** Hugging Science Organization
