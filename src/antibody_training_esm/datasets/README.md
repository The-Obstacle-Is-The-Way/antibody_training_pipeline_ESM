# Dataset Loaders

**IMPORTANT**: These classes are for **LOADING** preprocessed data, NOT for running preprocessing pipelines.

## Architecture Overview

This codebase maintains a clear separation between:

1. **Preprocessing Scripts** (`preprocessing/`) - Creates fragment files
2. **Dataset Loaders** (`src/antibody_training_esm/datasets/`) - Loads fragment files for training

```
preprocessing/                       src/antibody_training_esm/datasets/
├── jain/                            ├── jain.py
│   └── step2_preprocess_p5e_s2.py   │   (loads preprocessed data)
│       (CREATES fragments)          │
├── harvey/                          ├── harvey.py
│   └── step2_extract_fragments.py   │   (loads preprocessed data)
│       (CREATES fragments)          │
├── shehata/                         ├── shehata.py
│   └── step2_extract_fragments.py   │   (loads preprocessed data)
│       (CREATES fragments)          │
└── boughter/                        └── boughter.py
    └── stage2_stage3_annotation_qc.py   (loads preprocessed data)
        (CREATES fragments)
```

## Single Source of Truth (SSOT)

**Preprocessing scripts** in `preprocessing/` are the **canonical source of truth** for:
- Data transformation logic
- Filtering rules
- Quality control
- Fragment generation
- ANARCI annotation

**Dataset loaders** in `src/antibody_training_esm/datasets/` are **abstractions** for:
- Loading preprocessed fragment files
- Providing consistent API across datasets
- Validation and statistics
- Integration with training pipelines

## Usage

### Preprocessing (ONE-TIME, run scripts)

```bash
# Jain dataset
python preprocessing/jain/step2_preprocess_p5e_s2.py

# Harvey dataset
python preprocessing/harvey/step2_extract_fragments.py

# Shehata dataset
python preprocessing/shehata/step2_extract_fragments.py

# Boughter dataset
python preprocessing/boughter/stage2_stage3_annotation_qc.py
```

These scripts create fragment CSV files in:
- `test_datasets/<dataset>/fragments/`
- `train_datasets/<dataset>/fragments/`

### Loading (TRAINING, use dataset classes)

```python
# Option 1: Use dataset class
from antibody_training_esm.datasets import JainDataset

dataset = JainDataset()
df = dataset.load_data(stage="parity")  # Loads preprocessed 86-antibody set
print(f"Loaded {len(df)} sequences")

# Option 2: Use convenience function
from antibody_training_esm.datasets import load_jain_data

df = load_jain_data(stage="parity")
print(f"Loaded {len(df)} sequences")
```

## Available Dataset Loaders

### JainDataset
- **Source**: `test_datasets/jain/processed/`
- **Fragments**: `test_datasets/jain/fragments/`
- **Preprocessing**: `preprocessing/jain/step2_preprocess_p5e_s2.py`
- **Characteristics**:
  - 137 → 116 → 86 antibodies (Novo parity)
  - PSR/AC-SINS filtering
  - 16 fragment types (VH + VL)

### HarveyDataset
- **Source**: `test_datasets/harvey/raw/`
- **Fragments**: `test_datasets/harvey/fragments/`
- **Preprocessing**: `preprocessing/harvey/step2_extract_fragments.py`
- **Characteristics**:
  - 141,474 nanobody sequences (VHH only)
  - 6 fragment types (nanobody-specific)
  - IMGT position extraction

### ShehataDataset
- **Source**: `test_datasets/shehata/raw/shehata-mmc2.xlsx`
- **Fragments**: `test_datasets/shehata/fragments/`
- **Preprocessing**: `preprocessing/shehata/step2_extract_fragments.py`
- **Characteristics**:
  - 398 HIV antibodies
  - PSR threshold-based labeling (98.24th percentile)
  - 16 fragment types (VH + VL)

### BoughterDataset
- **Source**: `train_datasets/boughter/raw/` (DNA FASTA files)
- **Fragments**: `train_datasets/boughter/annotated/`
- **Preprocessing**: `preprocessing/boughter/stage2_stage3_annotation_qc.py`
- **Characteristics**:
  - Mouse antibodies (6 subsets)
  - DNA translation required
  - Novo flagging strategy (0/1-3/4+)
  - 16 fragment types (VH + VL)

## Design Principles

### Single Responsibility (SRP)
- Each dataset loader handles ONE dataset
- Preprocessing scripts handle ONE pipeline
- No overlap in responsibilities

### Open/Closed Principle (OCP)
- New datasets can be added by extending `AntibodyDataset`
- Existing code doesn't need modification
- Preprocessing scripts remain independent

### Dependency Inversion (DIP)
- Training code depends on `AntibodyDataset` abstraction
- Not on specific dataset implementations
- Preprocessing scripts are independent

## Why This Architecture?

### Benefits

1. **Clear Separation**: Preprocessing ≠ Loading
2. **Single Source of Truth**: Preprocessing scripts are authoritative
3. **Bit-for-Bit Parity**: Can validate new vs old outputs
4. **No Rewrites**: Preprocessing logic stays in scripts
5. **Professional Structure**: Industry-standard ML organization

### What This Architecture Prevents

❌ **DON'T DO THIS**:
```python
# WRONG: Trying to preprocess from dataset class
dataset = JainDataset()
# There is NO process() method - it was intentionally removed!
# Dataset loaders are for LOADING preprocessed data only.
```

✅ **DO THIS INSTEAD**:

**Step 1: Run preprocessing script ONCE to CREATE fragment files:**
```bash
python preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Step 2: LOAD the preprocessed data in your training code:**
```python
from antibody_training_esm.datasets import load_jain_data

df = load_jain_data(stage="parity")  # Fast, correct
```

## Future Work (Phase 4+)

The base class (`AntibodyDataset`) provides infrastructure for:
- `annotate_sequence()` - ANARCI annotation
- `create_fragments()` - Fragment generation
- `validate_sequences()` - Quality control

These methods are available for:
- Building NEW preprocessing pipelines
- Prototyping dataset variations
- Research experiments

But they should **NOT** be used to replace the canonical preprocessing scripts.

## Questions?

- **"Should I use dataset classes or preprocessing scripts?"**
  - **Preprocessing**: Use scripts in `preprocessing/`
  - **Training**: Use dataset classes in `src/antibody_training_esm/datasets/`

- **"Can I modify preprocessing logic in dataset classes?"**
  - **No**. Preprocessing logic belongs in `preprocessing/` scripts.
  - Dataset classes are for **loading**, not **creating** data.

- **"How do I add a new dataset?"**
  1. Create preprocessing script in `preprocessing/<new_dataset>/`
  2. Create dataset loader in `src/antibody_training_esm/datasets/<new_dataset>.py`
  3. Extend `AntibodyDataset` base class
  4. Implement `load_data()` method

---

**Remember**: Preprocessing scripts are the **SSOT**. Dataset loaders are **abstractions** for training.
