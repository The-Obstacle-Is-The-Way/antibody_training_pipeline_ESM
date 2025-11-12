# Directory Organization

This document describes the hierarchical directory structure for organizing models and test results.

## Motivation

As the model zoo grows (9+ models: 3 backbones × 3 classifiers), a flat directory structure becomes difficult to navigate and maintain. The hierarchical structure mirrors the conceptual organization of the model zoo and makes it easy to find models by backbone and classifier type.

## Directory Structure

### Models

Models are organized hierarchically by backbone and classifier:

```
models/
├── esm1v/
│   ├── logreg/
│   │   ├── boughter_vh_esm1v_logreg.pkl
│   │   ├── boughter_vh_esm1v_logreg.npz
│   │   └── boughter_vh_esm1v_logreg_config.json
│   ├── xgboost/
│   └── mlp/
├── esm2_650m/
│   ├── logreg/
│   └── xgboost/
└── antiberta/
    ├── logreg/
    └── xgboost/
```

**Structure:** `models/{model_shortname}/{classifier_type}/{model_files}`

**Model Shortnames:**
- `esm1v` - Facebook ESM-1v (650M)
- `esm2_650m` - Facebook ESM2 (650M)
- `esm2_3b` - Facebook ESM2 (3B)
- `antiberta` - AlchemAb AntiBERTa
- `protbert` - Rostlab ProtBERT
- `ablang` - AbLang

**Classifier Types:**
- `logreg` - Logistic Regression
- `xgboost` - XGBoost
- `mlp` - Multi-Layer Perceptron
- `svm` - Support Vector Machine
- `rf` - Random Forest

### Test Results

Test results are organized hierarchically by backbone, classifier, and dataset:

```
test_results/
├── esm1v/
│   └── logreg/
│       ├── jain/
│       │   ├── predictions.csv
│       │   └── metrics.json
│       ├── harvey/
│       └── shehata/
├── esm2_650m/
│   └── logreg/
│       ├── jain/
│       ├── harvey/
│       └── shehata/
└── antiberta/
    └── logreg/
        └── jain/
```

**Structure:** `test_results/{model_shortname}/{classifier_type}/{dataset}/`

## Implementation

### Automatic Directory Creation

The hierarchical directory structure is created automatically by the training pipeline:

```python
from antibody_training_esm.core.directory_utils import get_hierarchical_model_dir

# Generate hierarchical directory path
model_dir = get_hierarchical_model_dir(
    base_dir="./models",
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    classifier_config={"type": "logistic_regression"}
)
# Returns: Path("models/esm1v/logreg")
```

### Model Shortname Extraction

The system automatically extracts short identifiers from HuggingFace model names:

```python
from antibody_training_esm.core.directory_utils import extract_model_shortname

extract_model_shortname("facebook/esm1v_t33_650M_UR90S_1")  # "esm1v"
extract_model_shortname("facebook/esm2_t33_650M_UR50D")     # "esm2_650m"
extract_model_shortname("alchemab/antiberta2")              # "antiberta"
```

### Classifier Shortname Extraction

Similarly, classifier types are extracted from configuration:

```python
from antibody_training_esm.core.directory_utils import extract_classifier_shortname

extract_classifier_shortname({"type": "logistic_regression"})  # "logreg"
extract_classifier_shortname({"type": "xgboost"})              # "xgboost"
extract_classifier_shortname({"type": "mlp"})                  # "mlp"
```

## Migration

### Migrating Existing Models

Use the migration script to reorganize existing models:

```bash
# Preview changes (dry run)
python scripts/migrate_model_directories.py --dry-run

# Execute migration
python scripts/migrate_model_directories.py

# Custom models directory
python scripts/migrate_model_directories.py --models-dir /path/to/models
```

The script:
1. Scans the models directory for `.pkl`, `.npz`, and `.json` files
2. Parses filenames to extract model and classifier types
3. Moves files to hierarchical subdirectories
4. Preserves all file names

**Example:**
```
OLD: models/boughter_vh_esm1v_logreg.pkl
NEW: models/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

## Benefits

1. **Scalability**: Easily manage 9+ models without clutter
2. **Discoverability**: Find models by browsing `models/esm1v/logreg/`
3. **Consistency**: Matches test results directory structure
4. **Automation**: Directory creation is automatic and transparent
5. **Flexibility**: Easy to add new backbones and classifiers

## Configuration

The hierarchical structure is controlled by the `model_save_dir` config parameter:

```yaml
# conf/config.yaml
training:
  model_save_dir: ./models  # Base directory for hierarchical structure
```

The system automatically creates subdirectories:
- `{model_save_dir}/{model_shortname}/{classifier_type}/`

## FAQ

### Q: What if I want a flat structure?

The hierarchical structure is automatic, but you can still access models by their full path. All existing code that uses `model_paths["pickle"]` from `save_model()` will continue to work.

### Q: How do I load models from the new structure?

The same way as before:

```python
import pickle
with open("models/esm1v/logreg/boughter_vh_esm1v_logreg.pkl", "rb") as f:
    model = pickle.load(f)
```

Or using the utility:

```python
from antibody_training_esm.core.directory_utils import get_hierarchical_model_dir
from pathlib import Path

model_dir = get_hierarchical_model_dir("./models", model_name, classifier_config)
model_path = model_dir / "boughter_vh_esm1v_logreg.pkl"
```

### Q: Does this affect existing models?

No. Existing models in the flat structure continue to work. Use the migration script to reorganize them when ready.

### Q: Can I customize the shortnames?

Yes. Edit `extract_model_shortname()` and `extract_classifier_shortname()` in `src/antibody_training_esm/core/directory_utils.py`.

## Related Files

- `src/antibody_training_esm/core/directory_utils.py` - Directory path generation
- `src/antibody_training_esm/core/trainer.py` - Training pipeline integration
- `scripts/migrate_model_directories.py` - Migration script
- `tests/unit/core/test_directory_utils.py` - Unit tests
