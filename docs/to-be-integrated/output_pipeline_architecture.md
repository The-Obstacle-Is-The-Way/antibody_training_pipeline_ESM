# Output Pipeline Architecture: Implementation Complete

**Date:** 2025-11-11 (Updated 22:00 UTC)
**Status:** ✅ IMPLEMENTED - Hierarchical output system complete
**Author:** System trace + first principles analysis

---

## Executive Summary

**Problem:** Current output organization caused file overrides when testing multiple models.
**Impact:** ESM2-650M test results would overwrite ESM-1v baseline results.
**Solution:** ✅ IMPLEMENTED - Stratified outputs by backbone → classifier → dataset hierarchy.

**Implementation:** Complete hierarchical output system in `test.py:147-577`
- Helper functions to extract backbone/classifier from model config
- TestConfig method for computing hierarchical paths
- Automatic output directory organization
- Backward compatible (falls back to flat structure if model config missing)

---

## Current State: What We Have

### 1. Model Weights Output (`models/`)

**Location:** `./models/`

**Current Files:**
```
models/
├── .gitkeep
├── boughter_vh_esm1v_logreg.pkl          # ESM-1v weights (11KB)
├── boughter_vh_esm1v_logreg.npz          # ESM-1v NPZ format
├── boughter_vh_esm1v_logreg_config.json  # ESM-1v config
├── boughter_vh_esm2_650m_logreg.pkl      # ESM2-650M weights (11KB)
├── boughter_vh_esm2_650m_logreg.npz      # ESM2-650M NPZ format
└── boughter_vh_esm2_650m_logreg_config.json  # ESM2-650M config
```

**Naming Pattern (Code):**
```python
# From src/antibody_training_esm/core/trainer.py
model_name = config["training"]["model_name"]  # e.g., "boughter_vh_esm1v_logreg"
model_save_dir = config["training"]["model_save_dir"]  # "./models"

# Files saved:
f"{model_save_dir}/{model_name}.pkl"
f"{model_save_dir}/{model_name}.npz"
f"{model_save_dir}/{model_name}_config.json"
```

**CLI Override:**
```bash
antibody-train training.model_name=boughter_vh_esm2_650m_logreg
```

**✅ VERDICT:** Model weights are SAFE - unique names per backbone prevent overrides.

---

### 2. Test Results Output (`test_results/`)

**Location:** `./test_results/{dataset_name}/` (MANUAL STRATIFICATION)

**IMPORTANT:** The test CLI (test.py:63) writes all artifacts to a single `output_dir` (default: `./test_results`). The existing subdirectory structure below was created by manually running the CLI with `--output-dir test_results/jain`, `--output-dir test_results/harvey`, etc. The tool does NOT automatically stratify by dataset—it writes everything to whatever `output_dir` you pass.

**Current Files (ESM-1v baseline):**
```
test_results/
├── jain/  # Created via: antibody-test --output-dir test_results/jain
│   ├── confusion_matrix_VH_only_jain_test_PARITY_86.png
│   ├── detailed_results_VH_only_jain_test_PARITY_86_20251106_211815.yaml
│   └── predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251106_211815.csv
├── harvey/  # Created via: antibody-test --output-dir test_results/harvey
│   ├── confusion_matrix_VHH_only_harvey.png
│   ├── detailed_results_VHH_only_harvey_20251106_223905.yaml
│   └── predictions_boughter_vh_esm1v_logreg_VHH_only_harvey_20251106_223905.csv
└── shehata/  # Created via: antibody-test --output-dir test_results/shehata
    ├── confusion_matrix_VH_only_shehata.png
    ├── detailed_results_VH_only_shehata_20251106_212500.yaml
    └── predictions_boughter_vh_esm1v_logreg_VH_only_shehata_20251106_212500.csv
```

**Naming Patterns (Code Analysis):**

From `src/antibody_training_esm/cli/test.py`:

```python
# Line 442: Extract model name from path
model_name = Path(model_path).stem  # e.g., "boughter_vh_esm1v_logreg"

# Line 353: Confusion matrix (NO MODEL NAME!)
confusion_matrix_file = f"confusion_matrix_{dataset_name}.png"
# ❌ OVERRIDE RISK: Same filename for all models!

# Line 387: Predictions CSV (includes model name + timestamp)
predictions_file = f"predictions_{model_name}_{dataset_name}_{timestamp}.csv"
# ✅ SAFE: Unique per model + timestamp

# Line 393: Detailed results YAML (includes timestamp)
yaml_file = f"detailed_results_{dataset_name}_{timestamp}.yaml"
# ⚠️ PARTIAL RISK: No model name, but timestamp may differ
```

**✅ RESOLVED: Confusion Matrix Override Risk (FIXED)**

**Previous Risk** (now resolved):
```bash
# Test ESM-1v on Jain
antibody-test --model models/boughter_vh_esm1v_logreg.pkl --dataset jain
# Creates: test_results/jain/confusion_matrix_VH_only_jain_test_PARITY_86.png

# Test ESM2-650M on Jain
antibody-test --model models/boughter_vh_esm2_650m_logreg.pkl --dataset jain
# Previously OVERWROTE: test_results/jain/confusion_matrix_VH_only_jain_test_PARITY_86.png ❌
```

**Current Behavior** (after fix):
```bash
# Test ESM-1v on Jain
antibody-test --model models/boughter_vh_esm1v_logreg.pkl --dataset jain
# Creates: test_results/esm1v/logreg/jain/confusion_matrix_boughter_vh_esm1v_logreg_jain.png ✅

# Test ESM2-650M on Jain
antibody-test --model models/boughter_vh_esm2_650m_logreg.pkl --dataset jain
# Creates: test_results/esm2_650m/logreg/jain/confusion_matrix_boughter_vh_esm2_650m_logreg_jain.png ✅
```

**Both baseline and comparison results are preserved in separate hierarchical directories.**

---

### 3. Training Outputs (`outputs/`)

**Location:** `./outputs/{experiment.name}/{timestamp}/`

**Current Structure:**
```
outputs/
├── novo_replication/
│   ├── 2025-11-11_18-00-50/  # ESM-1v training run
│   │   ├── .hydra/
│   │   │   └── config.yaml
│   │   ├── logs/
│   │   │   └── training.log
│   │   └── trainer.log
│   ├── 2025-11-11_21-48-22/  # ESM-1v (from earlier attempt)
│   └── 2025-11-11_21-52-43/  # ESM2-650M training run ✅
│       ├── .hydra/
│       │   └── config.yaml  # Contains model.name=facebook/esm2_t33_650M_UR50D
│       ├── logs/
│       │   └── training.log  # Training metrics + benchmark results
│       └── trainer.log
└── test_dataset/
    └── ...
```

**Naming Pattern (Hydra):**
```python
# Experiment name from config
experiment.name = "novo_replication"  # Default

# Hydra auto-generates timestamp directories
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"outputs/{experiment.name}/{timestamp}/"
```

**✅ VERDICT:** Training outputs are SAFE - timestamped directories prevent overrides.

---

## Override Risk Matrix

| Output Type | File Pattern | Model Name? | Timestamp? | Override Risk |
|-------------|-------------|-------------|------------|---------------|
| **Model Weights (.pkl)** | `{model_name}.pkl` | ✅ Yes | ❌ No | 🟢 SAFE |
| **Model Weights (.npz)** | `{model_name}.npz` | ✅ Yes | ❌ No | 🟢 SAFE |
| **Model Config (.json)** | `{model_name}_config.json` | ✅ Yes | ❌ No | 🟢 SAFE |
| **Predictions CSV** | `predictions_{model}_{dataset}_{time}.csv` | ✅ Yes | ✅ Yes | 🟢 SAFE |
| **Confusion Matrix PNG** | `confusion_matrix_{model}_{dataset}.png` | ✅ YES | ❌ No | ✅ **FIXED** |
| **Detailed Results YAML** | `detailed_results_{model}_{dataset}_{time}.yaml` | ✅ YES | ✅ Yes | ✅ **FIXED** |
| **Training Logs** | `outputs/{exp}/{timestamp}/` | N/A | ✅ Yes | 🟢 SAFE |

**Note:** Confusion matrix and detailed results now include model names in filenames (Phase 1 fix) AND are organized hierarchically (Phase 2 fix).

---

## Implementation Status ✅ COMPLETE

### What Was Implemented

**File:** `src/antibody_training_esm/cli/test.py`

**1. Helper Functions** (lines 60-120)
- `extract_backbone_from_config()`: Extracts backbone from model config ("esm1v", "esm2_650m", "antiberta")
- `extract_classifier_from_config()`: Extracts classifier from model config ("logreg", "xgboost", "mlp")

**2. TestConfig Enhancement** (lines 147-168)
- Added `get_hierarchical_output_dir()` method
- Computes paths like: `test_results/{backbone}/{classifier}/{dataset}/`
- Example: `get_hierarchical_output_dir("esm1v", "logreg", "jain")` → `"./test_results/esm1v/logreg/jain"`

**3. Output Functions Refactored** (lines 405-513)
- `plot_confusion_matrix()`: Now accepts `output_dir` parameter, creates hierarchical directories
- `save_detailed_results()`: Now accepts `output_dir` parameter, creates hierarchical directories
- Both functions include model names in filenames (prevents collisions within same directory)

**4. Automatic Path Computation** (lines 526-577)
- `_compute_output_directory()`: Helper that loads model config JSON and extracts metadata
- Calls helper functions to determine backbone and classifier
- Computes hierarchical path automatically
- Falls back to flat structure if model config missing (backward compatible)

**5. Integration** (lines 580-647)
- `run_comprehensive_test()` calls `_compute_output_directory()` for each dataset
- Passes computed hierarchical paths to plotting and saving functions
- Creates directory structure automatically

### Result

```
test_results/
├── esm1v/
│   └── logreg/
│       ├── jain/
│       │   ├── confusion_matrix_boughter_vh_esm1v_logreg_jain.png
│       │   ├── detailed_results_boughter_vh_esm1v_logreg_jain_20251111_220000.yaml
│       │   └── predictions_boughter_vh_esm1v_logreg_jain_20251111_220000.csv
│       ├── harvey/
│       └── shehata/
└── esm2_650m/
    └── logreg/
        ├── jain/
        │   ├── confusion_matrix_boughter_vh_esm2_650m_logreg_jain.png  # No override!
        │   └── ...
        └── ...
```

---

## Original Proposed Fix (Now Implemented)

### Design Principles

1. **Stratify by Backbone First**: `esm1v/`, `esm2_650m/`, `antiberta/`
2. **Then by Classifier**: `logreg/`, `xgboost/`, `mlp/`
3. **Preserve Model Name for Flat Access**: Keep `boughter_vh_esm1v_logreg.pkl` naming
4. **Avoid Breaking Changes**: Backward compatible with existing paths

### Proposed Directory Structure

**CRITICAL:** This requires CODE CHANGES, not just filesystem reorganization. The trainer currently writes to a flat `training.model_save_dir` (config.yaml:21). To implement this hierarchy, we must either:
  a) Add new config knobs for `backbone` and `classifier` subdirectories, or
  b) Add logic in `train_pipeline()` (trainer.py) to derive subdirectories from `cfg.model.name` and `cfg.classifier` and dynamically update `model_save_dir` before saving.

Without these code changes, creating empty folders won't help—the trainer will still write to the root `./models/` directory.

```
models/
├── esm1v/
│   ├── logreg/
│   │   ├── boughter_vh_esm1v_logreg.pkl
│   │   ├── boughter_vh_esm1v_logreg.npz
│   │   └── boughter_vh_esm1v_logreg_config.json
│   ├── xgboost/  # Week 2
│   │   ├── boughter_vh_esm1v_xgboost.pkl
│   │   └── ...
│   └── mlp/  # Week 3
│       └── ...
├── esm2_650m/
│   ├── logreg/
│   │   ├── boughter_vh_esm2_650m_logreg.pkl
│   │   ├── boughter_vh_esm2_650m_logreg.npz
│   │   └── boughter_vh_esm2_650m_logreg_config.json
│   ├── xgboost/  # Week 2
│   └── mlp/      # Week 3
└── antiberta/    # Week 3
    ├── logreg/
    ├── xgboost/
    └── mlp/
```

**Result:** 9 models organized cleanly (3 backbones × 3 classifiers).

### Test Results Stratification

**CRITICAL:** This also requires CODE CHANGES in test.py. The current implementation writes to a single flat `self.config.output_dir`. To achieve this hierarchy, we must:
  1. Extract backbone and classifier from the model config or filename
  2. Dynamically compute output_dir as `test_results/{backbone}/{classifier}/{dataset}/`
  3. Update lines 353, 368, 387 in test.py to use this new directory structure

```
test_results/
├── esm1v/
│   ├── logreg/
│   │   ├── jain/
│   │   │   ├── confusion_matrix_jain.png
│   │   │   ├── predictions_jain_20251106.csv
│   │   │   └── detailed_results_jain_20251106.yaml
│   │   ├── harvey/
│   │   │   └── ...
│   │   └── shehata/
│   │       └── ...
│   ├── xgboost/
│   │   ├── jain/
│   │   └── ...
│   └── mlp/
│       └── ...
├── esm2_650m/
│   ├── logreg/
│   │   ├── jain/
│   │   │   ├── confusion_matrix_jain.png  # No override!
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── antiberta/
    └── ...
```

**Result:** No overrides - each model's test results isolated.

---

## Implementation Plan

### Phase 1: Fix Test CLI (Immediate - Before Running ESM2 Tests)

**Changes Required:**

1. **Add model name to confusion matrix filename**

```python
# BEFORE (test.py:353)
plot_file = f"confusion_matrix_{dataset_name}.png"

# AFTER
plot_file = f"confusion_matrix_{model_name}_{dataset_name}.png"
```

2. **Add model name to YAML filename**

```python
# BEFORE (test.py:393)
yaml_file = f"detailed_results_{dataset_name}_{timestamp}.yaml"

# AFTER
yaml_file = f"detailed_results_{model_name}_{dataset_name}_{timestamp}.yaml"
```

3. **Update output directory structure (REQUIRES CODE CHANGES)**

**Location:** `test.py` lines 60-91 (TestConfig initialization and ModelTester.__init__)

```python
# BEFORE (test.py:63)
output_dir: str = "./test_results"  # Flat, single directory

# AFTER (hierarchical - requires parsing model config)
# Option A: Pass backbone/classifier as CLI args
@dataclass
class TestConfig:
    backbone: str  # e.g., "esm1v", "esm2_650m"
    classifier: str  # e.g., "logreg", "xgboost"
    output_base_dir: str = "./test_results"

    def get_output_dir(self, dataset_name: str) -> str:
        return f"{self.output_base_dir}/{self.backbone}/{self.classifier}/{dataset_name}"

# Option B: Extract from model config JSON (better - automatic)
# Load model config from {model_path}_config.json
# Parse backbone and classifier from config
# Compute output_dir dynamically in ModelTester.__init__
```

**Implementation Note:** Without this code change, creating subdirectories manually won't prevent overrides—the CLI will still write to the root output_dir.

**Testing:**
```bash
# After fix
antibody-test --model models/boughter_vh_esm2_650m_logreg.pkl --dataset jain
# Output: test_results/esm2_650m/logreg/jain/confusion_matrix_esm2_650m_logreg_jain.png
```

### Phase 2: Reorganize Existing Results (Before New Tests)

**Move baseline results to new structure:**

```bash
# Backup first
cp -r test_results test_results_BACKUP_20251111

# Create new structure
mkdir -p test_results/esm1v/logreg/{jain,harvey,shehata}

# Move existing results
mv test_results/jain/* test_results/esm1v/logreg/jain/
mv test_results/harvey/* test_results/esm1v/logreg/harvey/
mv test_results/shehata/* test_results/esm1v/logreg/shehata/
```

### Phase 3: Reorganize Model Weights (Requires Trainer Code Changes)

**Can defer until Week 2 (XGBoost)** - current flat structure works for 2 models.

**CRITICAL:** This is NOT just a filesystem operation. We must modify `train_pipeline()` in trainer.py to:
1. Extract backbone identifier from `cfg.model.name` (e.g., "esm1v" from "facebook/esm1v_t33_650M_UR90S_1")
2. Extract classifier identifier from `cfg.classifier` (e.g., "logreg")
3. Update `model_save_dir` to include subdirectories:
   ```python
   # trainer.py, around line 700-800 where models are saved
   backbone = extract_backbone_name(cfg.model.name)  # "esm1v", "esm2_650m", etc.
   classifier = cfg.classifier.get("type", "logreg")  # From config
   model_save_dir = Path(cfg.training.model_save_dir) / backbone / classifier
   model_save_dir.mkdir(parents=True, exist_ok=True)
   ```

When we add XGBoost:
```bash
# After implementing the code changes, the trainer will automatically create:
models/
├── esm1v/
│   ├── logreg/
│   │   ├── boughter_vh_esm1v_logreg.pkl
│   │   └── ...
│   └── xgboost/
│       ├── boughter_vh_esm1v_xgboost.pkl
│       └── ...
└── esm2_650m/
    ├── logreg/
    └── xgboost/
```

---

## Migration Strategy

### Option A: Fix Now (Recommended)

1. ✅ Fix test CLI today (30 min)
2. ✅ Reorganize existing test results (5 min)
3. ✅ Test ESM2-650M on Jain without override risk
4. ⏳ Defer model weight reorganization until Week 2

### Option B: Quick Hotfix (If Urgent)

1. Just rename confusion matrix files manually after testing
2. Don't reorganize directories yet
3. Risk: Might forget, lose baseline results

### Option C: Do Nothing (Not Recommended)

1. Risk losing ESM-1v baseline confusion matrices
2. Manual tracking nightmare with 9 models coming

---

## Code Locations to Change

### 1. Test CLI: `src/antibody_training_esm/cli/test.py`

**IMMEDIATE CHANGES (Phase 1):**

**Line 353:** Confusion matrix filename - ADD model_name
```python
# CURRENT
plot_file = os.path.join(
    self.config.output_dir, f"confusion_matrix_{dataset_name}.png"
)

# FIXED
plot_file = os.path.join(
    self.config.output_dir, f"confusion_matrix_{model_name}_{dataset_name}.png"
)
```

**Line 368:** YAML filename - ADD model_name (currently line 393 in docstring, but actual location is 368)
```python
# CURRENT
results_file = os.path.join(
    self.config.output_dir, f"detailed_results_{dataset_name}_{timestamp}.yaml"
)

# FIXED
results_file = os.path.join(
    self.config.output_dir, f"detailed_results_{model_name}_{dataset_name}_{timestamp}.yaml"
)
```

**Line 387:** Predictions CSV - ALREADY CORRECT ✅
```python
pred_file = os.path.join(
    self.config.output_dir,
    f"predictions_{model_name}_{dataset_name}_{timestamp}.csv",
)
```

**HIERARCHICAL CHANGES (Phase 1 - if implementing full stratification):**

**Lines 56-79:** TestConfig - add backbone/classifier extraction
```python
@dataclass
class TestConfig:
    model_paths: list[str]
    data_paths: list[str]
    sequence_column: str = "sequence"
    label_column: str = "label"
    output_base_dir: str = "./test_results"  # Changed from output_dir
    # ... rest of fields

    def get_output_dir(self, backbone: str, classifier: str, dataset: str) -> str:
        """Compute hierarchical output directory"""
        return f"{self.output_base_dir}/{backbone}/{classifier}/{dataset}"
```

**Lines 420-520:** ModelTester.run_comprehensive_test - extract metadata and use hierarchical dirs
```python
# Inside the loop over models (around line 440)
# Load model config to extract backbone/classifier
model_config_path = Path(model_path).with_suffix('.json').with_name(
    Path(model_path).stem + '_config.json'
)
with open(model_config_path) as f:
    model_config = json.load(f)

backbone = extract_backbone_from_config(model_config)  # NEW helper
classifier = extract_classifier_from_config(model_config)  # NEW helper
dataset_name = extract_dataset_name(data_path)  # Existing or NEW

# Update output dir dynamically for this model/dataset combo
self.config.output_dir = self.config.get_output_dir(backbone, classifier, dataset_name)
os.makedirs(self.config.output_dir, exist_ok=True)
```

**NEW HELPER FUNCTIONS (add to test.py):**
```python
def extract_backbone_from_config(config: dict[str, Any]) -> str:
    """Extract backbone identifier from model config.

    Examples:
        facebook/esm1v_t33_650M_UR90S_1 → esm1v
        facebook/esm2_t33_650M_UR50D → esm2_650m
        allenai/biomed_roberta_base → antiberta
    """
    model_name = config.get("model_name", "")
    if "esm1v" in model_name.lower():
        return "esm1v"
    elif "esm2" in model_name.lower() and "650" in model_name:
        return "esm2_650m"
    elif "antiberta" in model_name.lower() or "biomed_roberta" in model_name.lower():
        return "antiberta"
    else:
        raise ValueError(f"Unknown backbone in model name: {model_name}")

def extract_classifier_from_config(config: dict[str, Any]) -> str:
    """Extract classifier identifier from model config.

    Returns:
        "logreg", "xgboost", or "mlp"
    """
    classifier_type = config.get("classifier", {}).get("type", "logreg")
    return classifier_type
```

### 2. Train CLI: `src/antibody_training_esm/core/trainer.py`

**ONLY IF REORGANIZING MODEL WEIGHTS (Phase 3 - defer to Week 2):**

**Lines 700-800:** Model saving logic in train_pipeline()
```python
# Find where models are saved (search for "model_save_dir")
# Current:
model_save_dir = cfg.training.model_save_dir  # "./models"

# After Phase 3:
backbone = extract_backbone_from_model_name(cfg.model.name)
classifier = cfg.classifier.get("type", "logreg")
model_save_dir = Path(cfg.training.model_save_dir) / backbone / classifier
model_save_dir.mkdir(parents=True, exist_ok=True)
```

**NEW HELPER FUNCTION (add to trainer.py):**
```python
def extract_backbone_from_model_name(model_name: str) -> str:
    """Extract backbone identifier from HuggingFace model name.

    Examples:
        facebook/esm1v_t33_650M_UR90S_1 → esm1v
        facebook/esm2_t33_650M_UR50D → esm2_650m
    """
    # Same logic as test.py helper
    if "esm1v" in model_name.lower():
        return "esm1v"
    elif "esm2" in model_name.lower() and "650" in model_name:
        return "esm2_650m"
    elif "antiberta" in model_name.lower() or "biomed_roberta" in model_name.lower():
        return "antiberta"
    else:
        raise ValueError(f"Unknown backbone: {model_name}")
```

### 3. Entry Point: `pyproject.toml`

**ONLY IF FIXING HYDRA CONFIG GROUP OVERRIDES (see training_pipeline_investigation.md):**

**Line 76:** Console script entry point
```toml
# CURRENT (wrapper causes config group overrides to fail)
antibody-train = "antibody_training_esm.cli.train:main"

# OPTION 1: Direct to Hydra-decorated function
antibody-train = "antibody_training_esm.core.trainer:main"

# OPTION 2: Keep wrapper but make it a thin passthrough (requires more work)
```

---

## Verification Checklist

Before running ESM2-650M tests:

- [ ] Code updated: Confusion matrix includes model name
- [ ] Code updated: YAML includes model name
- [ ] Code updated: Output dir stratified by backbone/classifier
- [ ] Existing results backed up: `test_results_BACKUP_20251111/`
- [ ] Existing results moved to new structure
- [ ] Test with ESM-1v first to verify no regression
- [ ] Test with ESM2-650M to verify no override
- [ ] Compare both confusion matrices side-by-side

---

## Expected Test Output (After Fix)

```bash
# Test ESM-1v
antibody-test --model models/boughter_vh_esm1v_logreg.pkl --dataset jain

# Output:
test_results/esm1v/logreg/jain/
├── confusion_matrix_esm1v_logreg_jain.png
├── predictions_esm1v_logreg_jain_20251111.csv
└── detailed_results_esm1v_logreg_jain_20251111.yaml

# Test ESM2-650M
antibody-test --model models/boughter_vh_esm2_650m_logreg.pkl --dataset jain

# Output (NO OVERRIDE!):
test_results/esm2_650m/logreg/jain/
├── confusion_matrix_esm2_650m_logreg_jain.png  # Different file!
├── predictions_esm2_650m_logreg_jain_20251111.csv
└── detailed_results_esm2_650m_logreg_jain_20251111.yaml
```

---

## Helper Functions Needed

```python
def extract_backbone(model_name: str) -> str:
    """Extract backbone name from model filename.

    Examples:
        boughter_vh_esm1v_logreg → esm1v
        boughter_vh_esm2_650m_logreg → esm2_650m
        boughter_vh_antiberta_xgboost → antiberta
    """
    # Pattern: boughter_vh_{backbone}_{classifier}
    parts = model_name.split("_")
    # Find backbone (between "vh" and classifier)
    # This is fragile - better to parse from config

def extract_classifier(model_name: str) -> str:
    """Extract classifier name from model filename.

    Examples:
        boughter_vh_esm1v_logreg → logreg
        boughter_vh_esm2_650m_xgboost → xgboost
    """
    # Last component after last underscore
    # Also fragile - better to parse from config
```

**Better approach:** Store backbone/classifier in model config JSON, read from there.

---

## Summary

**Current State:**
- ✅ Model weights: Safe (unique names)
- 🔴 Test results: OVERRIDE RISK (confusion matrix)
- ✅ Training logs: Safe (timestamped)

**Immediate Action Required:**
1. Fix test CLI to include model name in all outputs
2. Reorganize existing test results before ESM2 testing
3. Then proceed with benchmarking

**Next Steps:**
1. Review this document
2. Decide: Fix now vs. quick hotfix
3. Implement chosen approach
4. Verify with ESM-1v first
5. Run ESM2-650M benchmarks safely

---

---

## Implementation Priority & Effort Estimates

### CRITICAL PATH (Must Fix Before ESM2 Benchmarks)

**Phase 1A: Immediate Override Prevention (30 minutes)**
- Fix confusion matrix filename (test.py:353) - ADD model_name
- Fix YAML filename (test.py:368) - ADD model_name
- **Blocks:** ESM2 benchmark on Jain (would overwrite ESM-1v baseline)
- **Effort:** 5 min code + 5 min test + 20 min validation
- **Impact:** Prevents data loss, enables safe multi-model testing

**Phase 1B: Hierarchical Test Output (2 hours)**
- Add TestConfig.get_output_dir() method (test.py:56-79)
- Add helper functions for backbone/classifier extraction (test.py)
- Update ModelTester.run_comprehensive_test() to use hierarchical dirs (test.py:420-520)
- **Blocks:** Clean organization for 9-model benchmark (3×3 matrix)
- **Effort:** 30 min code + 30 min helpers + 30 min testing + 30 min validation
- **Impact:** Scales cleanly to full model zoo

**Phase 2: Reorganize Existing Results (15 minutes)**
- Backup current test_results/ directory
- Move ESM-1v results to test_results/esm1v/logreg/{jain,harvey,shehata}/
- **Blocks:** Consistent baseline for comparisons
- **Effort:** 5 min backup + 5 min moves + 5 min verification
- **Impact:** Preserves existing work in new structure

### DEFERRED (Week 2 - When Adding XGBoost)

**Phase 3: Hierarchical Model Weights (3 hours)**
- Add extract_backbone_from_model_name() helper (trainer.py)
- Update model saving logic to use backbone/classifier subdirs (trainer.py:700-800)
- Migrate existing model weights to new structure
- Update test code to find models in new locations
- **Blocks:** Clean organization when we have 6+ models
- **Effort:** 1 hr code + 1 hr testing + 1 hr migration
- **Impact:** Scales cleanly, but current flat structure works for 2 models

### OPTIONAL (If Fixing Hydra Override Bug)

**Fix CLI Entry Point (30 minutes)**
- Change pyproject.toml:76 to point directly to Hydra-decorated function
- Add integration test for config group overrides
- Update documentation
- **Blocks:** Nothing (workaround exists: use `python -m` invocation)
- **Effort:** 5 min code + 15 min test + 10 min docs
- **Impact:** Better UX, follows Hydra best practices

---

## Recommended Implementation Order

**TODAY (before ESM2 benchmarks):**
1. Phase 1A (override prevention) - 30 min - **MANDATORY**
2. Phase 1B (hierarchical output) - 2 hr - **STRONGLY RECOMMENDED**
3. Phase 2 (reorganize existing) - 15 min - **RECOMMENDED**

**Total effort for clean ESM2 benchmarks:** ~2.75 hours

**Week 2 (when adding XGBoost):**
4. Phase 3 (model weight hierarchy) - 3 hr

**Optional (better UX):**
5. Fix CLI entry point - 30 min

---

## Success Criteria

After implementing Phase 1A + 1B + 2, we should be able to:

```bash
# Test ESM-1v on Jain
antibody-test --model models/boughter_vh_esm1v_logreg.pkl \
              --dataset data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv

# Output: test_results/esm1v/logreg/jain/
#   ├── confusion_matrix_jain.png
#   ├── predictions_jain_20251111.csv
#   └── detailed_results_jain_20251111.yaml

# Test ESM2-650M on Jain
antibody-test --model models/boughter_vh_esm2_650m_logreg.pkl \
              --dataset data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv

# Output: test_results/esm2_650m/logreg/jain/  ✅ NO OVERRIDE!
#   ├── confusion_matrix_jain.png
#   ├── predictions_jain_20251111.csv
#   └── detailed_results_jain_20251111.yaml

# Compare confusion matrices side-by-side
open test_results/esm1v/logreg/jain/confusion_matrix_jain.png
open test_results/esm2_650m/logreg/jain/confusion_matrix_jain.png
```

---

## Validation Complete

This document has been validated against the actual codebase (2025-11-11). All claims have been verified from first principles by examining the source code directly:

- ✅ test.py:63 default output_dir behavior confirmed
- ✅ test.py:353 confusion matrix override risk confirmed
- ✅ test.py:368 YAML filename override risk confirmed
- ✅ test.py:387 predictions CSV safety confirmed
- ✅ config.yaml:21 flat model_save_dir confirmed
- ✅ Code change requirements explicitly documented

**The obstacle is the way. Trace first, fix properly, then execute.** 🚀
