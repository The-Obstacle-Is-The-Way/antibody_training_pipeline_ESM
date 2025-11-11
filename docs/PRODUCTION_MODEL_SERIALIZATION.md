# Production Model Serialization Issue

**Created:** 2025-11-10
**Status:** Pending Senior Approval
**Priority:** Medium (Production Deployment)
**Labels:** `enhancement`, `production`
**Estimated Effort:** 3-5 hours

---

## Problem Statement

**Current State:**
The training pipeline saves models in pickle format only (`.pkl` files). While this is standard for research and experimentation, pickle has known security vulnerabilities and is not suitable for production deployment.

**Issue:**
- Pickle files can execute arbitrary code when loaded (security risk)
- Not cross-platform compatible (Python-specific)
- Not suitable for HuggingFace model hub deployment
- Cannot be loaded by non-Python applications

**Impact:**
- Limits production deployment options
- Prevents HuggingFace model sharing
- Creates security concerns for public model distribution

---

## Current Implementation

**Location:** `src/antibody_training_esm/core/trainer.py` (lines 275-293)

**Current Code:**
```python
def save_model(
    classifier: BinaryClassifier,
    config: dict[str, Any],
    logger: logging.Logger,
) -> str | None:
    """Save trained model to pickle file"""
    if not config["training"]["save_model"]:
        return None

    model_name = config["training"]["model_name"]
    model_save_dir = config["training"]["model_save_dir"]
    model_path = os.path.join(model_save_dir, f"{model_name}.pkl")

    os.makedirs(model_save_dir, exist_ok=True)

    logger.info(f"Saving model to {model_path}")

    # Save the entire classifier (including scaler and fitted model)
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    logger.info("Model saved successfully")
    return model_path
```

**What it saves:**
- ✅ Pickle checkpoint (`.pkl`) - research use only
- ❌ Production format - NOT saved

---

## Proposed Solution

**Save BOTH formats during training:**

1. **Pickle (`.pkl`)** - For research/checkpointing
   - Fast iteration
   - Debugging
   - Hyperparameter sweeps

2. **NPZ + JSON** - For production/deployment
   - Secure (no code execution)
   - Cross-platform (language-agnostic)
   - HuggingFace compatible
   - Can be loaded in Rust/C++/JavaScript

**Key Design Principle:**
Save both formats **during training** (not post-training conversion).

---

## Technical Specification

### **File Structure After Training:**

```
models/
├── boughter_vh_esm1v_logreg.pkl        # Pickle checkpoint (existing)
├── boughter_vh_esm1v_logreg.npz        # NumPy arrays (NEW)
└── boughter_vh_esm1v_logreg_config.json # Metadata (NEW)
```

### **NPZ Contents:**
```python
# Saved arrays (all required for sklearn LogisticRegression reconstruction):
{
    "coef": model.classifier.coef_,              # LogReg coefficients (1280D)
    "intercept": model.classifier.intercept_,    # LogReg intercept (scalar)
    "classes": model.classifier.classes_,        # Class labels (required for predict_proba)
    "n_features_in": model.classifier.n_features_in_,  # Number of features (validation)
    "n_iter": model.classifier.n_iter_           # Number of iterations to convergence
}
```

**Why these fields:**
- `coef_`, `intercept_`: Model weights
- `classes_`: Required by sklearn for `predict_proba()` calls
- `n_features_in_`: Input validation when loading
- `n_iter_`: Convergence diagnostics

### **JSON Contents:**
```json
{
    "model_type": "LogisticRegression",
    "sklearn_version": "1.7.2",
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
    "esm_model": "facebook/esm1v_t33_650M_UR90S_1",
    "esm_revision": "main",
    "batch_size": 8,
    "device": "mps"
}
```

**Fields explanation:**
- **LogisticRegression params**: `C`, `penalty`, `solver`, `class_weight`, `max_iter`, `random_state` (from `classifier.classifier`)
- **ESM params**: `esm_model`, `esm_revision`, `batch_size`, `device` (from `classifier.embedding_extractor`)
- **Versions**: `sklearn_version` (for compatibility checks)

**Note on `class_weight`:**
- Can be `null` (None), `"balanced"` (string), or `{"0": 1.0, "1": 2.0}` (dict)
- JSON serializes all three natively - no need for `str()` conversion
- Loader restores exact value (no parsing needed)

**Note:** Fields like `training_date`, `assay_type`, `fragment_type`, `cv_accuracy_*` are **not included** because they're not available in `save_model()` signature. These can be added later if needed by changing the function signature.

---

## Implementation Plan

### **Step 1: Refactor `save_model()` Function**

**New signature:**
```python
def save_model(
    classifier: BinaryClassifier,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Save trained model in dual format (pickle + NPZ+JSON)

    Returns:
        Dictionary with paths to saved files:
        {
            "pickle": "models/model.pkl",
            "npz": "models/model.npz",
            "config": "models/model_config.json"
        }
    """
```

### **Step 2: Implementation**

**Required import (add to top of file):**
```python
import sklearn  # For sklearn.__version__
```

**Updated function:**
```python
def save_model(
    classifier: BinaryClassifier,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, str]:
    """Save model in both pickle (research) and NPZ+JSON (production) formats"""
    if not config["training"]["save_model"]:
        return {}

    model_name = config["training"]["model_name"]
    model_save_dir = config["training"]["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)

    base_path = os.path.join(model_save_dir, model_name)

    # Format 1: Pickle checkpoint (research)
    pickle_path = f"{base_path}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved pickle checkpoint: {pickle_path}")

    # Format 2: NPZ (production arrays - all sklearn LogisticRegression fitted attributes)
    npz_path = f"{base_path}.npz"
    np.savez(
        npz_path,
        coef=classifier.classifier.coef_,
        intercept=classifier.classifier.intercept_,
        classes=classifier.classifier.classes_,
        n_features_in=np.array([classifier.classifier.n_features_in_]),
        n_iter=classifier.classifier.n_iter_,
    )
    logger.info(f"Saved NPZ arrays: {npz_path}")

    # Format 3: JSON (production metadata - all BinaryClassifier params)
    json_path = f"{base_path}_config.json"
    metadata = {
        # Model architecture
        "model_type": "LogisticRegression",
        "sklearn_version": sklearn.__version__,

        # LogisticRegression hyperparameters
        "C": classifier.C,
        "penalty": classifier.penalty,
        "solver": classifier.solver,
        "class_weight": classifier.class_weight,  # JSON handles None, str, dict natively
        "max_iter": classifier.max_iter,
        "random_state": classifier.random_state,

        # ESM embedding extractor params
        "esm_model": classifier.model_name,
        "esm_revision": classifier.revision,
        "batch_size": classifier.batch_size,
        "device": classifier.device,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved JSON config: {json_path}")

    return {
        "pickle": pickle_path,
        "npz": npz_path,
        "config": json_path
    }
```

### **Step 3: Update Return Type (Breaking Change)**

**⚠️ Breaking Change:** Return type changes from `str | None` to `dict[str, str]`

**Files requiring updates:**

1. **`src/antibody_training_esm/core/trainer.py`:**
   - **Line 11:** Add `import sklearn` (needed for `sklearn.__version__`)
   - **Line 361-373:** Update `train_model()` to return `model_paths` dict:
     ```python
     # Save model
     model_paths = save_model(classifier, config, logger)  # Now returns dict

     # Compile results
     results = {
         "train_metrics": train_results,
         "cv_metrics": cv_results,
         "config": config,
         "model_paths": model_paths,  # Changed from "model_path": model_path
     }
     ```

2. **`tests/unit/core/test_trainer.py`:**
   - **Line 480-485:** Tests calling `save_model()` directly now get dict instead of string:
     ```python
     # OLD
     model_path = save_model(classifier, nested_config, mock_logger)
     assert model_path is not None
     assert Path(model_path).exists()

     # NEW
     model_paths = save_model(classifier, nested_config, mock_logger)
     assert model_paths is not None
     assert "pickle" in model_paths
     assert Path(model_paths["pickle"]).exists()
     ```
   - **Line 511-514:** Update `test_save_model_returns_none_when_save_disabled()`:
     ```python
     # OLD
     assert model_path is None

     # NEW
     assert model_paths == {}  # Returns empty dict when disabled
     ```

3. **Downstream code** (if any uses `results["model_path"]`):
   - Change: `results["model_path"]` → `results["model_paths"]["pickle"]`
   - Currently NO tests assert on `results["model_path"]`, so no breakage in test suite

**Backward compatibility note:** This is a breaking change for any code consuming `save_model()` or `train_model()["model_path"]`.

### **Step 4: Add Loading Function (Required)**

```python
def load_model_from_npz(npz_path: str, json_path: str) -> BinaryClassifier:
    """
    Load model from NPZ+JSON format (production deployment)

    Args:
        npz_path: Path to .npz file with arrays
        json_path: Path to .json file with metadata

    Returns:
        Reconstructed BinaryClassifier instance

    Notes:
        This function is REQUIRED for production deployment.
        It reconstructs a fully functional BinaryClassifier from NPZ+JSON format.
    """
    # Load arrays
    arrays = np.load(npz_path)
    coef = arrays["coef"]
    intercept = arrays["intercept"]
    classes = arrays["classes"]
    n_features_in = int(arrays["n_features_in"][0])
    n_iter = arrays["n_iter"]

    # Load metadata
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct BinaryClassifier with ALL required params
    params = {
        # ESM params
        "model_name": metadata["esm_model"],
        "device": metadata.get("device", "cpu"),  # Use saved device or default to CPU
        "batch_size": metadata["batch_size"],
        "revision": metadata["esm_revision"],

        # LogisticRegression hyperparameters
        "C": metadata["C"],
        "penalty": metadata["penalty"],
        "solver": metadata["solver"],
        "max_iter": metadata["max_iter"],
        "random_state": metadata["random_state"],
        "class_weight": metadata["class_weight"],  # JSON restores None, str, or dict correctly
    }

    # Create classifier (initializes with unfitted LogisticRegression)
    classifier = BinaryClassifier(params)

    # Restore fitted LogisticRegression state
    classifier.classifier.coef_ = coef
    classifier.classifier.intercept_ = intercept
    classifier.classifier.classes_ = classes
    classifier.classifier.n_features_in_ = n_features_in
    classifier.classifier.n_iter_ = n_iter
    classifier.is_fitted = True

    return classifier
```

**Why this is required:**
- Enables production deployment without pickle files
- Allows cross-platform model loading (Rust/C++/JavaScript can read NPZ+JSON)
- Provides secure model distribution (no code execution risk)
- Supports HuggingFace model hub deployment

---

## Testing Plan

### **Unit Tests:**
```python
def test_save_model_dual_format():
    """Test that save_model creates both pickle and NPZ+JSON"""
    classifier = train_dummy_model()
    paths = save_model(classifier, config, logger)

    assert os.path.exists(paths["pickle"])
    assert os.path.exists(paths["npz"])
    assert os.path.exists(paths["config"])

def test_npz_arrays_match_pickle():
    """Verify NPZ arrays match pickle model coefficients"""
    classifier = train_dummy_model()
    paths = save_model(classifier, config, logger)

    # Load pickle
    with open(paths["pickle"], "rb") as f:
        pkl_model = pickle.load(f)

    # Load NPZ
    arrays = np.load(paths["npz"])

    # Compare
    np.testing.assert_array_equal(
        pkl_model.classifier.coef_,
        arrays["coef"]
    )

def test_load_from_npz():
    """Test loading model from NPZ+JSON format"""
    original = train_dummy_model()
    paths = save_model(original, config, logger)

    loaded = load_model_from_npz(paths["npz"], paths["config"])

    # Verify predictions match
    X_test = generate_test_embeddings()
    np.testing.assert_array_equal(
        original.predict(X_test),
        loaded.predict(X_test)
    )
```

### **Integration Tests:**
```python
def test_full_training_saves_dual_format():
    """Test that train_model() saves both formats"""
    results = train_model("configs/config.yaml")

    assert "model_paths" in results
    assert "pickle" in results["model_paths"]
    assert "npz" in results["model_paths"]
    assert "config" in results["model_paths"]
```

---

## Benefits

### **Security:**
- ✅ NPZ+JSON cannot execute code (unlike pickle)
- ✅ Safe for public model distribution
- ✅ No arbitrary code execution risk

### **Portability:**
- ✅ Can be loaded in any language (Rust, C++, JavaScript)
- ✅ HuggingFace model hub compatible
- ✅ Cross-platform (no Python dependency)

### **Production Readiness:**
- ✅ Industry standard for ML model deployment
- ✅ Follows best practices (used by TensorFlow, PyTorch)
- ✅ Easier to integrate with web APIs

### **Backward Compatibility:**
- ✅ Pickle format still saved (no breaking changes)
- ✅ Existing research workflows unaffected
- ✅ Gradual migration path

---

## Timeline Estimate

**Total: 3-5 hours for competent developer**

- **Step 1:** Refactor `save_model()` function (1 hour)
- **Step 2:** Add `load_model_from_npz()` function (1 hour)
- **Step 3:** Update return type and fix breaking changes in tests (1 hour)
- **Step 4:** Write unit tests (1 hour)
- **Step 5:** Write integration tests (30 min)
- **Step 6:** Update documentation (30 min)

---

## Dependencies

**Required packages (already installed):**
- `numpy` (for NPZ saving)
- `json` (standard library)
- `sklearn` (already used)

**No new dependencies needed.** ✅

---

## References

### **Industry Examples:**

**HuggingFace Transformers:**
```python
model.save_pretrained("output/")
# Saves:
# - config.json (metadata)
# - model.safetensors (weights, NOT pickle)
```

**TensorFlow:**
```python
model.save("saved_model/")
# Saves:
# - saved_model.pb (architecture)
# - variables/ (weights, NOT pickle)
```

### **Security Documentation:**
- Python Pickle Security: https://docs.python.org/3/library/pickle.html#module-pickle
- OWASP Deserialization: https://owasp.org/www-community/vulnerabilities/Deserialization_of_untrusted_data

### **Related Docs:**
- `docs/developer-guide/security.md` - Current pickle usage discussion
- `SECURITY_REMEDIATION_PLAN.md` - Pickle migration plan

---

## Questions for Senior Review

1. **Should we add optional CV metrics to JSON?**
   - Pro: Complete reproducibility (cv_accuracy_mean, cv_accuracy_std)
   - Con: Requires changing `save_model()` signature to accept CV results
   - Current: NOT included (can be added later if needed)

2. **Should we add optional training metadata to JSON?**
   - Pro: Better model tracking (training_date, dataset_name, assay_type, fragment_type)
   - Con: Requires changing `save_model()` signature or extracting from config
   - Current: NOT included (can be added later if needed)

3. **Should we deprecate pickle-only format eventually?**
   - Pro: Force production-ready practices
   - Con: Breaking change for existing workflows
   - Current: Keep both formats (gradual migration)

4. **File naming convention?**
   - Current proposal: `model.npz` + `model_config.json`
   - Alternative: `model.npz` + `model.json`

---

## Conclusion

**This is a straightforward enhancement** that:
- ✅ Adds production-ready model serialization
- ✅ Maintains backward compatibility
- ✅ Follows industry best practices
- ✅ Enables HuggingFace deployment
- ✅ Improves security posture

**Recommended Label:** `enhancement`, `production`

**Estimated Effort:** 3-5 hours

**Note:** Removed `good-first-issue` label due to:
- Breaking changes requiring careful test updates
- Complete understanding of BinaryClassifier architecture needed
- Multiple files affected across codebase

---

**Status:** Ready for senior approval and GitHub issue creation
