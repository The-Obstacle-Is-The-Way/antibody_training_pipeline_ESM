# XGBoost Classifier - Test Plan

**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-15
**Philosophy:** Test behaviors, not implementation. Minimal mocking. Real tests for production quality.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Coverage Targets](#test-coverage-targets)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [End-to-End Tests](#end-to-end-tests)
6. [Performance Benchmarks](#performance-benchmarks)
7. [TDD Implementation Plan](#tdd-implementation-plan)

---

## Testing Philosophy

### Core Principles (From Your Existing Test Suite)

```python
"""
Philosophy:
- Test behaviors (WHAT code does), not implementation (HOW it does it)
- Minimal mocking (only ESM model loading)
- Test edge cases and error handling
"""
```

**NO BOGUS TESTS:**
- ❌ Testing private methods
- ❌ Testing implementation details (e.g., "assert classifier uses XGBClassifier")
- ❌ Excessive mocking that doesn't test real behavior
- ❌ Brittle tests that break when refactoring

**YES TO REAL TESTS:**
- ✅ Test public API behavior
- ✅ Test actual predictions on real (small) datasets
- ✅ Test error handling with invalid inputs
- ✅ Test serialization round-trips (save → load → predict same results)

### Mocking Strategy

**ONLY mock:**
1. **ESM model loading** (via `tests/fixtures/mock_models.py`)
   - Reason: Downloading 650M parameters is expensive
   - Mock returns deterministic embeddings (not random!)

**NEVER mock:**
1. **Classifiers** (LogReg, XGBoost) - test real sklearn/xgboost!
2. **Embeddings** (use cached or mock extractor output, but test real arrays)
3. **Predictions** (test actual model predictions on real data)
4. **Serialization** (test actual file I/O with tempdir)

---

## Test Coverage Targets

### Overall Coverage

- **Unit Tests:** 100% coverage for new code
- **Integration Tests:** 95%+ coverage for workflows
- **E2E Tests:** Full pipeline coverage (train → save → load → predict)

### File-Level Coverage

| File | Lines | Coverage Target | Critical Paths |
|------|-------|-----------------|----------------|
| `classifier_strategy.py` | ~50 | 100% | Protocol definition |
| `logistic_regression.py` | ~150 | 100% | fit(), predict(), serialization |
| `xgboost_strategy.py` | ~200 | 100% | fit(), predict(), save_model() |
| `classifier_factory.py` | ~50 | 100% | create_classifier(), registry |
| `classifier.py` (modified) | ~340 | 100% | BinaryClassifier delegates to strategies |

### Test Markers

```python
@pytest.mark.unit          # Fast (<1s), isolated, mocked ESM
@pytest.mark.integration   # Multi-component, real classifiers
@pytest.mark.e2e           # Full pipeline, expensive
```

---

## Unit Tests

### Test File: `tests/unit/core/strategies/test_logistic_regression.py`

**Purpose:** Test `LogisticRegressionStrategy` behavior (EXISTING classifier refactored)

#### Test 1: Initialization

```python
@pytest.mark.unit
def test_logreg_strategy_initializes_with_defaults() -> None:
    """Verify LogRegStrategy initializes with default hyperparameters."""
    # Arrange
    config = {}  # Empty config - should use defaults

    # Act
    strategy = LogisticRegressionStrategy(config)

    # Assert
    assert strategy.C == 1.0
    assert strategy.penalty == "l2"
    assert strategy.solver == "lbfgs"
    assert strategy.max_iter == 1000
    assert strategy.random_state == 42


@pytest.mark.unit
def test_logreg_strategy_initializes_with_custom_params() -> None:
    """Verify LogRegStrategy accepts custom hyperparameters."""
    # Arrange
    config = {
        "C": 0.1,
        "penalty": "l1",
        "solver": "liblinear",
        "max_iter": 500,
        "random_state": 123,
    }

    # Act
    strategy = LogisticRegressionStrategy(config)

    # Assert
    assert strategy.C == 0.1
    assert strategy.penalty == "l1"
    assert strategy.solver == "liblinear"
    assert strategy.max_iter == 500
    assert strategy.random_state == 123
```

**Philosophy:** Test BEHAVIOR (initialization works), not implementation (how params are stored).

#### Test 2: Fit & Predict (REAL sklearn behavior)

```python
@pytest.mark.unit
def test_logreg_strategy_fits_and_predicts_on_simple_dataset() -> None:
    """Verify LogRegStrategy can fit and predict on linearly separable data."""
    # Arrange: Create simple 2D dataset (linearly separable)
    X_train = np.array([
        [0.0, 0.0],  # Class 0
        [0.1, 0.1],  # Class 0
        [1.0, 1.0],  # Class 1
        [1.1, 1.1],  # Class 1
    ])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([
        [0.05, 0.05],  # Should predict 0
        [1.05, 1.05],  # Should predict 1
    ])

    config = {"random_state": 42}
    strategy = LogisticRegressionStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)

    # Assert: LogReg should learn this simple pattern
    assert predictions[0] == 0  # Near [0, 0] → class 0
    assert predictions[1] == 1  # Near [1, 1] → class 1


@pytest.mark.unit
def test_logreg_strategy_predict_proba_returns_valid_probabilities() -> None:
    """Verify predict_proba returns probabilities that sum to 1."""
    # Arrange
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    config = {"random_state": 42}
    strategy = LogisticRegressionStrategy(config)
    strategy.fit(X_train, y_train)

    X_test = np.random.rand(10, 10)

    # Act
    probs = strategy.predict_proba(X_test)

    # Assert
    assert probs.shape == (10, 2)  # (n_samples, n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities
```

**Philosophy:** Test REAL sklearn behavior, not mocked predictions.

#### Test 3: Serialization (Round-Trip Test)

```python
@pytest.mark.unit
def test_logreg_strategy_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify serialize → deserialize → predict gives same results."""
    # Arrange: Train model
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 10)

    config = {"C": 0.5, "random_state": 42}
    strategy = LogisticRegressionStrategy(config)
    strategy.fit(X_train, y_train)

    # Get original predictions
    original_preds = strategy.predict(X_test)
    original_probs = strategy.predict_proba(X_test)

    # Act: Serialize
    config_dict = strategy.to_dict()
    arrays_dict = strategy.to_arrays()

    # Save to disk (real file I/O)
    json_path = tmp_path / "config.json"
    npz_path = tmp_path / "arrays.npz"

    with open(json_path, "w") as f:
        json.dump(config_dict, f)
    np.savez(npz_path, **arrays_dict)

    # Load from disk
    with open(json_path) as f:
        loaded_config = json.load(f)
    loaded_arrays = dict(np.load(npz_path))

    # Deserialize
    loaded_strategy = LogisticRegressionStrategy.from_dict(loaded_config, loaded_arrays)

    # Assert: Predictions match exactly
    loaded_preds = loaded_strategy.predict(X_test)
    loaded_probs = loaded_strategy.predict_proba(X_test)

    np.testing.assert_array_equal(loaded_preds, original_preds)
    np.testing.assert_allclose(loaded_probs, original_probs, rtol=1e-10)
```

**Philosophy:** Test REAL serialization, not mocked file I/O. Use `tmp_path` fixture for temp files.

---

### Test File: `tests/unit/core/strategies/test_xgboost_strategy.py`

**Purpose:** Test `XGBoostStrategy` behavior (NEW classifier)

#### Test 1: Initialization

```python
@pytest.mark.unit
def test_xgboost_strategy_initializes_with_defaults() -> None:
    """Verify XGBoostStrategy initializes with default hyperparameters."""
    # Arrange
    config = {}

    # Act
    strategy = XGBoostStrategy(config)

    # Assert
    assert strategy.n_estimators == 100
    assert strategy.max_depth == 6
    assert strategy.learning_rate == 0.3
    assert strategy.random_state == 42
    assert strategy.objective == "binary:logistic"


@pytest.mark.unit
def test_xgboost_strategy_initializes_with_custom_params() -> None:
    """Verify XGBoostStrategy accepts custom hyperparameters."""
    # Arrange
    config = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "reg_lambda": 2.0,
        "random_state": 123,
    }

    # Act
    strategy = XGBoostStrategy(config)

    # Assert
    assert strategy.n_estimators == 50
    assert strategy.max_depth == 4
    assert strategy.learning_rate == 0.1
    assert strategy.reg_lambda == 2.0
    assert strategy.random_state == 123
```

#### Test 2: Fit & Predict (REAL XGBoost behavior)

```python
@pytest.mark.unit
def test_xgboost_strategy_fits_and_predicts_on_simple_dataset() -> None:
    """Verify XGBoost can fit and predict on linearly separable data."""
    # Arrange: Same dataset as LogReg test
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([
        [0.05, 0.05],
        [1.05, 1.05],
    ])

    config = {"random_state": 42, "n_estimators": 10}  # Small for speed
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)

    # Assert: XGBoost should learn this simple pattern
    assert predictions[0] == 0
    assert predictions[1] == 1


@pytest.mark.unit
def test_xgboost_strategy_handles_nonlinear_data() -> None:
    """Verify XGBoost learns non-linear decision boundary (XOR-like)."""
    # Arrange: XOR-like pattern (LogReg fails, XGBoost succeeds)
    X_train = np.array([
        [0, 0],  # Class 0
        [0, 1],  # Class 1
        [1, 0],  # Class 1
        [1, 1],  # Class 0
    ] * 25)  # Repeat for sufficient training data
    y_train = np.array([0, 1, 1, 0] * 25)

    X_test = np.array([
        [0.1, 0.1],  # Should predict 0
        [0.1, 0.9],  # Should predict 1
        [0.9, 0.1],  # Should predict 1
        [0.9, 0.9],  # Should predict 0
    ])

    config = {"random_state": 42, "n_estimators": 50, "max_depth": 3}
    strategy = XGBoostStrategy(config)

    # Act
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)

    # Assert: XGBoost should learn XOR pattern (at least 75% correct)
    accuracy = (predictions == np.array([0, 1, 1, 0])).mean()
    assert accuracy >= 0.75, f"XGBoost failed to learn XOR: {accuracy}"
```

**Philosophy:** Test REAL non-linear learning (XGBoost advantage over LogReg).

#### Test 3: Serialization (Native .xgb format)

```python
@pytest.mark.unit
def test_xgboost_strategy_save_and_load_model(tmp_path: Path) -> None:
    """Verify save_model() → load_model() gives same predictions."""
    # Arrange: Train model
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)

    config = {"random_state": 42, "n_estimators": 10}
    strategy = XGBoostStrategy(config)
    strategy.fit(X_train, y_train)

    original_preds = strategy.predict(X_test)
    original_probs = strategy.predict_proba(X_test)

    # Act: Save model
    xgb_path = tmp_path / "model.xgb"
    json_path = tmp_path / "config.json"

    strategy.save_model(str(xgb_path))
    config_dict = strategy.to_dict()
    with open(json_path, "w") as f:
        json.dump(config_dict, f)

    # Load model
    with open(json_path) as f:
        loaded_config = json.load(f)
    loaded_strategy = XGBoostStrategy.load_model(str(xgb_path), loaded_config)

    # Assert: Predictions match exactly
    loaded_preds = loaded_strategy.predict(X_test)
    loaded_probs = loaded_strategy.predict_proba(X_test)

    np.testing.assert_array_equal(loaded_preds, original_preds)
    np.testing.assert_allclose(loaded_probs, original_probs, rtol=1e-10)
```

**Philosophy:** Test REAL .xgb serialization (production format).

---

### Test File: `tests/unit/core/test_classifier_factory.py`

**Purpose:** Test factory pattern and registry

```python
@pytest.mark.unit
def test_create_classifier_defaults_to_logreg() -> None:
    """Verify factory defaults to LogReg if type not specified."""
    # Arrange
    config = {"random_state": 42}  # No "type" field

    # Act
    strategy = create_classifier(config)

    # Assert
    assert isinstance(strategy, LogisticRegressionStrategy)


@pytest.mark.unit
def test_create_classifier_creates_logreg() -> None:
    """Verify factory creates LogReg when type='logistic_regression'."""
    # Arrange
    config = {"type": "logistic_regression", "C": 0.5}

    # Act
    strategy = create_classifier(config)

    # Assert
    assert isinstance(strategy, LogisticRegressionStrategy)
    assert strategy.C == 0.5


@pytest.mark.unit
def test_create_classifier_creates_xgboost() -> None:
    """Verify factory creates XGBoost when type='xgboost'."""
    # Arrange
    config = {"type": "xgboost", "n_estimators": 50}

    # Act
    strategy = create_classifier(config)

    # Assert
    assert isinstance(strategy, XGBoostStrategy)
    assert strategy.n_estimators == 50


@pytest.mark.unit
def test_create_classifier_raises_on_unknown_type() -> None:
    """Verify factory raises ValueError for unknown classifier type."""
    # Arrange
    config = {"type": "random_forest"}  # Not implemented

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown classifier type: 'random_forest'"):
        create_classifier(config)
```

---

### Test File: `tests/unit/core/test_classifier.py` (MODIFIED)

**Purpose:** Test `BinaryClassifier` with strategy pattern

#### Test: Backward Compatibility

```python
@pytest.mark.unit
def test_binary_classifier_defaults_to_logreg(mock_transformers_model: Any) -> None:
    """Verify BinaryClassifier defaults to LogReg if type not specified."""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        # No "type" field - should default to logistic_regression
    }

    # Act
    classifier = BinaryClassifier(params)

    # Assert
    assert isinstance(classifier.classifier, LogisticRegressionStrategy)


@pytest.mark.unit
def test_binary_classifier_creates_xgboost(mock_transformers_model: Any) -> None:
    """Verify BinaryClassifier creates XGBoost strategy when type='xgboost'."""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "type": "xgboost",
        "n_estimators": 50,
    }

    # Act
    classifier = BinaryClassifier(params)

    # Assert
    assert isinstance(classifier.classifier, XGBoostStrategy)
    assert classifier.classifier.n_estimators == 50
```

#### Test: Fit & Predict with Real Embeddings

```python
@pytest.mark.unit
def test_binary_classifier_xgboost_fits_and_predicts(
    mock_transformers_model: Any,
) -> None:
    """Verify BinaryClassifier with XGBoost can fit and predict on embeddings."""
    # Arrange
    params = {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 1000,
        "type": "xgboost",
        "n_estimators": 10,
    }

    classifier = BinaryClassifier(params)

    # Create mock embeddings (realistic dimensions: ESM1v = 1280-dim)
    X_train = np.random.rand(50, 1280)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 1280)

    # Act
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    # Assert
    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 2)
    assert set(predictions).issubset({0, 1})
    assert classifier.is_fitted is True
```

**Philosophy:** Test REAL embeddings (correct dimensions), not toy data.

---

## Integration Tests

### Test File: `tests/integration/test_xgboost_training.py`

**Purpose:** Test full training workflow with XGBoost

```python
@pytest.mark.integration
def test_train_xgboost_on_boughter_subset(
    tmp_path: Path,
    mock_transformers_model: Any,
) -> None:
    """
    Test training XGBoost on Boughter dataset subset.

    Tests full workflow:
    1. Load data
    2. Extract embeddings (mocked ESM, but real embedding extractor)
    3. Train XGBoost classifier
    4. Evaluate on test set
    5. Save model (dual format)
    6. Load model and verify predictions match
    """
    # Arrange: Create small synthetic dataset (Boughter-like)
    # Use tmp_path for all file I/O
    train_csv = tmp_path / "train.csv"
    train_data = pd.DataFrame({
        "VH": ["EVQL" + "A" * 100, "EVQL" + "G" * 100] * 25,  # 50 sequences
        "ELISA_Plate_OD450": [0, 1] * 25,  # Binary labels
    })
    train_data.to_csv(train_csv, index=False)

    # Create config
    config = {
        "data": {
            "train_file": str(train_csv),
            "test_file": str(train_csv),  # Use same for simplicity
            "embeddings_cache_dir": str(tmp_path / "cache"),
            "sequence_column": "VH",
            "label_column": "ELISA_Plate_OD450",
        },
        "model": {
            "name": "facebook/esm1v_t33_650M_UR90S_1",
            "device": "cpu",
        },
        "classifier": {
            "type": "xgboost",
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
        },
        "training": {
            "save_model": True,
            "model_name": "test_xgboost",
            "model_save_dir": str(tmp_path / "models"),
            "log_level": "INFO",
            "log_file": str(tmp_path / "training.log"),
            "metrics": ["accuracy", "f1", "roc_auc"],
            "batch_size": 8,
        },
        "experiment": {
            "name": "test_xgboost_integration",
        },
    }

    cfg = OmegaConf.create(config)

    # Act: Train model
    results = train_pipeline(cfg)

    # Assert: Training succeeded
    assert "train_metrics" in results
    assert "model_paths" in results
    assert results["train_metrics"]["accuracy"] > 0.5  # Better than random

    # Assert: Model files created
    model_pickle = Path(results["model_paths"]["pickle"])
    assert model_pickle.exists()

    # Assert: Load model and predictions match
    with open(model_pickle, "rb") as f:
        loaded_classifier = pickle.load(f)

    # Get test embeddings (use cached)
    X_test, y_test = load_data(config)
    embedding_extractor = loaded_classifier.embedding_extractor
    X_test_embedded = embedding_extractor.extract_batch_embeddings(X_test)

    # Predictions should be deterministic
    preds1 = loaded_classifier.predict(X_test_embedded)
    preds2 = loaded_classifier.predict(X_test_embedded)
    np.testing.assert_array_equal(preds1, preds2)
```

**Philosophy:** Test REAL training pipeline, not mocked workflow.

---

### Test File: `tests/integration/test_cross_validation.py` (UPDATED)

```python
@pytest.mark.integration
def test_cross_validation_with_xgboost() -> None:
    """Verify cross-validation works with XGBoost classifier."""
    # Arrange: Create synthetic dataset
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 2, 100)

    config = {
        "classifier": {
            "type": "xgboost",
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "cv_folds": 5,
            "stratify": True,
        },
        "model": {
            "name": "facebook/esm1v_t33_650M_UR90S_1",
            "device": "cpu",
        },
        "training": {
            "batch_size": 8,
        },
    }

    # Act: Run cross-validation
    cv_results = perform_cross_validation(X, y, config, logger)

    # Assert: CV results structure
    assert "cv_accuracy" in cv_results
    assert "cv_f1" in cv_results
    assert "cv_roc_auc" in cv_results

    # Assert: Reasonable performance (>random)
    assert cv_results["cv_accuracy"]["mean"] > 0.4
    assert cv_results["cv_accuracy"]["std"] < 0.5
```

---

## End-to-End Tests

### Test File: `tests/e2e/test_xgboost_pipeline.py`

```python
@pytest.mark.e2e
def test_full_xgboost_pipeline(tmp_path: Path, mock_transformers_model: Any) -> None:
    """
    End-to-end test: Train XGBoost, save, load, predict.

    Full workflow:
    1. Train ESM1v + XGBoost on synthetic Boughter-like data
    2. Save model (pickle + .xgb + JSON)
    3. Load model from both formats
    4. Verify predictions match
    5. Test on synthetic Jain-like data
    """
    # Arrange: Create synthetic datasets
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    # Synthetic training data (50 sequences, 2 classes)
    train_data = pd.DataFrame({
        "VH": [f"EVQL{'A' * (100 + i)}" for i in range(50)],
        "label": [i % 2 for i in range(50)],
    })
    train_data.to_csv(train_csv, index=False)

    # Synthetic test data (10 sequences)
    test_data = pd.DataFrame({
        "VH": [f"QVQL{'G' * (100 + i)}" for i in range(10)],
        "label": [i % 2 for i in range(10)],
    })
    test_data.to_csv(test_csv, index=False)

    # Config
    config = {
        "data": {
            "train_file": str(train_csv),
            "test_file": str(test_csv),
            "embeddings_cache_dir": str(tmp_path / "cache"),
            "sequence_column": "VH",
            "label_column": "label",
        },
        "model": {
            "name": "facebook/esm1v_t33_650M_UR90S_1",
            "device": "cpu",
        },
        "classifier": {
            "type": "xgboost",
            "n_estimators": 20,
            "max_depth": 4,
            "random_state": 42,
        },
        "training": {
            "save_model": True,
            "model_name": "e2e_xgboost",
            "model_save_dir": str(tmp_path / "models"),
            "log_level": "INFO",
            "log_file": str(tmp_path / "training.log"),
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "batch_size": 8,
            "n_splits": 3,
        },
        "experiment": {
            "name": "e2e_xgboost_test",
        },
    }

    cfg = OmegaConf.create(config)

    # Act 1: Train
    results = train_pipeline(cfg)

    # Assert 1: Training succeeded
    assert results["train_metrics"]["accuracy"] > 0.0
    assert Path(results["model_paths"]["pickle"]).exists()

    # Act 2: Load from pickle
    with open(results["model_paths"]["pickle"], "rb") as f:
        loaded_classifier = pickle.load(f)

    # Act 3: Predict on test data
    X_test, y_test = load_data(config)
    X_test_embedded = loaded_classifier.embedding_extractor.extract_batch_embeddings(
        X_test
    )
    test_preds = loaded_classifier.predict(X_test_embedded)

    # Assert 2: Predictions are valid
    assert test_preds.shape == (10,)
    assert set(test_preds).issubset({0, 1})

    # Act 4: Test determinism (same inputs → same outputs)
    test_preds_2 = loaded_classifier.predict(X_test_embedded)
    np.testing.assert_array_equal(test_preds, test_preds_2)
```

**Philosophy:** Test REAL end-to-end workflow, not mocked pipeline.

---

## Performance Benchmarks

### Benchmark 1: Training Speed

```python
@pytest.mark.integration
def test_xgboost_training_speed_comparable_to_logreg() -> None:
    """Verify XGBoost training is ≤2× slower than LogReg."""
    import time

    # Arrange: Create dataset
    X = np.random.rand(500, 1280)  # ESM1v embedding dimension
    y = np.random.randint(0, 2, 500)

    logreg_config = {"type": "logistic_regression", "random_state": 42}
    xgboost_config = {
        "type": "xgboost",
        "n_estimators": 100,
        "random_state": 42,
    }

    # Act: Benchmark LogReg
    logreg_strategy = LogisticRegressionStrategy(logreg_config)
    start = time.time()
    logreg_strategy.fit(X, y)
    logreg_time = time.time() - start

    # Act: Benchmark XGBoost
    xgboost_strategy = XGBoostStrategy(xgboost_config)
    start = time.time()
    xgboost_strategy.fit(X, y)
    xgboost_time = time.time() - start

    # Assert: XGBoost is ≤2× slower
    assert xgboost_time <= logreg_time * 2, (
        f"XGBoost too slow: {xgboost_time:.2f}s vs LogReg {logreg_time:.2f}s"
    )
```

### Benchmark 2: Model Size

```python
@pytest.mark.integration
def test_xgboost_model_size_reasonable(tmp_path: Path) -> None:
    """Verify XGBoost model size ≤100MB."""
    # Arrange: Train model
    X = np.random.rand(500, 1280)
    y = np.random.randint(0, 2, 500)

    config = {"n_estimators": 100, "random_state": 42}
    strategy = XGBoostStrategy(config)
    strategy.fit(X, y)

    # Act: Save model
    xgb_path = tmp_path / "model.xgb"
    strategy.save_model(str(xgb_path))

    # Assert: Model size ≤100MB
    model_size_mb = xgb_path.stat().st_size / (1024 * 1024)
    assert model_size_mb <= 100, f"Model too large: {model_size_mb:.2f}MB"
```

---

## TDD Implementation Plan

### Phase 1: Refactoring (Test-Driven)

**Day 1-2: LogRegStrategy**

1. **Write tests FIRST** (`test_logistic_regression.py`):
   - test_logreg_strategy_initializes_with_defaults()
   - test_logreg_strategy_fits_and_predicts()
   - test_logreg_strategy_serialization_roundtrip()

2. **Run tests** → They FAIL (class doesn't exist)

3. **Implement `LogisticRegressionStrategy`** → Tests PASS

4. **Refactor `BinaryClassifier`** to use strategy → All existing tests STILL PASS

**Success Criteria:**
- ✅ All NEW tests pass (LogRegStrategy)
- ✅ All EXISTING tests pass (BinaryClassifier backward compat)
- ✅ 100% coverage for LogRegStrategy

### Phase 2: XGBoost Implementation (Test-Driven)

**Day 3-4: XGBoostStrategy**

1. **Write tests FIRST** (`test_xgboost_strategy.py`):
   - test_xgboost_strategy_initializes_with_defaults()
   - test_xgboost_strategy_fits_and_predicts()
   - test_xgboost_strategy_handles_nonlinear_data()
   - test_xgboost_strategy_save_and_load_model()

2. **Run tests** → They FAIL (XGBoostStrategy doesn't exist)

3. **Implement `XGBoostStrategy`** → Tests PASS

4. **Update factory** → Factory tests PASS

**Success Criteria:**
- ✅ All XGBoost tests pass
- ✅ 100% coverage for XGBoostStrategy
- ✅ Can run: `antibody-train classifier=xgboost`

### Phase 3: Integration Tests (Test-Driven)

**Day 5-6: Full Pipeline**

1. **Write integration tests FIRST**:
   - test_train_xgboost_on_boughter_subset()
   - test_cross_validation_with_xgboost()

2. **Run tests** → They FAIL (config/serialization issues)

3. **Fix issues** → Tests PASS

**Success Criteria:**
- ✅ Integration tests pass
- ✅ E2E test passes (full pipeline)
- ✅ All tests pass (unit + integration + e2e)

---

## Test Execution

### Local Development

```bash
# Run unit tests (fast)
uv run pytest tests/unit -v

# Run specific test file
uv run pytest tests/unit/core/strategies/test_xgboost_strategy.py -v

# Run integration tests
uv run pytest tests/integration -v

# Run e2e tests (expensive)
uv run pytest tests/e2e -v

# Run ALL tests
uv run pytest -v

# Coverage report
uv run pytest --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=100
```

### CI Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  test:
    - name: Run unit tests
      run: uv run pytest tests/unit --cov=src --cov-fail-under=100

    - name: Run integration tests
      run: uv run pytest tests/integration

    # E2E tests run on schedule (too expensive for every PR)
```

---

## Coverage Enforcement

### Fail CI if Coverage < 100% for New Code

```toml
# pyproject.toml
[tool.coverage.report]
fail_under = 100  # NEW CODE MUST HAVE 100% COVERAGE

exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## Summary

### Test Philosophy (Repeated for Emphasis)

**NO BOGUS TESTS:**
- Test BEHAVIORS, not implementation
- Minimal mocking (only ESM model loading)
- Real classifiers, real predictions, real file I/O

**YES TO REAL TESTS:**
- Test actual sklearn/xgboost behavior
- Test serialization round-trips with temp files
- Test full workflows (train → save → load → predict)
- Test performance (speed, model size)

### Coverage Targets

| Category | Files | Target | Status |
|----------|-------|--------|--------|
| Unit Tests | 5 new files | 100% | TBD |
| Integration Tests | 2 new files | 95%+ | TBD |
| E2E Tests | 1 new file | Full pipeline | TBD |

### Next Steps

1. **Implement Phase 1** (Refactoring) with TDD
2. **Implement Phase 2** (XGBoost) with TDD
3. **Run benchmarks** (train ESM1v+XGBoost, ESM2+XGBoost)
4. **Document results** in `docs/research/benchmark-results.md`

---

**Document Status:** Draft → Ready for Implementation
**Next Action:** Begin TDD Implementation (Phase 1: Refactoring)
**Review Required:** Yes
