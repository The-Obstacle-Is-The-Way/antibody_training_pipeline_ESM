# Test Suite Implementation Plan

**Status:** Planning (awaiting senior approval)
**Date:** 2025-11-07
**Author:** Claude Code
**Philosophy:** Robert C. Martin (Uncle Bob) Clean Code Principles

---

## Executive Summary

This document outlines a comprehensive test suite for the antibody non-specificity prediction pipeline. The current codebase has **~3,474 lines of production code** with **only 4 integration tests** focused on embedding compatibility. We will implement a professional test suite following test-driven development (TDD) principles retroactively.

**Current Coverage:** ~5% (integration tests only)
**Target Coverage:** 80%+ meaningful behavioral coverage
**No Bogus Tests:** Every test must verify real behavior, not implementation details

---

## Testing Philosophy (Uncle Bob / Clean Code)

### Core Principles

1. **Test Behaviors, Not Implementation**
   - ✅ Test WHAT the code does (contracts, interfaces, outcomes)
   - ❌ Don't test HOW it does it (private methods, internal state)
   - Example: Test that `classifier.predict(sequence)` returns 0 or 1, not that it calls LogisticRegression internally

2. **Minimize Mocking (No Bogus Mocks)**
   - ✅ Mock only I/O boundaries: network calls, file system (when necessary), external APIs
   - ✅ Mock heavyweight dependencies: ESM model downloads (~650MB), GPU operations
   - ❌ Don't mock domain logic, data transformations, or business rules
   - Example: Mock the ESM model loading, but don't mock DataFrame operations

3. **FIRST Principles**
   - **F**ast: Unit tests run in milliseconds (no model downloads, no disk I/O unless necessary)
   - **I**ndependent: Tests don't depend on each other, can run in any order
   - **R**epeatable: Same results every time, no flaky tests
   - **S**elf-validating: Pass/fail with clear assertions, no manual inspection
   - **T**imely: Written alongside (or retroactively for) production code

4. **Test Pyramid**
   ```
        /\
       /e2e\      (few, slow, brittle)
      /------\
     /integ. \    (some, medium speed)
    /----------\
   /   unit     \  (many, fast, focused)
   --------------
   ```
   - **70% Unit tests:** Fast, isolated, test single units of behavior
   - **20% Integration tests:** Test component interactions (datasets → embeddings → training)
   - **10% End-to-end tests:** Test full pipeline (CLI → model → results)

5. **Arrange-Act-Assert (AAA)**
   ```python
   def test_classifier_predicts_nonspecific_antibody():
       # Arrange: Set up test data
       sequence = "QVQLVQSGAEVKKPGASVKVSCKASGYTFT..."
       classifier = BinaryClassifier(params=TEST_PARAMS)

       # Act: Execute behavior
       prediction = classifier.predict([sequence])

       # Assert: Verify outcome
       assert prediction[0] in [0, 1]
   ```

6. **Single Responsibility (Tests Too)**
   - One test verifies one behavior
   - Test name describes the behavior: `test_classifier_rejects_sequence_with_gaps`
   - Test body is short (<20 lines), focused, readable

7. **DRY (Don't Repeat Yourself)**
   - Use pytest fixtures for shared setup
   - Extract common test data to conftest.py
   - Reuse test utilities (e.g., `create_mock_dataset()`)

---

## Current State Analysis

### Production Code Structure

```
src/antibody_training_esm/
├── cli/                    # 3 files - Command-line interfaces
│   ├── preprocess.py       # Dataset preprocessing CLI
│   ├── test.py            # Model testing CLI
│   └── train.py           # Model training CLI
├── core/                   # 3 files - Core ML logic
│   ├── classifier.py      # BinaryClassifier (sklearn-compatible)
│   ├── embeddings.py      # ESMEmbeddingExtractor
│   └── trainer.py         # Training pipeline
├── data/                   # 1 file - Data utilities
│   └── loaders.py         # Data loading helpers
├── datasets/               # 5 files - Dataset abstractions
│   ├── base.py            # AntibodyDataset ABC
│   ├── boughter.py        # Boughter dataset (training)
│   ├── harvey.py          # Harvey dataset (141k nanobodies)
│   ├── jain.py            # Jain dataset (86 clinical antibodies)
│   └── shehata.py         # Shehata dataset (398 PSR antibodies)
├── evaluation/             # Empty (no evaluation code yet)
└── utils/                  # Empty (no utils yet)

Total: ~3,474 lines of production code
```

### Current Test Structure

```
tests/
└── integration/            # 4 files - All embedding compatibility
    ├── test_boughter_embedding_compatibility.py
    ├── test_harvey_embedding_compatibility.py
    ├── test_jain_embedding_compatibility.py
    └── test_shehata_embedding_compatibility.py

Total: 4 integration tests (no unit tests)
```

**Gap Analysis:**
- ❌ No unit tests for core modules (classifier, embeddings, trainer)
- ❌ No tests for CLI commands
- ❌ No tests for dataset classes (only integration tests)
- ❌ No tests for data loaders
- ❌ No tests for edge cases (invalid sequences, missing files, etc.)
- ✅ Good integration tests for dataset embedding compatibility

---

## Proposed Test Structure

### Mirror `src/` Structure in `tests/`

```
tests/
├── conftest.py                    # Shared fixtures, test data, utilities
├── fixtures/                      # Test data (CSVs, mock sequences)
│   ├── mock_sequences.py          # Sample antibody sequences
│   ├── mock_datasets/             # Small CSV files for fast tests
│   │   ├── boughter_sample.csv
│   │   ├── jain_sample.csv
│   │   └── shehata_sample.csv
│   └── mock_models.py             # Mock ESM model for tests
│
├── unit/                          # Unit tests (70% of tests)
│   ├── cli/
│   │   ├── test_preprocess.py     # Preprocessing CLI behavior
│   │   ├── test_test.py           # Testing CLI behavior
│   │   └── test_train.py          # Training CLI behavior
│   ├── core/
│   │   ├── test_classifier.py     # BinaryClassifier behavior
│   │   ├── test_embeddings.py     # ESMEmbeddingExtractor behavior
│   │   └── test_trainer.py        # Training pipeline behavior
│   ├── data/
│   │   └── test_loaders.py        # Data loading utilities
│   └── datasets/
│       ├── test_base.py           # AntibodyDataset ABC contract
│       ├── test_boughter.py       # Boughter dataset behavior
│       ├── test_harvey.py         # Harvey dataset behavior
│       ├── test_jain.py           # Jain dataset behavior
│       └── test_shehata.py        # Shehata dataset behavior
│
├── integration/                   # Integration tests (20% of tests)
│   ├── test_boughter_embedding_compatibility.py  # (existing)
│   ├── test_harvey_embedding_compatibility.py    # (existing)
│   ├── test_jain_embedding_compatibility.py      # (existing)
│   ├── test_shehata_embedding_compatibility.py   # (existing)
│   ├── test_dataset_pipeline.py   # Dataset → Embedding → Training
│   ├── test_cross_validation.py   # Full CV pipeline
│   └── test_model_persistence.py  # Save/load model workflow
│
└── e2e/                           # End-to-end tests (10% of tests)
    ├── test_train_pipeline.py     # Full training pipeline (CLI)
    ├── test_predict_pipeline.py   # Full prediction pipeline
    └── test_reproduce_novo.py     # Reproduce Novo Nordisk results
```

---

## Test Categories and Examples

### 1. Unit Tests (`tests/unit/`)

**Purpose:** Test single units of behavior in isolation (fast, no I/O)

#### `tests/unit/core/test_classifier.py`

**What to Test:**
- ✅ sklearn API compatibility (`get_params`, `set_params`)
- ✅ Fit/predict behavior with mock embeddings
- ✅ Binary classification output (0 or 1)
- ✅ Probability calibration for PSR vs ELISA assays
- ✅ Edge cases: empty sequences, single sequence, large batches

**Example Tests:**
```python
def test_classifier_implements_sklearn_api():
    """Verify BinaryClassifier implements sklearn estimator interface"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)

    # Act
    params = classifier.get_params()

    # Assert
    assert "C" in params
    assert "penalty" in params
    assert "random_state" in params

def test_classifier_predicts_binary_labels():
    """Verify predictions are binary (0 or 1)"""
    # Arrange
    X_train = np.random.rand(100, 1280)  # Mock embeddings
    y_train = np.array([0, 1] * 50)
    classifier = BinaryClassifier(params=TEST_PARAMS)

    # Act
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_train[:10])

    # Assert
    assert all(pred in [0, 1] for pred in predictions)

def test_classifier_applies_psr_threshold_calibration():
    """Verify PSR assay uses 0.4 decision threshold"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.is_fitted = True
    classifier.classifier.predict_proba = lambda X: np.array([[0.3, 0.7]])

    # Act
    prediction = classifier.predict(np.zeros((1, 1280)), assay_type="PSR")

    # Assert
    assert prediction[0] == 1  # 0.7 > 0.4 threshold

def test_classifier_rejects_invalid_sequences():
    """Verify classifier validates sequence format"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)
    invalid_sequence = "QVQL-VQSG"  # Contains gap

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid amino acid"):
        classifier.predict([invalid_sequence])
```

**Mocking Strategy:**
- Mock `ESMEmbeddingExtractor.extract_embeddings()` to return fake embeddings (np.random.rand)
- Don't mock LogisticRegression (it's lightweight, part of the contract)
- Don't mock DataFrame operations (part of domain logic)

---

#### `tests/unit/datasets/test_jain.py`

**What to Test:**
- ✅ Dataset loading from CSV
- ✅ Filtering strategies (full, ssot, parity stages)
- ✅ Fragment extraction (VH, VL, CDRs, etc.)
- ✅ Label assignment (threshold-based)
- ✅ Edge cases: missing files, corrupted CSV, missing columns

**Example Tests:**
```python
def test_jain_dataset_loads_full_stage():
    """Verify Jain dataset loads all 137 antibodies in 'full' stage"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load(stage="full")

    # Assert
    assert len(df) == 137
    assert "sequence" in df.columns
    assert "label" in df.columns

def test_jain_dataset_parity_stage_returns_86_antibodies():
    """Verify 'parity' stage matches Novo Nordisk (86 antibodies)"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load(stage="parity")

    # Assert
    assert len(df) == 86

def test_jain_dataset_excludes_mild_antibodies():
    """Verify 1-3 ELISA flags are excluded from parity set"""
    # Arrange
    dataset = JainDataset()

    # Act
    df_full = dataset.load(stage="full")
    df_parity = dataset.load(stage="parity")

    # Assert
    # Mild antibodies (1-3 flags) should be filtered out
    assert len(df_parity) < len(df_full)
    # Parity set should only have 0 flags (specific) or >=4 flags (non-specific)
    # (This assumes internal flag column exists - adapt to actual implementation)

def test_jain_dataset_validates_required_columns():
    """Verify dataset raises error if required columns missing"""
    # Arrange
    dataset = JainDataset()
    # Mock a corrupted CSV (missing 'sequence' column)

    # Act & Assert
    with pytest.raises(ValueError, match="Missing required column"):
        dataset.load(stage="full")
```

**Mocking Strategy:**
- Use small mock CSV files in `tests/fixtures/mock_datasets/` (10-20 rows)
- Mock file I/O only for error cases (missing files)
- Don't mock pandas operations (part of dataset logic)

---

#### `tests/unit/core/test_embeddings.py`

**What to Test:**
- ✅ Model initialization (device selection, batch size)
- ✅ Embedding extraction (sequence → 1280-dim vector)
- ✅ Batch processing
- ✅ Sequence validation (gaps, invalid amino acids)
- ✅ Edge cases: empty sequences, very long sequences

**Example Tests:**
```python
@pytest.fixture
def mock_esm_model():
    """Mock ESM model to avoid downloading 650MB model"""
    class MockESMModel:
        def __call__(self, tokens):
            # Return fake embeddings (batch_size, seq_len, 1280)
            batch_size = tokens.shape[0]
            return {"representations": {33: torch.rand(batch_size, 128, 1280)}}

    return MockESMModel()

def test_embeddings_extracts_1280_dim_vectors(mock_esm_model, monkeypatch):
    """Verify embeddings are 1280-dimensional (ESM-1v representation size)"""
    # Arrange
    monkeypatch.setattr("esm.pretrained.esm1v_t33_650M_UR90S_1", lambda: (mock_esm_model, None))
    extractor = ESMEmbeddingExtractor(model_name="esm1v_t33_650M_UR90S_1", device="cpu")

    # Act
    embeddings = extractor.extract_embeddings(["QVQLVQSGAEVKKPGA"])

    # Assert
    assert embeddings.shape == (1, 1280)

def test_embeddings_processes_batches(mock_esm_model, monkeypatch):
    """Verify batch processing handles multiple sequences"""
    # Arrange
    monkeypatch.setattr("esm.pretrained.esm1v_t33_650M_UR90S_1", lambda: (mock_esm_model, None))
    extractor = ESMEmbeddingExtractor(model_name="esm1v_t33_650M_UR90S_1", device="cpu", batch_size=2)
    sequences = ["QVQLVQSGAEVKKPGA"] * 10

    # Act
    embeddings = extractor.extract_embeddings(sequences)

    # Assert
    assert embeddings.shape == (10, 1280)

def test_embeddings_rejects_sequences_with_gaps(mock_esm_model, monkeypatch):
    """Verify sequences with gaps are rejected"""
    # Arrange
    monkeypatch.setattr("esm.pretrained.esm1v_t33_650M_UR90S_1", lambda: (mock_esm_model, None))
    extractor = ESMEmbeddingExtractor(model_name="esm1v_t33_650M_UR90S_1", device="cpu")

    # Act & Assert
    with pytest.raises(ValueError, match="gap character"):
        extractor.extract_embeddings(["QVQL-VQSG"])
```

**Mocking Strategy:**
- ✅ Mock ESM model loading (`esm.pretrained.esm1v_t33_650M_UR90S_1`)
- ✅ Return fake torch tensors for embeddings
- ❌ Don't mock sequence validation (that's the behavior we're testing!)

---

### 2. Integration Tests (`tests/integration/`)

**Purpose:** Test component interactions (medium speed, some I/O)

#### `tests/integration/test_dataset_pipeline.py`

**What to Test:**
- ✅ Dataset loading → Embedding extraction → Training
- ✅ Cross-dataset compatibility (Boughter → Jain testing)
- ✅ Fragment-specific workflows (VH only, CDRs only, etc.)

**Example Tests:**
```python
def test_boughter_to_jain_pipeline():
    """Verify Boughter training set can predict on Jain test set"""
    # Arrange: Load Boughter training data
    boughter = BoughterDataset()
    df_train = boughter.load(fragment="VH_only", include_mild=False)

    # Arrange: Load Jain test data
    jain = JainDataset()
    df_test = jain.load(stage="parity", fragment="VH_only")

    # Act: Train classifier (with mocked embeddings to speed up test)
    classifier = BinaryClassifier(params=TEST_PARAMS)
    X_train, y_train = mock_embed_sequences(df_train["sequence"], df_train["label"])
    classifier.fit(X_train, y_train)

    # Act: Predict on Jain
    X_test, y_test = mock_embed_sequences(df_test["sequence"], df_test["label"])
    predictions = classifier.predict(X_test)

    # Assert: Predictions are valid
    assert len(predictions) == len(df_test)
    assert all(pred in [0, 1] for pred in predictions)
    # Don't assert exact accuracy (that's fragile), just verify pipeline works

def test_fragment_compatibility():
    """Verify all 16 fragment types can be loaded and embedded"""
    # Arrange
    fragments = ["VH_only", "VL_only", "H-CDR3", "Full", ...]  # All 16 fragments
    dataset = BoughterDataset()

    # Act & Assert
    for fragment in fragments:
        df = dataset.load(fragment=fragment)
        assert len(df) > 0
        assert "sequence" in df.columns
        # Verify sequences are valid for embedding (no gaps)
        assert not df["sequence"].str.contains("-").any()
```

---

### 3. End-to-End Tests (`tests/e2e/`)

**Purpose:** Test full user workflows (slow, realistic)

#### `tests/e2e/test_train_pipeline.py`

**What to Test:**
- ✅ CLI commands execute successfully
- ✅ Full training pipeline produces model file
- ✅ Model can be loaded and used for predictions

**Example Tests:**
```python
def test_train_command_creates_model_file(tmp_path):
    """Verify 'antibody-train' CLI creates a trained model"""
    # Arrange
    output_model = tmp_path / "test_model.pkl"

    # Act: Run CLI command
    result = subprocess.run(
        ["antibody-train", "--dataset", "boughter", "--output", str(output_model)],
        capture_output=True,
        text=True
    )

    # Assert
    assert result.returncode == 0
    assert output_model.exists()
    # Verify model can be loaded
    with open(output_model, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict")

def test_reproduce_novo_nordisk_results():
    """Verify we can reproduce Novo Nordisk Jain accuracy (66-71%)"""
    # This test uses REAL data and REAL model (slow, but critical)
    # Arrange: Train on Boughter
    # Act: Test on Jain
    # Assert: Accuracy within 60-75% range (allow some variance)
    # (Implementation details depend on final testing strategy)
    pass  # TODO: Implement when senior approves
```

---

## Mocking Strategy

### What to Mock (✅ Allowed)

1. **ESM Model Loading**
   - Mock `esm.pretrained.esm1v_t33_650M_UR90S_1()` to avoid 650MB download
   - Return fake torch tensors for embeddings

2. **File I/O (Selectively)**
   - Mock missing files for error handling tests
   - Use small mock CSV files for fast unit tests

3. **External APIs / Network Calls** (if added)
   - Mock HuggingFace API calls
   - Mock any web requests

4. **GPU Operations** (if applicable)
   - Mock CUDA availability checks
   - Use CPU for all tests

### What NOT to Mock (❌ Forbidden)

1. **Domain Logic**
   - Don't mock pandas operations (filtering, groupby, merging)
   - Don't mock sklearn LogisticRegression (part of the contract)
   - Don't mock dataset transformations (that's what we're testing!)

2. **Data Transformations**
   - Don't mock sequence validation
   - Don't mock fragment extraction
   - Don't mock label assignment

3. **Business Rules**
   - Don't mock threshold logic (PSR 0.4, ELISA 0.5)
   - Don't mock flagging strategies (0 vs 1-3 vs 4+)

**Principle:** Mock I/O boundaries, test behavior everywhere else.

---

## Test Data Strategy

### Fixtures (`tests/fixtures/`)

1. **Mock Sequences** (`mock_sequences.py`)
   ```python
   # Valid antibody sequences
   VALID_VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS"
   VALID_VL = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"

   # Invalid sequences (for error testing)
   SEQUENCE_WITH_GAP = "QVQL-VQSGAEVKKPGA"
   SEQUENCE_WITH_INVALID_AA = "QVQLVQSGAEVKKPGABBB"  # 'B' is invalid
   ```

2. **Mock Datasets** (`mock_datasets/`)
   - Small CSV files (10-20 rows) for fast unit tests
   - Covers edge cases: balanced labels, imbalanced labels, single class, etc.

3. **Mock Model** (`mock_models.py`)
   ```python
   class MockESMModel:
       """Mock ESM model for tests (no 650MB download)"""
       def __call__(self, tokens):
           batch_size = tokens.shape[0]
           seq_len = tokens.shape[1]
           return {"representations": {33: torch.rand(batch_size, seq_len, 1280)}}
   ```

### Test Database
- No real database needed (data is CSV-based)
- Use in-memory pandas DataFrames for unit tests
- Use small mock CSVs for integration tests

---

## Pytest Configuration

### `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test execution
addopts =
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    -ra

# Coverage
addopts +=
    --cov=src/antibody_training_esm
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# Markers
markers =
    unit: Unit tests (fast, no I/O)
    integration: Integration tests (medium speed, some I/O)
    e2e: End-to-end tests (slow, full pipeline)
    slow: Tests that take >1s to run
    gpu: Tests that require GPU (skip in CI)

# Disable warnings from dependencies
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest -m unit

# Run with coverage report
pytest --cov=src/antibody_training_esm --cov-report=html

# Run specific test file
pytest tests/unit/core/test_classifier.py

# Run specific test
pytest tests/unit/core/test_classifier.py::test_classifier_predicts_binary_labels

# Skip slow tests (for quick feedback)
pytest -m "not slow"

# Run in parallel (fast)
pytest -n auto
```

---

## Coverage Targets

### By Module

| Module | Target Coverage | Priority | Rationale |
|--------|----------------|----------|-----------|
| `core/classifier.py` | 90%+ | High | Critical ML logic, sklearn API |
| `core/embeddings.py` | 85%+ | High | ESM integration, sequence validation |
| `core/trainer.py` | 85%+ | High | Training pipeline, CV logic |
| `datasets/*.py` | 80%+ | High | Data loading, filtering, fragment extraction |
| `data/loaders.py` | 80%+ | Medium | Data utilities |
| `cli/*.py` | 70%+ | Medium | CLI commands (harder to test) |
| `evaluation/` | N/A | Low | Empty module |
| `utils/` | N/A | Low | Empty module |

### Coverage Metrics

- **Line Coverage:** 80%+ (meaningful lines, not imports)
- **Branch Coverage:** 70%+ (all major code paths)
- **Function Coverage:** 90%+ (every public function tested)

**What NOT to Cover:**
- `__init__.py` files (just imports)
- Private methods (test through public API)
- Deprecated code (remove it instead)
- Debug print statements (remove them)

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal:** Set up test infrastructure and core unit tests

1. ✅ Create test directory structure
2. ✅ Write `conftest.py` with shared fixtures
3. ✅ Create mock datasets in `tests/fixtures/`
4. ✅ Write `tests/unit/core/test_classifier.py` (30+ tests)
5. ✅ Write `tests/unit/core/test_embeddings.py` (20+ tests)
6. ✅ Configure pytest.ini, coverage reporting

**Deliverable:** Core modules have 80%+ coverage

### Phase 2: Datasets (Week 2)
**Goal:** Test all dataset classes

1. ✅ Write `tests/unit/datasets/test_base.py`
2. ✅ Write `tests/unit/datasets/test_jain.py`
3. ✅ Write `tests/unit/datasets/test_boughter.py`
4. ✅ Write `tests/unit/datasets/test_harvey.py`
5. ✅ Write `tests/unit/datasets/test_shehata.py`

**Deliverable:** Dataset modules have 80%+ coverage

### Phase 3: Integration (Week 3)
**Goal:** Test component interactions

1. ✅ Write `tests/integration/test_dataset_pipeline.py`
2. ✅ Write `tests/integration/test_cross_validation.py`
3. ✅ Write `tests/integration/test_model_persistence.py`
4. ✅ Keep existing embedding compatibility tests

**Deliverable:** Integration tests verify end-to-end workflows

### Phase 4: CLI & E2E (Week 4)
**Goal:** Test user-facing workflows

1. ✅ Write `tests/unit/cli/test_train.py`
2. ✅ Write `tests/unit/cli/test_test.py`
3. ✅ Write `tests/unit/cli/test_preprocess.py`
4. ✅ Write `tests/e2e/test_train_pipeline.py`
5. ✅ Write `tests/e2e/test_reproduce_novo.py`

**Deliverable:** Full pipeline tested from CLI to results

---

## Continuous Integration (CI)

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install uv
        uv sync

    - name: Run unit tests
      run: pytest -m unit --cov=src/antibody_training_esm

    - name: Run integration tests
      run: pytest -m integration

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

**CI Mocking Strategy:**
- ✅ Mock ESM model loading (no 650MB download in CI)
- ✅ Use CPU-only tests (no GPU in CI)
- ✅ Use small mock datasets (fast CI runs)

---

## Examples of Good vs Bogus Tests

### ✅ Good Test (Tests Behavior)

```python
def test_classifier_handles_empty_sequence_list():
    """Verify classifier raises ValueError for empty input"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.is_fitted = True

    # Act & Assert
    with pytest.raises(ValueError, match="empty"):
        classifier.predict([])
```

**Why it's good:**
- Tests observable behavior (raises ValueError)
- Tests edge case (empty input)
- Clear test name describes behavior
- Single assertion, focused

### ❌ Bogus Test (Tests Implementation)

```python
def test_classifier_calls_logistic_regression():
    """Verify classifier uses LogisticRegression internally"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)

    # Act
    classifier.fit(X, y)

    # Assert
    assert isinstance(classifier.classifier, LogisticRegression)
```

**Why it's bogus:**
- Tests implementation detail (LogisticRegression)
- Breaks if we refactor to use a different model
- Doesn't verify actual behavior (predictions work)
- Couples test to internal structure

### ✅ Good Test (Minimal Mock)

```python
def test_embeddings_validates_sequences_before_extraction(mock_esm_model, monkeypatch):
    """Verify invalid sequences are rejected before embedding"""
    # Arrange
    monkeypatch.setattr("esm.pretrained.esm1v_t33_650M_UR90S_1", lambda: (mock_esm_model, None))
    extractor = ESMEmbeddingExtractor(model_name="esm1v_t33_650M_UR90S_1", device="cpu")

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.extract_embeddings(["QVQLVQSG-AEVKKPGA"])  # Gap character
```

**Why it's good:**
- Mocks only I/O boundary (ESM model loading)
- Tests domain logic (sequence validation)
- Verifies error handling behavior

### ❌ Bogus Test (Over-mocked)

```python
def test_embeddings_processes_sequences(mocker):
    """Verify embeddings are extracted"""
    # Arrange
    mock_extractor = mocker.Mock()
    mock_extractor.extract_embeddings.return_value = np.zeros((1, 1280))

    # Act
    result = mock_extractor.extract_embeddings(["QVQLVQSG"])

    # Assert
    assert result.shape == (1, 1280)
```

**Why it's bogus:**
- Mocks the thing we're testing (ESMEmbeddingExtractor)
- Test always passes (mock returns what we tell it to)
- Doesn't test any real behavior

---

## Success Criteria

### Definition of Done

- [ ] Test coverage ≥80% (line coverage)
- [ ] All public functions have unit tests
- [ ] All edge cases tested (empty inputs, invalid data, etc.)
- [ ] Integration tests verify cross-module workflows
- [ ] E2E test reproduces Novo Nordisk results (66-71% Jain accuracy)
- [ ] No flaky tests (100% reproducible)
- [ ] CI pipeline runs tests on every commit
- [ ] Test suite runs in <5 minutes (unit + integration)
- [ ] Documentation updated with testing guidelines

### Quality Gates

**Before merging any code:**
1. ✅ All tests pass
2. ✅ Coverage doesn't decrease
3. ✅ No new flaky tests introduced
4. ✅ Linting passes (ruff)
5. ✅ Type checking passes (mypy)

---

## FAQ

### Q: Why not test private methods?

**A:** Private methods are implementation details. Test them indirectly through public APIs. If a private method is complex enough to need testing, it might deserve to be a separate class.

### Q: When is it OK to mock?

**A:** Mock only I/O boundaries:
- ✅ ESM model loading (heavy, slow)
- ✅ Network requests (unreliable in tests)
- ✅ File system (for error cases)
- ❌ Not domain logic, not transformations, not business rules

### Q: How do I test code that depends on large datasets?

**A:** Use small mock datasets in `tests/fixtures/`. 10-20 rows is enough to test logic. Integration tests can use real data (but keep them few and fast).

### Q: Should I test for 100% coverage?

**A:** No. 80%+ meaningful coverage is the target. Focus on:
- Critical paths (training, prediction)
- Edge cases (empty inputs, invalid data)
- Error handling
- Don't waste time testing trivial getters/setters.

### Q: What if a test is too slow?

**A:**
1. Mark it with `@pytest.mark.slow`
2. Mock heavyweight dependencies (ESM model)
3. Use smaller test data
4. Move to integration/e2e tests (run less frequently)

---

## References

- **Robert C. Martin (Uncle Bob):** *Clean Code*, *Clean Architecture*
- **Martin Fowler:** *Refactoring*, *Patterns of Enterprise Application Architecture*
- **pytest documentation:** https://docs.pytest.org/
- **Test-Driven Development:** Kent Beck, *Test Driven Development: By Example*

---

**Next Steps:**
1. Senior review and approval of this plan
2. Implement Phase 1 (Foundation)
3. Iteratively build out test suite
4. Maintain 80%+ coverage as code evolves

**Questions for Senior Review:**
- Are the coverage targets reasonable?
- Any additional test categories needed?
- Mock strategy approved (ESM model, no domain logic)?
- CI configuration looks good?

---

**Author:** Claude Code
**Date:** 2025-11-07
**Status:** Awaiting Senior Approval
