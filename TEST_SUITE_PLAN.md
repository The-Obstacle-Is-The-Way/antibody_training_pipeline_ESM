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
   - Example: Test that `classifier.predict(embeddings)` returns 0 or 1, not that it calls LogisticRegression internally

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
       # Arrange: Set up test data (embeddings, not sequences)
       embeddings = np.random.rand(1, 1280)  # Mock ESM embedding
       classifier = BinaryClassifier(params=TEST_PARAMS)
       classifier.fit(np.random.rand(100, 1280), np.array([0, 1] * 50))  # Pre-fit

       # Act: Execute behavior
       prediction = classifier.predict(embeddings)

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
    """Verify PSR assay uses 0.5495 decision threshold (Novo parity value)"""
    # Arrange
    X_train = np.random.rand(100, 1280)
    y_train = np.array([0, 1] * 50)
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.fit(X_train, y_train)

    # Mock predict_proba to return known probability
    classifier.classifier.predict_proba = lambda X: np.array([[0.45, 0.55]])

    # Act
    prediction = classifier.predict(np.zeros((1, 1280)), assay_type="PSR")

    # Assert
    assert prediction[0] == 1  # 0.55 > 0.5495 threshold

def test_classifier_requires_fit_before_predict():
    """Verify classifier raises error when predicting before fit"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)
    embeddings = np.random.rand(10, 1280)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted"):
        classifier.predict(embeddings)

def test_classifier_handles_single_sample():
    """Verify classifier handles single embedding (edge case)"""
    # Arrange
    X_train = np.random.rand(100, 1280)
    y_train = np.array([0, 1] * 50)
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.fit(X_train, y_train)

    # Act
    prediction = classifier.predict(np.random.rand(1, 1280))

    # Assert
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
```

**Mocking Strategy:**
- Don't mock LogisticRegression (it's lightweight, part of the contract)
- Don't mock DataFrame operations (part of domain logic)
- Sequence validation happens in ESMEmbeddingExtractor, not classifier

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
    df = dataset.load_data(stage="full")

    # Assert
    assert len(df) == 137
    assert "VH_sequence" in df.columns  # Dataset returns VH_sequence, not sequence
    assert "VL_sequence" in df.columns  # Also includes VL_sequence
    assert "label" in df.columns

def test_jain_dataset_parity_stage_returns_86_antibodies():
    """Verify 'parity' stage matches Novo Nordisk (86 antibodies)"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(stage="parity")

    # Assert
    assert len(df) == 86

def test_jain_dataset_excludes_mild_antibodies():
    """Verify 1-3 ELISA flags are excluded from parity set"""
    # Arrange
    dataset = JainDataset()

    # Act
    df_full = dataset.load_data(stage="full")
    df_parity = dataset.load_data(stage="parity")

    # Assert
    # Mild antibodies (1-3 flags) should be filtered out in parity stage
    assert len(df_parity) < len(df_full)
    assert len(df_full) == 137  # Full dataset
    assert len(df_parity) == 86  # Parity set (excludes mild + low-confidence)

def test_jain_dataset_creates_fragment_csvs():
    """Verify dataset can create all 16 fragment types"""
    # Arrange
    dataset = JainDataset()
    df = dataset.load_data(stage="full")
    output_dir = Path("test_output/jain_fragments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Act
    dataset.create_fragment_csvs(df, output_dir=str(output_dir))

    # Assert
    expected_fragments = dataset.get_supported_fragments()
    assert len(expected_fragments) == 16
    for fragment in expected_fragments:
        fragment_file = output_dir / f"{fragment}_jain.csv"
        assert fragment_file.exists(), f"Fragment {fragment} not created"

def test_jain_dataset_handles_missing_file():
    """Verify dataset raises error for missing input files"""
    # Arrange
    dataset = JainDataset()

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        dataset.load_data(full_csv_path="nonexistent.csv")
```

**Mocking Strategy:**
- Use small mock CSV files in `tests/fixtures/mock_datasets/` (10-20 rows)
- Mock file I/O only for error cases (missing files)
- Don't mock pandas operations (part of dataset logic)

---

#### `tests/unit/core/test_embeddings.py`

**What to Test:**
- ✅ Model initialization (device selection, batch size)
- ✅ Single sequence embedding (`embed_sequence`)
- ✅ Batch embedding extraction (`extract_batch_embeddings`)
- ✅ Sequence validation (gaps, invalid amino acids)
- ✅ Edge cases: empty sequences, very long sequences, invalid characters

**Example Tests:**
```python
@pytest.fixture
def mock_transformers_model(monkeypatch):
    """Mock Hugging Face transformers model to avoid downloading 650MB"""
    class MockESMModel:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # Mock hidden states (last layer)
            hidden_states = torch.rand(batch_size, seq_len, 1280)
            return type('obj', (object,), {'hidden_states': (None, hidden_states)})()

    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, sequences, **kwargs):
            if isinstance(sequences, str):
                sequences = [sequences]
            batch_size = len(sequences)
            max_len = max(len(s) for s in sequences) + 2  # +2 for CLS/EOS
            return {
                "input_ids": torch.randint(0, 100, (batch_size, max_len)),
                "attention_mask": torch.ones(batch_size, max_len)
            }

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", MockESMModel)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", MockTokenizer)
    return MockESMModel, MockTokenizer

def test_embed_sequence_extracts_1280_dim_vector(mock_transformers_model):
    """Verify single sequence embedding returns 1280-dimensional vector"""
    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )

    # Act
    embedding = extractor.embed_sequence("QVQLVQSGAEVKKPGA")

    # Assert
    assert embedding.shape == (1280,)  # Single vector
    assert isinstance(embedding, np.ndarray)

def test_extract_batch_embeddings_handles_multiple_sequences(mock_transformers_model):
    """Verify batch processing returns correct shape"""
    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        batch_size=2
    )
    sequences = ["QVQLVQSGAEVKKPGA"] * 10

    # Act
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Assert
    assert embeddings.shape == (10, 1280)  # (n_sequences, embedding_dim)

def test_embed_sequence_rejects_invalid_amino_acids():
    """Verify embed_sequence raises ValueError for invalid sequences"""
    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )

    # Act & Assert - embed_sequence raises on invalid input
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.embed_sequence("QVQL-VQSG")  # Gap character

def test_extract_batch_embeddings_handles_invalid_sequences_gracefully(mock_transformers_model):
    """Verify batch extractor logs warning and uses placeholder for invalid sequences"""
    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )
    sequences = ["QVQLVQSG", "INVALID-SEQ", "QVQLVQSG"]  # Middle one has gap

    # Act - extract_batch_embeddings uses placeholder "M" instead of raising
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Assert - Still returns embeddings (placeholder used for invalid)
    assert embeddings.shape == (3, 1280)
```

**Mocking Strategy:**
- ✅ Mock `transformers.AutoModel.from_pretrained()` (uses HuggingFace, not esm.pretrained)
- ✅ Mock `transformers.AutoTokenizer.from_pretrained()`
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
def test_boughter_to_jain_pipeline(mock_transformers_model):
    """Verify Boughter training set can predict on Jain test set"""
    # Arrange: Load Boughter training data (VH only)
    boughter = BoughterDataset()
    df_train = boughter.load_data(include_mild=False)  # Returns VH_sequence, VL_sequence

    # Arrange: Load Jain test data
    jain = JainDataset()
    df_test = jain.load_data(stage="parity")  # Returns VH_sequence, VL_sequence

    # Arrange: Extract embeddings from VH sequences
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )
    X_train = extractor.extract_batch_embeddings(df_train["VH_sequence"].tolist())
    y_train = df_train["label"].values

    X_test = extractor.extract_batch_embeddings(df_test["VH_sequence"].tolist())

    # Act: Train classifier
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.fit(X_train, y_train)

    # Act: Predict on Jain
    predictions = classifier.predict(X_test)

    # Assert: Predictions are valid
    assert len(predictions) == len(df_test)
    assert all(pred in [0, 1] for pred in predictions)
    # Don't assert exact accuracy (that's fragile), just verify pipeline works

def test_fragment_csv_creation():
    """Verify all 16 fragment types can be created from full dataset"""
    # Arrange
    dataset = BoughterDataset()
    df = dataset.load_data(include_mild=False)
    output_dir = Path("test_output/boughter_fragments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Act: Create fragment CSVs
    dataset.create_fragment_csvs(df, output_dir=str(output_dir))

    # Assert: All 16 fragments exist
    expected_fragments = dataset.get_supported_fragments()
    assert len(expected_fragments) == 16
    for fragment in expected_fragments:
        fragment_file = output_dir / f"{fragment}_boughter.csv"
        assert fragment_file.exists(), f"Fragment {fragment} not created"

        # Verify fragment CSV has valid sequences (no gaps)
        df_fragment = pd.read_csv(fragment_file, comment="#")
        assert "sequence" in df_fragment.columns
        assert not df_fragment["sequence"].str.contains("-").any()
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
   - Mock `transformers.AutoModel.from_pretrained()` to avoid 650MB download
   - Mock `transformers.AutoTokenizer.from_pretrained()`
   - Return fake torch tensors for embeddings
   - Model: `facebook/esm1v_t33_650M_UR90S_1`

2. **File I/O (Selectively)**
   - Mock missing files for error handling tests
   - Use small mock CSV files for fast unit tests

3. **External APIs / Network Calls** (if added)
   - Mock HuggingFace API calls (model downloads)
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
   - Don't mock threshold logic (PSR 0.5495, ELISA 0.5)
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
       """Mock HuggingFace transformers ESM model (no 650MB download)"""
       def __init__(self, *args, **kwargs):
           pass

       def to(self, device):
           return self

       def eval(self):
           return self

       def __call__(self, input_ids, attention_mask, output_hidden_states=False):
           batch_size = input_ids.shape[0]
           seq_len = input_ids.shape[1]
           hidden_states = torch.rand(batch_size, seq_len, 1280)
           return type('obj', (object,), {'hidden_states': (None, hidden_states)})()

   class MockTokenizer:
       """Mock HuggingFace tokenizer"""
       def __init__(self, *args, **kwargs):
           pass

       def __call__(self, sequences, **kwargs):
           if isinstance(sequences, str):
               sequences = [sequences]
           batch_size = len(sequences)
           max_len = max(len(s) for s in sequences) + 2  # +2 for CLS/EOS
           return {
               "input_ids": torch.randint(0, 100, (batch_size, max_len)),
               "attention_mask": torch.ones(batch_size, max_len)
           }
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
- ✅ Mock transformers model loading (no 650MB ESM download in CI)
- ✅ Use CPU-only tests (no GPU in CI)
- ✅ Use small mock datasets (fast CI runs)
- ✅ Mock HuggingFace Hub API calls

---

## Examples of Good vs Bogus Tests

### ✅ Good Test (Tests Behavior)

```python
def test_classifier_handles_empty_embedding_array():
    """Verify classifier behavior with empty embeddings array"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)
    X_train = np.random.rand(100, 1280)
    y_train = np.array([0, 1] * 50)
    classifier.fit(X_train, y_train)

    empty_embeddings = np.array([]).reshape(0, 1280)  # Shape: (0, 1280)

    # Act & Assert - sklearn will raise on empty input
    with pytest.raises(ValueError):
        classifier.predict(empty_embeddings)
```

**Why it's good:**
- Tests observable behavior (error on invalid input)
- Uses correct API (embeddings array, not sequences)
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
def test_embed_sequence_validates_before_extraction(mock_transformers_model):
    """Verify invalid sequences are rejected by embed_sequence"""
    # Arrange
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )

    # Act & Assert - embed_sequence raises ValueError for invalid sequences
    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.embed_sequence("QVQLVQSG-AEVKKPGA")  # Gap character
```

**Why it's good:**
- Mocks only I/O boundary (transformers model loading via fixture)
- Tests domain logic (sequence validation)
- Verifies error handling behavior
- Uses correct method name (`embed_sequence` not `extract_embeddings`)

### ❌ Bogus Test (Over-mocked)

```python
def test_embeddings_processes_sequences(mocker):
    """Verify embeddings are extracted"""
    # Arrange
    mock_extractor = mocker.Mock()
    mock_extractor.embed_sequence.return_value = np.zeros(1280)

    # Act
    result = mock_extractor.embed_sequence("QVQLVQSG")

    # Assert
    assert result.shape == (1280,)
```

**Why it's bogus:**
- Mocks the thing we're testing (ESMEmbeddingExtractor itself)
- Test always passes (mock returns exactly what we tell it to)
- Doesn't test any real behavior (no validation, no processing)
- Completely useless - would pass even if implementation is broken

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
- ✅ Transformers model loading (650MB ESM model, heavy, slow)
- ✅ Network requests (unreliable in tests)
- ✅ File system (for error cases)
- ❌ Not domain logic, not transformations, not business rules

**How to mock ESM model:**
```python
@pytest.fixture
def mock_transformers_model(monkeypatch):
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", MockESMModel)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", MockTokenizer)
```

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
