# Testing Strategy

**Target Audience:** Developers writing tests

**Purpose:** Understand test architecture, patterns, and best practices for maintaining high-quality test suite

---

## When to Use This Guide

Use this guide if you're:
- ✅ **Writing new tests** (unit, integration, or E2E)
- ✅ **Understanding test markers** (when to use unit vs integration vs e2e)
- ✅ **Fixing test failures** (debugging test issues)
- ✅ **Improving coverage** (understanding coverage requirements)
- ✅ **Understanding mocking strategy** (what to mock, what NOT to mock)

---

## Related Documentation

- **Workflow:** [Development Workflow](development-workflow.md) - Test commands (`make test`, `uv run pytest`)
- **Architecture:** [Architecture](architecture.md) - System design and components
- **Type Checking:** [Type Checking Guide](type-checking.md) - Type safety requirements

---

## Testing Philosophy

### Core Principles (Robert C. Martin / Uncle Bob)

**1. Test Behaviors, Not Implementation**
- ✅ Test WHAT the code does (contracts, interfaces, outcomes)
- ❌ Don't test HOW it does it (private methods, internal state)
- Example: Test that `classifier.predict(embeddings)` returns 0 or 1, not that it calls LogisticRegression internally

**2. Minimize Mocking (No Bogus Mocks)**
- ✅ Mock only I/O boundaries: network calls, file system (when necessary), external APIs
- ✅ Mock heavyweight dependencies: ESM model downloads (~650MB), GPU operations
- ❌ Don't mock domain logic, data transformations, or business rules
- Example: Mock the ESM model loading, but don't mock DataFrame operations

**3. FIRST Principles**
- **F**ast: Unit tests run in milliseconds (no model downloads, no disk I/O unless necessary)
- **I**ndependent: Tests don't depend on each other, can run in any order
- **R**epeatable: Same results every time, no flaky tests
- **S**elf-validating: Pass/fail with clear assertions, no manual inspection
- **T**imely: Written alongside (or retroactively for) production code

**4. Test Pyramid**
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

**5. Arrange-Act-Assert (AAA)**
```python
def test_classifier_predicts_nonspecific_antibody():
    # Arrange: Set up test data
    embeddings = np.random.rand(1, 1280)  # Mock ESM embedding
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.fit(np.random.rand(100, 1280), np.array([0, 1] * 50))  # Pre-fit

    # Act: Execute behavior
    prediction = classifier.predict(embeddings)

    # Assert: Verify outcome
    assert prediction[0] in [0, 1]
```

**6. Single Responsibility (Tests Too)**
- One test verifies one behavior
- Test name describes the behavior: `test_classifier_rejects_invalid_embeddings`
- Test body is short (<20 lines), focused, readable

**7. DRY (Don't Repeat Yourself)**
- Use pytest fixtures for shared setup
- Extract common test data to conftest.py
- Reuse test utilities (e.g., `mock_transformers_model` fixture)

---

## Test Architecture

### Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures, test data, utilities
├── fixtures/                      # Test data (CSVs, mock sequences)
│   ├── mock_sequences.py          # Sample antibody sequences
│   ├── mock_datasets/             # Small CSV files for fast tests
│   │   ├── boughter_sample.csv
│   │   ├── boughter_annotated.csv  # ANARCI-annotated fixtures
│   │   ├── jain_sample.csv
│   │   ├── jain_annotated.csv
│   │   ├── harvey_high_sample.csv
│   │   ├── harvey_low_sample.csv
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
│       ├── test_base_annotation.py # ANARCI annotation logic
│       ├── test_boughter.py       # Boughter dataset behavior
│       ├── test_harvey.py         # Harvey dataset behavior
│       ├── test_jain.py           # Jain dataset behavior
│       └── test_shehata.py        # Shehata dataset behavior
│
├── integration/                   # Integration tests (20% of tests)
│   ├── test_boughter_embedding_compatibility.py
│   ├── test_harvey_embedding_compatibility.py
│   ├── test_jain_embedding_compatibility.py
│   ├── test_shehata_embedding_compatibility.py
│   ├── test_dataset_pipeline.py   # Dataset → Embedding → Training
│   ├── test_cross_validation.py   # Full CV pipeline
│   ├── test_model_persistence.py  # Save/load model workflow
│   └── test_model_tester.py       # ModelTester integration
│
└── e2e/                           # End-to-end tests (10% of tests)
    ├── test_train_pipeline.py     # Full training pipeline (CLI)
    └── test_reproduce_novo.py     # Reproduce Novo Nordisk results
```

### Current Status

**Test Counts:**
- **Total tests:** 403 tests (311 unit + 75 integration + 17 E2E)
- **Test files:** 24 test modules

**Coverage:**
- **Overall:** 90.80% (enforced ≥70% in CI)
- **Core modules:** 97.96% average (classifier 100%, embeddings 94.50%, trainer 99.37%)
- **Datasets:** 89.58% average (boughter 91.67%, harvey 86.11%, jain 96.64%, shehata 88.42%, base 85.06%)
- **CLI:** 85.84% (test.py), 100% (train.py), 78.12% (preprocess.py)
- **Data loaders:** 98.41% (loaders.py)

**Quality:**
- ✅ Zero linting errors (ruff)
- ✅ Zero type errors (mypy)
- ✅ Zero test failures
- ✅ All tests passing consistently

---

## Test Markers & When to Use

### Registered Markers (pyproject.toml)

**`@pytest.mark.unit`** - Fast unit tests (run on every PR)
- **Speed:** <1s per test
- **Dependencies:** Mocked (transformers, file I/O)
- **Use cases:** Single function/method behavior
- **Example:** Test that `classifier.predict()` returns binary labels

**`@pytest.mark.integration`** - Integration tests (run on every PR)
- **Speed:** <10s per test
- **Dependencies:** Some mocked (transformers), some real (datasets, pandas)
- **Use cases:** Multi-component interactions
- **Example:** Test dataset → embeddings → training pipeline

**`@pytest.mark.e2e`** - End-to-end tests (expensive, run on schedule)
- **Speed:** >30s per test
- **Dependencies:** Mostly real (small test datasets), transformers mocked
- **Use cases:** Full user workflows (CLI → model → results)
- **Example:** Test full training pipeline from CLI

**`@pytest.mark.slow`** - Slow tests (>30s, run on schedule)
- **Speed:** >30s per test
- **Dependencies:** Real data, real computations
- **Use cases:** Expensive operations (large dataset processing, hyperparameter sweeps)
- **Example:** Test cross-validation on full dataset

### How to Use Markers

```python
import pytest

@pytest.mark.unit
def test_classifier_predicts_binary_labels():
    """Verify predictions are binary (0 or 1)"""
    # ...

@pytest.mark.integration
def test_boughter_to_jain_pipeline():
    """Verify Boughter training set can predict on Jain test set"""
    # ...

@pytest.mark.e2e
def test_full_training_pipeline_end_to_end():
    """Verify complete training pipeline from CLI"""
    # ...
```

---

## Writing Tests

### Test Structure (AAA Pattern)

**Every test should follow Arrange-Act-Assert:**

```python
def test_embed_sequence_extracts_1280_dim_vector(mock_transformers_model):
    """Verify single sequence embedding returns 1280-dimensional vector"""
    # Arrange: Set up test data
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )

    # Act: Execute behavior
    embedding = extractor.embed_sequence("QVQLVQSGAEVKKPGA")

    # Assert: Verify outcome
    assert embedding.shape == (1280,)
    assert isinstance(embedding, np.ndarray)
```

### Naming Conventions

**Test names should describe behavior:**
- ✅ `test_classifier_rejects_invalid_embeddings`
- ✅ `test_dataset_loads_full_stage`
- ✅ `test_embed_sequence_validates_before_extraction`
- ❌ `test_predict` (too vague)
- ❌ `test_case_1` (meaningless)

### Fixture Usage

**Use pytest fixtures for shared setup:**

```python
@pytest.fixture
def mock_transformers_model(monkeypatch):
    """Mock HuggingFace transformers model to avoid downloading 650MB"""
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
```

### Mocking Strategy

**What to Mock (✅ Allowed):**

1. **ESM Model Loading**
   - Mock `transformers.AutoModel.from_pretrained()` to avoid 650MB download
   - Mock `transformers.AutoTokenizer.from_pretrained()`
   - Return fake torch tensors for embeddings
   - **Model:** `facebook/esm1v_t33_650M_UR90S_1`
   - **Library:** HuggingFace transformers (NOT `esm.pretrained`)

2. **File I/O (Selectively)**
   - Mock missing files for error handling tests
   - Use small mock CSV files for fast unit tests
   - Use `tmp_path` fixture for temporary file tests

3. **External APIs / Network Calls** (if added)
   - Mock HuggingFace API calls (model downloads)
   - Mock any web requests

4. **GPU Operations** (if applicable)
   - Mock CUDA availability checks
   - Use CPU for all tests

**What NOT to Mock (❌ Forbidden):**

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

## Coverage Requirements

### Enforcement

**CI Enforcement:**
```bash
# .github/workflows/ci.yml
uv run coverage report --fail-under=70
```

**Current Coverage:** 90.80% (enforced minimum: ≥70%)

### Per-Module Targets

| Module / Area | Target | Current Status |
|---------------|--------|----------------|
| `core/classifier.py` | ≥90% | ✅ 100.00% (75/75 statements) |
| `core/embeddings.py` | ≥85% | ✅ 94.50% (89/89 statements) |
| `core/trainer.py` | ≥85% | ✅ 99.37% (136/136 statements) |
| `datasets/*.py` (each) | ≥80% | ✅ 89.58% avg (boughter 91.67%, harvey 86.11%, jain 96.64%, shehata 88.42%) |
| `datasets/base.py` | ≥80% | ✅ 85.06% (183/183 statements) |
| `data/loaders.py` | ≥80% | ✅ 98.41% (49/49 statements) |
| `cli/train.py` | ≥70% | ✅ 100.00% (18/18 statements) |
| `cli/test.py` | ≥70% | ✅ 85.84% (269/269 statements) |
| `cli/preprocess.py` | ≥70% | ✅ 78.12% (30/30 statements) |

**What NOT to Cover:**
- `__init__.py` files (just imports)
- Private methods (test through public API)
- Deprecated code (remove it instead)
- Debug print statements (remove them)

**Coverage Philosophy:**
- ✅ Focus on critical paths (training, prediction)
- ✅ Focus on edge cases (empty inputs, invalid data)
- ✅ Focus on error handling
- ❌ Don't waste time testing trivial getters/setters

---

## Test Fixtures and Mocking

### Mock Datasets (`tests/fixtures/mock_datasets/`)

**Small CSV files (10-20 rows) for fast unit tests:**

**boughter_sample.csv** (20 rows):
```csv
id,VH_sequence,VL_sequence,label,flagging_rate
seq_001,QVQLVQSGAEVKKPGA...,DIQMTQSPSSLSASVGD...,0,0
seq_002,EVQLLESGGGLVQPGG...,EIVLTQSPGTLSLSPGE...,1,4
...
```

**jain_sample.csv** (15 rows):
```csv
antibody_id,VH_sequence,VL_sequence,ELISA_flags,PSR_ranking
mAb_001,QVQLVQSGAEVKKPGA...,DIQMTQSPSSLSASVGD...,0,Low
mAb_002,EVQLLESGGGLVQPGG...,EIVLTQSPGTLSLSPGE...,5,High
...
```

**ANARCI-annotated fixtures** (for fragment testing):
- `boughter_annotated.csv` - Includes VH_CDR1, VH_CDR2, VH_CDR3, VH_FWR1, etc.
- `jain_annotated.csv` - Includes VH/VL CDR/FWR columns

### Mock Sequences (`tests/fixtures/mock_sequences.py`)

```python
# Valid antibody sequences
VALID_VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS"
VALID_VL = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"

# Invalid sequences (for error testing)
SEQUENCE_WITH_GAP = "QVQL-VQSGAEVKKPGA"
SEQUENCE_WITH_INVALID_AA = "QVQLVQSGAEVKKPGABBB"  # 'B' is invalid
```

### Common Fixtures

**From conftest.py:**

```python
@pytest.fixture
def mock_transformers_model(monkeypatch):
    """Mock HuggingFace transformers (avoid 650MB download)"""
    # See full implementation in "Fixture Usage" section above

@pytest.fixture
def tmp_path():
    """Pytest built-in fixture for temporary directories"""
    # Auto-cleanup after test

@pytest.fixture
def cv_params():
    """Cross-validation parameters"""
    return {
        "n_splits": 5,
        "random_state": 42,
        "stratified": True
    }

@pytest.fixture
def test_params():
    """BinaryClassifier test parameters"""
    return {
        "C": 1.0,
        "penalty": "l2",
        "random_state": 42
    }
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest

# Run only unit tests (fast)
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run only E2E tests
uv run pytest -m e2e

# Skip slow tests (for quick feedback)
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/unit/core/test_classifier.py

# Run specific test
uv run pytest tests/unit/core/test_classifier.py::test_classifier_predicts_binary_labels
```

### Coverage Commands

```bash
# Run with coverage report
uv run pytest --cov=src/antibody_training_esm --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src/antibody_training_esm --cov-report=html

# Enforce minimum coverage (CI)
uv run pytest --cov=src/antibody_training_esm --cov-fail-under=70

# Coverage with branch analysis
uv run pytest --cov=src/antibody_training_esm --cov-report=term --cov-branch
```

### CI Integration

**What runs in CI (.github/workflows/ci.yml):**

```yaml
# Unit tests with coverage
- name: Run unit tests with coverage
  run: |
    uv run pytest tests/unit/ \
      --cov=src/antibody_training_esm \
      --cov-report=xml \
      -v

# Integration tests
- name: Run integration tests
  run: |
    uv run pytest tests/integration/ \
      --junitxml=junit-integration.xml \
      -v

# Coverage enforcement
- name: Enforce coverage minimum
  run: uv run coverage report --fail-under=70
```

**What runs on schedule (not every PR):**
- E2E tests (`@pytest.mark.e2e`)
- Slow tests (`@pytest.mark.slow`)

**CI Mocking Strategy:**
- ✅ Mock transformers model loading (no 650MB ESM download in CI)
- ✅ Use CPU-only tests (no GPU in CI)
- ✅ Use small mock datasets (fast CI runs)
- ✅ Mock HuggingFace Hub API calls

---

## Best Practices

### 1. Test Behaviors, Not Implementation

```python
# ✅ GOOD: Test observable behavior
def test_classifier_handles_empty_embedding_array():
    """Verify classifier behavior with empty embeddings array"""
    classifier = BinaryClassifier(params=TEST_PARAMS)
    X_train = np.random.rand(100, 1280)
    y_train = np.array([0, 1] * 50)
    classifier.fit(X_train, y_train)

    empty_embeddings = np.array([]).reshape(0, 1280)

    with pytest.raises(ValueError):
        classifier.predict(empty_embeddings)

# ❌ BAD: Test implementation detail
def test_classifier_uses_logistic_regression():
    """Verify classifier uses LogisticRegression internally"""
    classifier = BinaryClassifier(params=TEST_PARAMS)
    classifier.fit(X, y)

    assert isinstance(classifier.classifier, LogisticRegression)  # Fragile!
```

### 2. Minimize Mocking

```python
# ✅ GOOD: Mock only I/O boundary
def test_embed_sequence_validates_before_extraction(mock_transformers_model):
    """Verify invalid sequences are rejected"""
    extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu"
    )

    with pytest.raises(ValueError, match="Invalid amino acid"):
        extractor.embed_sequence("QVQLVQSG-AEVKKPGA")  # Gap character

# ❌ BAD: Over-mocked
def test_embeddings_processes_sequences(mocker):
    """Verify embeddings are extracted"""
    mock_extractor = mocker.Mock()
    mock_extractor.embed_sequence.return_value = np.zeros(1280)

    result = mock_extractor.embed_sequence("QVQLVQSG")

    assert result.shape == (1280,)  # Always passes (mock returns what we tell it)
```

### 3. Use AAA Pattern

```python
# ✅ GOOD: Clear AAA structure
def test_jain_dataset_loads_full_stage():
    """Verify Jain dataset loads all 137 antibodies in 'full' stage"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(stage="full")

    # Assert
    assert len(df) == 137
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns
```

### 4. Clear Test Names

```python
# ✅ GOOD: Descriptive test names
def test_classifier_applies_psr_threshold_calibration():
    """Verify PSR assay uses 0.5495 decision threshold (Novo parity value)"""
    # ...

def test_embed_sequence_rejects_invalid_amino_acids():
    """Verify embed_sequence raises ValueError for invalid sequences"""
    # ...

# ❌ BAD: Vague test names
def test_predict():
    """Test predict function"""
    # What behavior? What input? What expected output?

def test_case_1():
    """Test case 1"""
    # Meaningless
```

### 5. Single Responsibility

```python
# ✅ GOOD: One test, one behavior
def test_classifier_predicts_binary_labels():
    """Verify predictions are binary (0 or 1)"""
    # Test only binary output

def test_classifier_applies_psr_threshold():
    """Verify PSR threshold is 0.5495"""
    # Test only PSR threshold

# ❌ BAD: Multiple behaviors in one test
def test_classifier():
    """Test classifier works"""
    # Test binary output
    # Test PSR threshold
    # Test ELISA threshold
    # Test error handling
    # ... too much!
```

### 6. Use Fixtures for DRY

```python
# ✅ GOOD: Shared setup via fixture
@pytest.fixture
def trained_classifier():
    """Provide pre-trained classifier for tests"""
    classifier = BinaryClassifier(params=TEST_PARAMS)
    X_train = np.random.rand(100, 1280)
    y_train = np.array([0, 1] * 50)
    classifier.fit(X_train, y_train)
    return classifier

def test_predict_binary(trained_classifier):
    """Test binary prediction"""
    predictions = trained_classifier.predict(np.random.rand(10, 1280))
    assert all(pred in [0, 1] for pred in predictions)

# ❌ BAD: Copy-paste setup
def test_predict_binary():
    """Test binary prediction"""
    classifier = BinaryClassifier(params=TEST_PARAMS)
    X_train = np.random.rand(100, 1280)
    y_train = np.array([0, 1] * 50)
    classifier.fit(X_train, y_train)  # Duplicated setup
    # ...
```

---

## Common Test Patterns

### Testing Classifiers

```python
@pytest.mark.unit
def test_classifier_predicts_binary_labels():
    """Verify predictions are binary (0 or 1)"""
    # Arrange
    X_train = np.random.rand(100, 1280)  # Mock embeddings (NOT sequences!)
    y_train = np.array([0, 1] * 50)
    classifier = BinaryClassifier(params=TEST_PARAMS)

    # Act
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_train[:10])

    # Assert
    assert all(pred in [0, 1] for pred in predictions)
```

**Key points:**
- Classifier operates on **embeddings**, not sequences
- Use mock embeddings (random arrays) for speed
- Don't mock LogisticRegression (it's lightweight, part of the contract)

### Testing Embeddings

```python
@pytest.mark.unit
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
    assert embedding.shape == (1280,)
    assert isinstance(embedding, np.ndarray)
```

**Key points:**
- Mock **transformers** (AutoModel, AutoTokenizer), NOT `esm.pretrained`
- Return deterministic fake tensors
- Don't mock sequence validation (that's the behavior we're testing)

### Testing Datasets

```python
@pytest.mark.unit
def test_jain_dataset_loads_full_stage():
    """Verify Jain dataset loads all 137 antibodies in 'full' stage"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(stage="full")

    # Assert
    assert len(df) == 137
    assert "VH_sequence" in df.columns  # NOT "sequence"!
    assert "VL_sequence" in df.columns
    assert "label" in df.columns
```

**Key points:**
- Datasets return `VH_sequence` and `VL_sequence`, NOT `sequence`
- Fragments created separately via `create_fragment_csvs(df, suffix="")`
- Use small mock CSVs (10-20 rows) for unit tests

### Testing Error Handling

```python
@pytest.mark.unit
def test_classifier_requires_fit_before_predict():
    """Verify classifier raises error when predicting before fit"""
    # Arrange
    classifier = BinaryClassifier(params=TEST_PARAMS)
    embeddings = np.random.rand(10, 1280)

    # Act & Assert
    with pytest.raises(ValueError, match="Classifier must be fitted"):
        classifier.predict(embeddings)
```

**Key points:**
- Use `pytest.raises()` for expected errors
- Match error message with `match` parameter (regex)
- Test both the error type AND message

---

## Troubleshooting

### Test Failures

**Run specific test with verbose output:**
```bash
uv run pytest tests/unit/core/test_classifier.py::test_predict_binary -v
```

**Show print statements:**
```bash
uv run pytest tests/unit/core/test_classifier.py -s
```

**Drop into debugger on failure:**
```bash
uv run pytest tests/unit/core/test_classifier.py --pdb
```

**Show full traceback:**
```bash
uv run pytest tests/unit/core/test_classifier.py --tb=long
```

### Fixture Issues

**List all available fixtures:**
```bash
uv run pytest --fixtures
```

**Check fixture usage:**
```bash
# Fixtures are in conftest.py
cat tests/conftest.py

# Or check mock_datasets/
ls tests/fixtures/mock_datasets/
```

**Common fixture errors:**
- ❌ Fixture not found: Check spelling, check conftest.py
- ❌ Fixture scope error: Use `tmp_path` (function scope), not `tmp_path_factory` (session scope)
- ❌ Fixture pollution: Each test should get clean fixture, check fixture scope

### Coverage Gaps

**Show missing lines:**
```bash
uv run pytest --cov=src/antibody_training_esm --cov-report=term-missing
```

**Generate HTML report for browsing:**
```bash
uv run pytest --cov=src/antibody_training_esm --cov-report=html
open htmlcov/index.html
```

**Check specific module:**
```bash
uv run pytest tests/unit/core/test_classifier.py --cov=src/antibody_training_esm/core/classifier
```

**Coverage too low:**
- ✅ Identify missing edge cases (empty inputs, invalid data)
- ✅ Add error handling tests
- ❌ Don't write bogus tests just to hit lines

---

## Resources

### Internal

- **Test suite:** `tests/` directory
- **Fixtures:** `tests/fixtures/` and `tests/conftest.py`
- **pytest config:** `pyproject.toml` (lines 85-110)
- **CI workflow:** `.github/workflows/ci.yml`

### External

- **pytest docs:** https://docs.pytest.org/
- **Robert C. Martin:** *Clean Code*, *Clean Architecture*
- **Martin Fowler:** *Refactoring*
- **Kent Beck:** *Test Driven Development: By Example*

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
