# Type Checking

**Target Audience:** Developers writing type-safe code

**Purpose:** Understand type safety requirements, mypy configuration, and best practices for adding type annotations

---

## When to Use This Guide

Use this guide if you're:
- ✅ **Writing new functions** (must add complete type annotations)
- ✅ **Fixing mypy errors** (understanding common issues and solutions)
- ✅ **Understanding CI failures** (type checking gate)
- ✅ **Contributing code** (all PRs must pass strict type checking)

---

## Related Documentation

- **Workflow:** [Development Workflow](development-workflow.md) - Code quality commands (`make typecheck`)
- **Architecture:** [Architecture](architecture.md) - Type safety philosophy
- **Security:** [Security Guide](security.md) - Security scanning and best practices

---

## Type Safety Requirements

### Enforcement Level

**This repository maintains 100% type safety on production code.**

- **Production code (`src/`):** Must pass `mypy --strict` with 0 errors
- **Mypy mode:** `disallow_untyped_defs = true` (all functions require type annotations)
- **CI enforcement:** Type failures block PR merges
- **No exceptions:** All public functions must have complete type annotations

### Current Status

✅ **Production code (`src/antibody_training_esm/`):** 100% type-safe (21 files, 0 errors)

**Verification:**
```bash
uv run mypy src/ --strict
# Expected: Success: no issues found in 21 source files
```

---

## Mypy Configuration

### Configuration Location

All mypy settings are in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true  # ✅ STRICT: All functions must be typed
ignore_missing_imports = true  # External libraries without stubs
exclude = [
    "experiments/",
    "reference_repos/",
]
```

### Key Settings

- **`disallow_untyped_defs = true`**: Every function must have type annotations (parameters + return type)
- **`warn_return_any = true`**: Prevents functions from returning `Any` type
- **`ignore_missing_imports = true`**: Allows using libraries without type stubs
- **`python_version = "3.12"`**: Target Python version (matches `.python-version`)

### Exclusions

Directories excluded from type checking:
- `experiments/` - Research/experimental code
- `reference_repos/` - Third-party reference implementations

---

## Type Annotation Patterns

### Function Signatures

**Basic function with return value:**
```python
def process_sequences(sequences: list[str]) -> pd.DataFrame:
    """Process a list of sequences."""
    return pd.DataFrame({"sequence": sequences})
```

**Function with no return value:**
```python
def log_message(message: str, level: str = "INFO") -> None:
    """Log a message."""
    print(f"[{level}] {message}")
```

**Function with optional parameters (use `| None`):**
```python
def load_data(
    file_path: str | Path,
    delimiter: str = ",",
    header: int | None = 0
) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(file_path, delimiter=delimiter, header=header)
```

### Complex Types

**Union types (use `|` syntax):**
```python
from pathlib import Path

def read_file(path: str | Path) -> str:
    """Read file from string or Path object."""
    return Path(path).read_text()
```

**Sequences (prefer `Sequence` for read-only):**
```python
from collections.abc import Sequence

def calculate_mean(values: Sequence[float]) -> float:
    """Calculate mean of numeric values."""
    return sum(values) / len(values)
```

**Dictionaries:**
```python
def get_config() -> dict[str, Any]:
    """Return configuration dictionary."""
    return {"model": "esm1v", "batch_size": 16}
```

**NumPy arrays:**
```python
import numpy as np
from numpy.typing import NDArray

def embed_sequences(sequences: list[str]) -> NDArray[np.float32]:
    """Generate embeddings for sequences."""
    # Returns numpy array of float32
    return np.random.randn(len(sequences), 1280).astype(np.float32)
```

### Test Functions

**Test functions always return `None`:**
```python
def test_embedding_shape() -> None:
    """Test that embeddings have correct shape."""
    embeddings = generate_embeddings(["ACGT"])
    assert embeddings.shape == (1, 1280)
```

**Pytest fixtures:**
```python
import pytest
from collections.abc import Generator

@pytest.fixture
def mock_model() -> Generator[MockESM, None, None]:
    """Provide mock ESM model for testing."""
    model = MockESM()
    yield model
    model.cleanup()
```

**Fixtures returning concrete types:**
```python
@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provide sample DataFrame for testing."""
    return pd.DataFrame({"sequence": ["ACGT"], "label": [0]})
```

---

## Common Type Errors and Solutions

### 1. PEP 484 Implicit Optional

**Error:**
```
error: Incompatible default for argument "param" (default has type "None", argument has type "str")
```

**Problem:** Parameter has default value `None` but type doesn't include `None`.

**Solution:** Use `| None` for optional parameters:
```python
# ❌ WRONG
def load_model(path: str = None) -> Model:
    ...

# ✅ CORRECT
def load_model(path: str | None = None) -> Model:
    ...
```

### 2. Missing Return Type Annotation

**Error:**
```
error: Function is missing a return type annotation [no-untyped-def]
```

**Problem:** Function doesn't specify what it returns.

**Solution:** Add explicit return type:
```python
# ❌ WRONG
def calculate_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# ✅ CORRECT
def calculate_metrics(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32]
) -> float:
    return accuracy_score(y_true, y_pred)
```

### 3. List vs ndarray Type Mismatches

**Error:**
```
error: Incompatible types in assignment (expression has type "list[int]", variable has type "ndarray")
```

**Problem:** Mixing Python lists and NumPy arrays without explicit conversion.

**Solution:** Be explicit about types and conversions:
```python
# ❌ WRONG
labels: NDArray[np.int32] = [0, 1, 0, 1]  # list assigned to ndarray type

# ✅ CORRECT
labels: NDArray[np.int32] = np.array([0, 1, 0, 1], dtype=np.int32)
```

### 4. Any Return Type Issues

**Error:**
```
error: Returning Any from function declared to return "DataFrame" [no-any-return]
```

**Problem:** Function calls untyped code that returns `Any`.

**Solution:** Add explicit type annotation at call site:
```python
# ❌ WRONG
def load_data(path: str) -> pd.DataFrame:
    return pickle.load(open(path, 'rb'))  # pickle.load returns Any

# ✅ CORRECT
def load_data(path: str) -> pd.DataFrame:
    data: pd.DataFrame = pickle.load(open(path, 'rb'))
    return data
```

### 5. Parameter Type Inference Failures

**Error:**
```
error: Function is missing a type annotation for one or more arguments [no-untyped-def]
```

**Problem:** Function parameters don't have type annotations.

**Solution:** Add type annotations to all parameters:
```python
# ❌ WRONG
def process_batch(sequences, labels):
    ...

# ✅ CORRECT
def process_batch(
    sequences: list[str],
    labels: list[int]
) -> pd.DataFrame:
    ...
```

---

## CI Integration

### How Mypy Runs in CI

Type checking is enforced in the `quality` job of `.github/workflows/ci.yml`:

```yaml
- name: Type checking with mypy
  run: uv run mypy src/ --strict
  continue-on-error: false  # ✅ ENFORCED: Failures block merge
```

**What's checked:**
- Only `src/` directory (production code)
- Uses `--strict` mode (maximum type safety)
- Failures block PR merges

**What's NOT checked in CI:**
- Test files (`tests/`) - type checking recommended but not enforced
- Preprocessing scripts (`preprocessing/`) - lower priority, not enforced
- Utility scripts (`scripts/`) - not enforced

### Local Verification

**Before committing:**
```bash
make typecheck
# Runs: uv run mypy src/ --strict
```

**Check specific files:**
```bash
uv run mypy src/antibody_training_esm/core/classifier.py
```

**Check entire codebase (including tests):**
```bash
uv run mypy .
# Note: May show errors in tests/preprocessing if not fully typed
```

---

## Best Practices

### 1. Type All New Code

Every new function must have complete type annotations:

```python
# ✅ COMPLETE TYPE ANNOTATIONS
def train_model(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    config: dict[str, Any]
) -> LogisticRegression:
    """Train logistic regression model."""
    classifier = LogisticRegression(**config)
    classifier.fit(X, y)
    return classifier
```

### 2. Avoid `Any` Type

Use `Any` only when absolutely necessary:

```python
# ❌ AVOID
def process_data(data: Any) -> Any:
    ...

# ✅ PREFER SPECIFIC TYPES
def process_data(data: pd.DataFrame | dict[str, list[str]]) -> pd.DataFrame:
    ...
```

### 3. No Blanket `# type: ignore`

Only use `# type: ignore` for library stub gaps, with justification:

```python
# ❌ WRONG - Hides real type errors
def broken_function(x: int) -> str:
    return x  # type: ignore

# ✅ CORRECT - Justified for library limitation
from sklearn.base import BaseEstimator

def get_params(model: BaseEstimator) -> dict[str, Any]:
    # sklearn stubs are incomplete for get_params()
    return model.get_params()  # type: ignore[no-any-return]
```

### 4. Be Specific with Collection Types

Always specify element types for collections:

```python
# ❌ VAGUE
sequences: list = ["ACGT", "TGCA"]

# ✅ SPECIFIC
sequences: list[str] = ["ACGT", "TGCA"]

# ✅ EVEN BETTER (for read-only)
from collections.abc import Sequence
sequences: Sequence[str] = ("ACGT", "TGCA")
```

### 5. Use Type Aliases for Complex Types

For frequently used complex types, create aliases:

```python
from typing import TypeAlias
from numpy.typing import NDArray
import numpy as np

# Define alias
EmbeddingArray: TypeAlias = NDArray[np.float32]

# Use in signatures
def extract_embeddings(sequences: list[str]) -> EmbeddingArray:
    """Extract embeddings for sequences."""
    return np.zeros((len(sequences), 1280), dtype=np.float32)
```

---

## Troubleshooting

### "Function is missing a type annotation"

**Fix:** Add `-> None` or `-> ReturnType` to function signature:
```python
def my_function(param: str) -> None:  # Add this
    print(param)
```

### "Incompatible types in assignment"

**Fix:** Ensure variable types match assigned values:
```python
# Wrong: list[str] = list[int]
values: list[str] = [1, 2, 3]  # ❌

# Correct: list[int] = list[int]
values: list[int] = [1, 2, 3]  # ✅
```

### "Cannot infer type of variable"

**Fix:** Add explicit type annotation:
```python
# Mypy can't infer type from empty list
data = []  # ❌ Type unknown

# Add explicit type
data: list[str] = []  # ✅ Type known
```

### Mypy taking too long

**Fix:** Check specific directories instead of entire codebase:
```bash
# Fast: Check only production code
uv run mypy src/

# Slow: Check everything (including tests)
uv run mypy .
```

---

## Resources

### Official Documentation

- **MyPy Docs:** https://mypy.readthedocs.io/
- **Python Typing:** https://docs.python.org/3/library/typing.html
- **PEP 484:** Type Hints - https://peps.python.org/pep-0484/

### Internal Resources

- **Mypy config:** `pyproject.toml` (lines 85-100)
- **CI workflow:** `.github/workflows/ci.yml` (mypy job)
- **Example typed code:** `src/antibody_training_esm/core/` (all modules)

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
