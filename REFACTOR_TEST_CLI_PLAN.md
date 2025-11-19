# Refactoring Plan: Split `src/antibody_training_esm/cli/test.py`

**Date:** 2025-11-18
**Objective:** Refactor the monolithic `test.py` (630+ lines) into a modular package structure to improve maintainability, testability, and readability (Fix #6 from Architectural Plan).

## 1. New Directory Structure

We will create a new package `src/antibody_training_esm/cli/testing/` to house the components.

```
src/antibody_training_esm/cli/
├── test.py                  # Entry point (lean wrapper)
└── testing/                 # New package
    ├── __init__.py
    ├── config.py            # Configuration management
    ├── data.py              # Dataset loading and validation
    ├── evaluation.py        # Metric calculation and model evaluation
    ├── visualization.py     # Plotting and result saving
    └── tester.py            # Core ModelTester orchestrator
```

## 2. Component Mapping

| Original Component | Destination Module | Notes |
|-------------------|--------------------|-------|
| `TestConfig` | `testing/config.py` | Dataclass definition |
| `load_config_file` | `testing/config.py` | Helper function |
| `create_sample_test_config` | `testing/config.py` | Helper function |
| `ModelTester.load_dataset` | `testing/data.py` | Converted to standalone function `load_dataset(path, config)` |
| `ModelTester.detect_assay_type` | `testing/evaluation.py` | Converted to standalone function |
| `ModelTester.evaluate_pretrained` | `testing/evaluation.py` | Converted to standalone function |
| `ModelTester.plot_confusion_matrix` | `testing/visualization.py` | Converted to standalone function |
| `ModelTester.save_detailed_results` | `testing/visualization.py` | Converted to standalone function |
| `ModelTester` | `testing/tester.py` | Remains as the orchestrator class, importing helpers |
| `main` | `test.py` | Remains as CLI entry point |

## 3. Implementation Steps

1.  **Create Directory:** `mkdir -p src/antibody_training_esm/cli/testing`
2.  **Create Modules:**
    *   `config.py`: Extract `TestConfig` and helpers.
    *   `data.py`: Extract `load_dataset` logic.
    *   `evaluation.py`: Extract metric logic.
    *   `visualization.py`: Extract plotting logic.
    *   `tester.py`: Reconstruct `ModelTester` to use these modules.
3.  **Update Entry Point:** Rewrite `src/antibody_training_esm/cli/test.py` to import `ModelTester` from the new package.
4.  **Verification:**
    *   Run `uv run antibody-test --help` to ensure CLI is intact.
    *   Run existing tests: `uv run pytest tests/unit/cli/test_preprocess.py` (or relevant tests).

## 4. Dependencies

All imports currently in `test.py` will be distributed to the relevant modules.
- `pandas`, `numpy`: Used in `data.py`, `evaluation.py`, `visualization.py`.
- `matplotlib`, `seaborn`: Used in `visualization.py`.
- `sklearn`: Used in `evaluation.py`.
- `torch`: Used in `tester.py` (for cleanup/inference).

## 5. Justification

This refactor isolates "mechanisms" (loading, plotting, measuring) from "policy" (orchestrating the test run). It solves the "god object" problem of `ModelTester`.
