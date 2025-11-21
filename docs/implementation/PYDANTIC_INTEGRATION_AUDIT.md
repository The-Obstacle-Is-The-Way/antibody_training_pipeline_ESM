# Pydantic Integration Audit (Refined)

**Status:** Phases 1-4 implemented (see phase docs); audit retained for future hardening

Purpose: identify where runtime schema validation (Pydantic v2 + optional `pydantic-settings`, and Pandera for DataFrames) would harden the pipeline. This is an audit of current boundaries, reflecting the post-refactor codebase structure.

## Guiding Principles
- **Validate at Boundaries:** (I/O, configs, API requests, artifact metadata), not inside hot loops or per-token paths.
- **Static vs. Runtime:** Keep `mypy` strict as the static layer; Pydantic adds runtime guards and clear error messages.
- **DataFrame vs. Row:** Prefer **Pandera** for DataFrame-wide validation (batch); **Pydantic** for configs, request/response objects, and per-row records (streaming/interactive).
- **Hydra Compatibility:** Make Pydantic optional (extra) and Hydra-compatible: convert `DictConfig` → dict → `model_validate`.
- **Fail Fast:** Emit actionable errors pointing to specific inputs rather than generic `KeyError` or `AttributeError`.

## Integration Surface Inventory

### 1) Data Ingestion & Preprocessing
- **Modules:**
  - `preprocessing/validation_utils.py` (Current manual validation logic).
  - `src/antibody_training_esm/datasets/base.py` (Base loader & mixins).
  - `src/antibody_training_esm/datasets/*.py` (Specific loaders).
- **Current State:**
  - `validation_utils.py` provides manual functions (`validate_dataframe_columns`, `validate_amino_acids`) akin to a "manual Pandera".
  - Dataset loaders (`Jain`, `Harvey`, etc.) use these utils but lack formal schemas.
  - Paths are managed via `preprocessing/paths.py` with some global constants.
- **Proposed:**
  - **Pandera Schemas:** Define `DatasetSchema` (columns: sequence, label, etc.) to replace manual `validate_dataframe_columns` checks. This ensures data integrity immediately upon load.
  - **Row Models:** `SequenceRecord` for single-row validation (alphabet check, length constraints) during iteration or inference, replacing `validate_amino_acids`.
  - **Settings:** `DataSettings` (via `pydantic-settings`) to replace `preprocessing/paths.py` globals, allowing robust env var overrides for CI/Docker.

### 2) Training Configuration & Orchestration
- **Modules:**
  - `src/antibody_training_esm/core/trainer.py` (Orchestrator & config validation).
  - `src/antibody_training_esm/conf/**/*.yaml` (Hydra configs).
- **Current State:**
  - `validate_config` in `trainer.py` performs manual dictionary checks (e.g., `if "classifier" not in config...`).
  - Values are accessed via raw dictionary lookups or `getattr`, risking runtime `KeyError`s deep in the stack.
- **Proposed:**
  - **Config Models:** Create a hierarchy matching Hydra:
    - `TrainingConfig` (epochs, save_model flags).
    - `ModelConfig` (ESM model name, device).
    - `ClassifierConfig` (strategy, hyperparameters).
  - **Validation Strategy:** In `trainer.py`, convert the resolved Hydra `DictConfig` to these Pydantic models immediately. This ensures type safety and valid enum values (e.g., `device` must be "cpu", "cuda", or "mps") before training starts.

### 3) Prediction Surfaces (CLI & Web App)
- **Modules:**
  - `src/antibody_training_esm/core/prediction.py` (Core logic).
  - `src/antibody_training_esm/cli/app.py` (Gradio).
  - `src/antibody_training_esm/cli/predict.py` (CLI).
- **Current State:**
  - `Predictor` accepts raw strings/paths.
  - `app.py` contains a custom `validate_input` function for amino acid checking.
  - `run_prediction` helper manually sets defaults via `getattr`.
- **Proposed:**
  - **Request Models:** `PredictionRequest` (single sequence) and `BatchPredictionRequest` (list/file).
    - Move `validate_input` logic into a Pydantic validator within `PredictionRequest`.
  - **Response Models:** `PredictionResult` containing label, probability, and metadata.
  - **Impact:** Standardizes validation across CLI and Web App, removing duplicated logic in `app.py`.

### 4) Model Artifacts & Serialization
- **Modules:**
  - `src/antibody_training_esm/core/training/serialization.py` (Save/Load logic).
- **Current State:**
  - `save_model` manually constructs a metadata dictionary.
  - `load_model_from_npz` manually parses JSON and handles type casting quirks (e.g., converting string keys back to integers for `class_weight`).
  - Configuration is loaded via `yaml.safe_load` without schema enforcement.
- **Proposed:**
  - **Metadata Model:** `ModelArtifactMetadata` to structure the JSON sidecar file.
    - Fields: `model_name`, `sklearn_version`, `classifier_params`, `metrics`.
    - *Crucial:* Handles the serialization/deserialization of complex types like `class_weight` automatically.
  - **Load Validation:** Validate the loaded JSON against this model to ensure the artifact is compatible with the current code version.

### 5) Evaluation & Metrics
- **Modules:**
  - `src/antibody_training_esm/core/training/metrics.py`.
- **Current State:**
  - Metrics are passed around as loose dictionaries (`results = {}`).
  - `save_cv_results` manually cleans types for YAML export.
- **Proposed:**
  - **Metric Models:** `EvaluationMetrics` (accuracy, f1, etc.) and `CVResults` (mean/std per metric).
  - **Serialization:** Use `model_dump()` for robust export to JSON/YAML, replacing manual type conversion helpers.

## Prioritized Implementation Plan

1.  **Phase 1: Prediction Hardening (High Impact/Low Risk)**
    *   Create `PredictionRequest`/`PredictionResult` models.
    *   Integrate into `core/prediction.py` and `cli/app.py`.
    *   *Benefit:* Immediate validation for end-users; removes ad-hoc checks in Gradio app.

2.  **Phase 2: Configuration Safety (Developer Experience)**
    *   Define `TrainingConfig` hierarchy.
    *   Integrate into `core/trainer.py` to validate Hydra output.
    *   *Benefit:* Catches config errors at startup; eliminates "runtime key errors".

3.  **Phase 3: Data Integrity (Reliability)**
    *   Implement Pandera schemas for datasets in `datasets/`.
    *   Replace manual `validation_utils` checks.
    *   *Benefit:* Prevents silent data corruption issues (like the "Jain column mismatch" of the past).

4.  **Phase 4: Artifacts & Metrics (Robustness)**
    *   Implement `ModelArtifactMetadata` and `EvaluationMetrics`.
    *   Update `serialization.py` and `metrics.py`.
    *   *Benefit:* Ensures production models are self-describing and load reliably.
