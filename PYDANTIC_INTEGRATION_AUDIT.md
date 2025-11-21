# Pydantic Integration Audit (Baseline)

Purpose: identify where runtime schema validation (Pydantic v2 + optional
`pydantic-settings`, and Pandera for DataFrames) would harden the pipeline.
This is an audit of current boundaries, not an implementation plan. Use it as
the source of truth to design phased work next.

## Guiding principles
- Validate at boundaries (I/O, configs, API requests, artifact metadata), not
  inside hot loops or per-token paths.
- Keep mypy strict as the static layer; Pydantic adds runtime guards and clear
  error messages.
- Prefer Pandera for DataFrame-wide validation; Pydantic for configs, request/
  response objects, and per-row records where useful.
- Make Pydantic optional (extra) and Hydra-compatible: convert `DictConfig` →
  dict → `model_validate`.
- Fail fast with actionable errors that point to the exact input/field.

## Integration surface inventory

### 1) Data ingestion and preprocessing
- Modules: `preprocessing/**/*.py`, `preprocessing/paths.py`,
  `src/antibody_training_esm/datasets/*.py`,
  `src/antibody_training_esm/data/loaders.py`.
- Current state: manual checks for paths/columns, ad hoc renames, and string
  parsing; sequence validation lives in helpers but not enforced uniformly.
- Proposed:
  - `BaseSettings` for dataset root paths (train/test/processed/cache) to
    replace free-form globals in `preprocessing/paths.py` and
    `datasets/default_paths.py`; allow env overrides for CI vs local data.
  - Row model (e.g., `SequenceRecord`) with validators for amino-acid alphabet,
    min/max length per assay, label domain {0,1}, and optional VH/VL presence.
    Use for ingestion in `datasets/*.py` and `data/loaders.load_local_data`.
  - Pandera schema per dataset stage to enforce column presence/dtypes and to
    catch corruption early (e.g., missing PSR/AC-SINS columns in Jain).
  - Structured stage selection enum for Jain (`full|ssot|parity`) and similar
    flags in Harvey/Shehata/Boughter loaders to prevent silent fallbacks.

### 2) Training config and Hydra boundary
- Modules: `src/antibody_training_esm/cli/train.py`,
  `src/antibody_training_esm/core/trainer.py` (`validate_config`),
  Hydra configs in `src/antibody_training_esm/conf/**`.
- Current state: manual dict checks for required keys, metrics, and device,
  then implicit casts in the trainer.
- Proposed:
  - Pydantic config models (`DataConfig`, `ModelConfig`, `ClassifierConfig`,
    `TrainingConfig`, `ExperimentConfig`) that validate after Hydra merge
    (`OmegaConf.to_container` → `model_validate`).
  - Enum fields for metrics/device/solver, numeric bounds for batch size,
    n_splits, thresholds, and logging levels.
  - Typed path fields for `train_file`, `test_file`, `embeddings_cache_dir`
    with existence checks and clearer errors than `FileNotFoundError` stacks.
  - Emit a single validated `TrainingSettings` object that downstream functions
    consume instead of raw dicts.

### 3) Prediction surfaces (CLI + Gradio)
- Modules: `src/antibody_training_esm/core/prediction.py`,
  `src/antibody_training_esm/cli/predict.py`,
  `src/antibody_training_esm/cli/app.py` (Gradio).
- Current state: string-based validation, manual file checks, and loose dict
  responses.
- Proposed:
  - Request/response models:
    - `PredictionRequest` (sequence, optional chain type, threshold, assay_type)
      with validators for alphabet, casing, and max length.
    - `PredictionBatchRequest` for CSV/DF inputs with required columns.
    - `PredictionResult` with prediction label, probability, and optional
      calibrated fields.
  - Classifier artifact model that validates `.pkl` vs `.npz` + config pairing
    and device hints before loading.
  - Use the models to sanitize Gradio inputs and CLI CSV rows before calling
    `Predictor`, so invalid inputs fail fast with user-friendly messages.

### 4) Model artifacts and metadata
- Modules: `src/antibody_training_esm/core/training/serialization.py`,
  `src/antibody_training_esm/core/classifier.py`,
  `src/antibody_training_esm/core/training/cache.py`.
- Current state: JSON metadata for NPZ saves is an untyped dict; cache keys and
  saved arrays are unchecked beyond existence.
- Proposed:
  - `ModelMetadata` Pydantic model to structure the JSON emitted with NPZ:
    classifier type, hyperparameters, embedding model, device, revision,
    batch size, sklearn version, and schema version.
  - `CacheEntry` model for embedding cache to assert array shapes/dtypes and
    model identifiers before reuse.
  - Validate `class_weight` and other deserialized fields on load to avoid
    silent shape/type drift across versions.

### 5) Evaluation and reporting
- Modules: `src/antibody_training_esm/core/training/metrics.py`,
  `src/antibody_training_esm/evaluation/**`.
- Current state: metrics are returned as loose dicts and written via helpers.
- Proposed:
  - `FoldMetrics` and `CrossValSummary` models with optional fields for ROC-AUC
    (nullable) and confusion matrices as fixed shapes.
  - Serialize metrics via `model_dump` to JSON for reproducible experiment
    artifacts and safer downstream analysis.

### 6) Paths, settings, and feature flags
- Modules: `preprocessing/paths.py`, `tests` env flags (e.g., heavy e2e toggles),
  `src/antibody_training_esm/conf/**`.
- Current state: constants in modules; env flags are implicit strings.
- Proposed:
  - `BaseSettings` for path roots (data, experiments, cache, checkpoints) to
    allow env overrides without code edits.
  - Typed settings for feature flags (`RUN_NOVO_E2E`, `RUN_PREDICT_CLI_E2E`,
    GPU enablement) with coercion to bool and centralized defaults.

### 7) External dataset descriptors
- Modules: `src/antibody_training_esm/data/loaders.py` (HF datasets).
- Current state: free-form strings for dataset name/split/columns.
- Proposed:
  - `HFDatasetConfig` model with allowed splits, column names, and revision
    strings; validate before calling `load_dataset` to avoid remote fetch
    errors mid-run.

## Non-goals and guardrails
- Do not replace Hydra; Pydantic should wrap the merged config for validation.
- Avoid per-token validation in performance-critical loops (embedding and model
  inference); validate inputs once at entry.
- Keep dependency optional via extras to avoid forcing Pydantic on minimal
  inference environments.
- Favor Pandera for DataFrame-wide checks where column types and lengths matter
  more than per-row business logic.

## Recommended starting points (for the phased plan)
1) Prediction request/response models (`core/prediction.py`, `cli/predict.py`,
   `cli/app.py`) — highest user impact and low risk.
2) Training config models wrapping Hydra output (`core/trainer.py`,
   `conf/**`) — replaces manual validation and reduces runtime config errors.
3) Dataset row + Pandera schemas at ingestion (`datasets/*.py`, `data/loaders.py`)
   — catches data corruption early with clear errors.

These three anchor areas should precede deeper artifact and metrics modeling,
and will inform how to stage implementation in the follow-up phased guide.
