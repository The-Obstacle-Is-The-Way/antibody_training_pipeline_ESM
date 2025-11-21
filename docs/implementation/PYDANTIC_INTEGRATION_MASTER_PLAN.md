# Pydantic Integration Master Plan

**Status:** Ready for Implementation
**Created:** 2025-11-20
**Version:** 1.0.0

---

## Executive Summary

This master plan documents the **complete Pydantic v2 integration** across the antibody training pipeline. The integration is split into **4 sequential phases** that progressively harden the codebase with runtime validation while maintaining 100% backward compatibility.

**Why Pydantic?**
- **Runtime Safety:** Catch errors at boundaries (I/O, configs, APIs) before expensive computation
- **Clear Error Messages:** "device must be 'cpu', 'cuda', or 'mps'" vs generic `KeyError`
- **Type Safety:** End-to-end type checking from CLI → Core → Results
- **Self-Documentation:** Schemas serve as living documentation
- **Industry Standard:** Used by FastAPI, Prefect, Dagster, LangChain

**Dependencies:**
```toml
[project.optional-dependencies]
validation = [
    "pydantic>=2.10.0",           # Core validation (Phases 1, 2, 4)
    "pydantic-settings>=2.6.0",   # Settings management (Phase 2)
    "pandera>=0.20.0",            # DataFrame validation (Phase 3)
]
```

**Installation:**
```bash
uv sync --extra validation
```

---

## Phase Overview

| Phase | Name | Priority | Risk | Dependencies | Est. Tests |
|-------|------|----------|------|--------------|------------|
| **1** | [Prediction Hardening](#phase-1-prediction-hardening) | HIGH | LOW | None | 18 |
| **2** | [Configuration Safety](#phase-2-configuration-safety) | HIGH | MEDIUM | Phase 1 | 23 |
| **3** | [Data Integrity](#phase-3-data-integrity) | MEDIUM | MEDIUM | Phases 1, 2 | 16 |
| **4** | [Artifacts & Metrics](#phase-4-artifacts--metrics) | MEDIUM | LOW | Phase 1 | 12 |

**Total estimated tests:** 69

---

## Phase 1: Prediction Hardening

**Goal:** Harden user-facing prediction surfaces (CLI, Gradio, Predictor) with request/response models.

**Benefits:**
- ✅ Fail fast with clear errors before ESM computation
- ✅ Remove duplicated validation logic in `app.py`
- ✅ API-ready models for future FastAPI endpoints

**Scope:**
- Create `PredictionRequest`, `BatchPredictionRequest`, `PredictionResult` models
- Integrate into `core/prediction.py` (maintain backward compat with raw strings)
- Replace `validate_input()` in `cli/app.py` with Pydantic validation
- Update `cli/predict.py` to use request models

**Files Created:**
- `src/antibody_training_esm/models/__init__.py`
- `src/antibody_training_esm/models/prediction.py`
- `tests/unit/models/test_prediction.py`
- `tests/integration/test_prediction_integration.py`

**Files Modified:**
- `src/antibody_training_esm/core/prediction.py`
- `src/antibody_training_esm/cli/app.py` (delete `validate_input()`)
- `src/antibody_training_esm/cli/predict.py`

**Success Criteria:**
- [ ] 18 tests passing
- [ ] Gradio app validates input with Pydantic
- [ ] CLI validates input with Pydantic
- [ ] Predictor accepts both raw strings AND PredictionRequest (backward compat)
- [ ] Invalid sequences raise `ValidationError` (not generic exceptions)

**Documentation:** [PYDANTIC_PHASE_1_PREDICTION_HARDENING.md](PYDANTIC_PHASE_1_PREDICTION_HARDENING.md)

---

## Phase 2: Configuration Safety

**Goal:** Replace manual dictionary config validation with Pydantic models that mirror Hydra structure.

**Benefits:**
- ✅ Catch config errors at startup (before ESM model loading)
- ✅ Type-safe config access: `config.model.device` (IDE autocomplete)
- ✅ Clear enum validation: "device must be 'cpu', 'cuda', or 'mps'"

**Scope:**
- Create config model hierarchy: `ModelConfig`, `DataConfig`, `ClassifierConfig`, `TrainingConfig`, `ExperimentConfig`, `TrainingPipelineConfig`
- Replace `validate_config()` in `trainer.py` with Pydantic validation
- Update all config access to dot notation (`config.model.device`)
- Add self-documenting comments to Hydra YAML files

**Files Created:**
- `src/antibody_training_esm/models/config.py`
- `tests/unit/models/test_config.py`
- `tests/integration/test_config_integration.py`

**Files Modified:**
- `src/antibody_training_esm/core/trainer.py` (replace `validate_config()`)
- `src/antibody_training_esm/data/loaders.py`
- `src/antibody_training_esm/conf/config.yaml` (add comments)

**Success Criteria:**
- [ ] 23 tests passing
- [ ] All Hydra configs validate with Pydantic
- [ ] Invalid device/metrics raise `ValidationError` at startup
- [ ] Missing files raise `FileNotFoundError` at startup
- [ ] Config access uses dot notation throughout codebase
- [ ] Hydra overrides work: `antibody-train model.device=cuda`

**Documentation:** [PYDANTIC_PHASE_2_CONFIGURATION_SAFETY.md](PYDANTIC_PHASE_2_CONFIGURATION_SAFETY.md)

---

## Phase 3: Data Integrity

**Goal:** Replace manual DataFrame validation with Pandera schemas (Pydantic for DataFrames).

**Benefits:**
- ✅ Prevent silent data corruption (like Jain column mismatch incident)
- ✅ Schema-as-code: DataFrame structure explicitly documented
- ✅ Deduplicate ~200 lines of manual validation in `validation_utils.py`

**Scope:**
- Create Pandera schemas: `SequenceDatasetSchema` (base), `BoughterSchema`, `JainSchema`, `HarveySchema`, `ShehataSchema`
- Integrate into `datasets/base.py` (validate on load)
- Refactor `preprocessing/validation_utils.py` (remove manual DataFrame checks)
- Update preprocessing validation scripts

**Files Created:**
- `src/antibody_training_esm/schemas/__init__.py`
- `src/antibody_training_esm/schemas/dataset.py`
- `tests/unit/schemas/test_dataset.py`
- `tests/integration/test_dataset_loading.py`

**Files Modified:**
- `src/antibody_training_esm/datasets/base.py`
- `src/antibody_training_esm/datasets/*.py` (all loaders)
- `preprocessing/validation_utils.py` (remove DataFrame validation functions)
- `preprocessing/*/validate_*.py` (use Pandera)

**Success Criteria:**
- [ ] 16 tests passing
- [ ] All datasets validate with Pandera on load
- [ ] Invalid amino acids raise `SchemaError`
- [ ] Gap characters raise `SchemaError`
- [ ] Missing columns raise `SchemaError`
- [ ] Manual validation functions removed from `validation_utils.py`

**Documentation:** [PYDANTIC_PHASE_3_DATA_INTEGRITY.md](PYDANTIC_PHASE_3_DATA_INTEGRITY.md)

---

## Phase 4: Artifacts & Metrics

**Goal:** Replace manual dict construction in serialization/metrics with Pydantic models.

**Benefits:**
- ✅ Self-describing model artifacts (JSON sidecar with full metadata)
- ✅ Type-safe serialization/deserialization
- ✅ No manual type casting (Pydantic handles `class_weight` int keys)

**Scope:**
- Create artifact models: `ModelArtifactMetadata`, `EvaluationMetrics`, `CVResults`
- Update `save_model()` and `load_model_from_npz()` to use Pydantic
- Update `evaluate_model()` and `perform_cross_validation()` to return Pydantic models

**Files Created:**
- `src/antibody_training_esm/models/artifact.py`
- `tests/unit/models/test_artifact.py`

**Files Modified:**
- `src/antibody_training_esm/core/training/serialization.py`
- `src/antibody_training_esm/core/training/metrics.py`

**Success Criteria:**
- [ ] 12 tests passing
- [ ] `ModelArtifactMetadata` serializes/deserializes correctly
- [ ] `class_weight` with int keys preserved (no JSON string conversion bug)
- [ ] `EvaluationMetrics` constructs from sklearn predictions
- [ ] `CVResults` constructs from sklearn cross_validate output
- [ ] No manual type casting in serialization code

**Documentation:** [PYDANTIC_PHASE_4_ARTIFACTS_METRICS.md](PYDANTIC_PHASE_4_ARTIFACTS_METRICS.md)

---

## Implementation Strategy

### TDD Workflow (Required)

All phases MUST follow strict Test-Driven Development:

```bash
# 1. Write tests FIRST (should FAIL)
uv run pytest tests/unit/models/test_prediction.py -xvs
# ❌ FAIL (models don't exist yet)

# 2. Implement models to make tests PASS
# ... write code ...

uv run pytest tests/unit/models/test_prediction.py -v
# ✅ PASS (all green)

# 3. Integration tests
uv run pytest tests/integration/test_prediction_integration.py -v
# ✅ PASS

# 4. Full test suite
make test
# ✅ PASS
```

### PR Strategy (3 PRs per Phase)

Each phase follows a 3-PR rollout:

**PR 1: Models Only (Low Risk)**
- Add Pydantic models
- Add unit tests
- No integration (just models)
- **Goal:** Establish schemas, verify validation logic

**PR 2: Integration (Medium Risk)**
- Integrate models into core code
- Add integration tests
- Maintain backward compatibility
- **Goal:** Replace manual validation

**PR 3: Cleanup (Low Risk)**
- Remove old validation logic
- Update documentation
- Final polish
- **Goal:** Code deduplication

### Quality Gates (All Phases)

Every PR must pass:
- [ ] `make test` (unit + integration tests)
- [ ] `make lint` (ruff)
- [ ] `make typecheck` (mypy strict)
- [ ] Code coverage ≥70%
- [ ] No new Bandit findings

### Backward Compatibility

**Critical:** All phases maintain backward compatibility during rollout.

**Example (Phase 1):**
```python
# Predictor accepts BOTH raw strings AND PredanticRequest
def predict_single(
    self,
    sequence: str | PredictionRequest,  # Union type
    ...
) -> PredictionResult:
    # Normalize to PydanticRequest internally
    if isinstance(sequence, str):
        request = PredictionRequest(sequence=sequence)
    else:
        request = sequence
    # ... rest of logic ...
```

---

## Directory Structure (Post-Integration)

```
src/antibody_training_esm/
├── models/                    # Pydantic models (Phases 1, 2, 4)
│   ├── __init__.py
│   ├── prediction.py          # Phase 1: Request/response models
│   ├── config.py              # Phase 2: Config models
│   └── artifact.py            # Phase 4: Artifact/metrics models
├── schemas/                   # Pandera schemas (Phase 3)
│   ├── __init__.py
│   └── dataset.py             # Phase 3: DataFrame schemas
├── core/
│   ├── prediction.py          # Modified: Phase 1
│   ├── trainer.py             # Modified: Phase 2
│   └── training/
│       ├── serialization.py   # Modified: Phase 4
│       └── metrics.py         # Modified: Phase 4
├── datasets/
│   ├── base.py                # Modified: Phase 3
│   └── *.py                   # Modified: Phase 3
└── cli/
    ├── app.py                 # Modified: Phase 1
    └── predict.py             # Modified: Phase 1

tests/
├── unit/
│   ├── models/
│   │   ├── test_prediction.py      # Phase 1
│   │   ├── test_config.py          # Phase 2
│   │   └── test_artifact.py        # Phase 4
│   └── schemas/
│       └── test_dataset.py         # Phase 3
└── integration/
    ├── test_prediction_integration.py   # Phase 1
    ├── test_config_integration.py       # Phase 2
    └── test_dataset_loading.py          # Phase 3
```

---

## Rollout Timeline (Suggested)

**Assumption:** 1 week per phase (conservative estimate)

| Week | Phase | Deliverables |
|------|-------|--------------|
| **1** | Phase 1 | Prediction hardening (3 PRs) |
| **2** | Phase 2 | Config safety (3 PRs) |
| **3** | Phase 3 | Data integrity (3 PRs) |
| **4** | Phase 4 | Artifacts & metrics (3 PRs) |

**Total:** 4 weeks, 12 PRs

**Accelerated:** Can run Phases 1+4 in parallel (independent), reducing to 3 weeks.

---

## Risk Mitigation

### Phase 1 (LOW Risk)
- **Risk:** Breaking Gradio app
- **Mitigation:** Keep `validate_input()` until PR 3, maintain dual API

### Phase 2 (MEDIUM Risk)
- **Risk:** Breaking Hydra integration
- **Mitigation:** Test with all config groups, validate overrides work

### Phase 3 (MEDIUM Risk)
- **Risk:** Breaking dataset loading
- **Mitigation:** Test with all 4 datasets, validate fragment loading

### Phase 4 (LOW Risk)
- **Risk:** Model serialization incompatibility
- **Mitigation:** Test with existing model artifacts, maintain JSON format

---

## Success Metrics

### Code Quality
- [ ] **Type Coverage:** 100% (mypy strict)
- [ ] **Test Coverage:** ≥70%
- [ ] **Code Deduplication:** ~400 lines removed (manual validation logic)
- [ ] **Security:** No new Bandit findings

### Developer Experience
- [ ] **IDE Support:** Autocomplete for config/models
- [ ] **Error Messages:** Clear, actionable validation errors
- [ ] **Documentation:** All schemas documented with examples

### Runtime Reliability
- [ ] **Fail Fast:** Config errors caught at startup (not mid-training)
- [ ] **Data Integrity:** DataFrame validation prevents corruption
- [ ] **API Ready:** Request/response models for FastAPI

---

## Non-Goals (Out of Scope)

- ❌ FastAPI endpoint creation (future work)
- ❌ Environment variable overrides via `pydantic-settings` (future)
- ❌ Migration of existing model artifacts (manual if needed)
- ❌ Pydantic v1 support (v2 only)
- ❌ Advanced Pandera features (regex columns, custom parsers)

---

## References

### Documentation
- [Pydantic v2 Docs](https://docs.pydantic.dev/2.10/)
- [Pandera Docs](https://pandera.readthedocs.io/)
- [Hydra + Pydantic Integration](https://hydra.cc/docs/advanced/pydantic/)

### Internal Docs
- [Phase 1: Prediction Hardening](PYDANTIC_PHASE_1_PREDICTION_HARDENING.md)
- [Phase 2: Configuration Safety](PYDANTIC_PHASE_2_CONFIGURATION_SAFETY.md)
- [Phase 3: Data Integrity](PYDANTIC_PHASE_3_DATA_INTEGRITY.md)
- [Phase 4: Artifacts & Metrics](PYDANTIC_PHASE_4_ARTIFACTS_METRICS.md)
- [Original Audit](../../PYDANTIC_INTEGRATION_AUDIT.md)

### Related Work
- [Phase D Refactoring](../archive/plans/PHASE_D_CODE_DEDUPLICATION.md) (shared utilities)
- [Jain Corruption Investigation](../archive/investigations/INVESTIGATION_REPORT_JAIN_CORRUPTION.md) (data integrity motivation)

---

## Appendix: Command Reference

### Install Dependencies
```bash
uv sync --extra validation
```

### Run Phase Tests
```bash
# Phase 1
uv run pytest tests/unit/models/test_prediction.py -v
uv run pytest tests/integration/test_prediction_integration.py -v

# Phase 2
uv run pytest tests/unit/models/test_config.py -v
uv run pytest tests/integration/test_config_integration.py -v

# Phase 3
uv run pytest tests/unit/schemas/test_dataset.py -v
uv run pytest tests/integration/test_dataset_loading.py -v

# Phase 4
uv run pytest tests/unit/models/test_artifact.py -v
```

### Full Quality Check
```bash
make all  # format → lint → typecheck → test
```

### Generate JSON Schema (for docs)
```python
from antibody_training_esm.models.prediction import PredictionRequest
print(PredictionRequest.model_json_schema())
```

---

**Last Updated:** 2025-11-20
**Version:** 1.0.0
**Status:** ✅ Ready for Implementation
