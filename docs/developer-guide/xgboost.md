# XGBoost Classifier Guide

**Status:** Production ready (Phase 1 complete)
**Audience:** Developers extending/operating classifier backends
**Last Updated:** 2025-11-19

---

## 1. Overview

XGBoost is the second supported classifier backend for the antibody polyreactivity pipeline. It plugs into the existing `ClassifierStrategy` interface so you can swap between Logistic Regression and XGBoost (and future heads) without changing training or inference code.

**Why XGBoost?**
- Learns **nonlinear decision boundaries** that Logistic Regression cannot capture
- Adds model diversity for the roadmap's **3×3 model grid** (backbones × classifiers)
- Ships with **native serialization (.xgb)** for pickle-free production deployments

The implementation landed in v0.5.0 and is validated by the `2025-11-19` XGBoost branch audit (`docs/archive/audits/2025-11-19-xgboost-branch-audit.md`).

---

## 2. Architecture

```
src/antibody_training_esm/core/
├── classifier_strategy.py      # Protocol (fit/predict/predict_proba/...)
├── classifier_factory.py       # Instantiates strategies from Hydra config
├── strategies/
│   ├── logistic_regression.py  # Existing baseline
│   └── xgboost_strategy.py     # NEW: wraps xgboost.XGBClassifier
└── trainer.py                  # Persists native .xgb checkpoint + metadata
```

Key points:
- `XGBoostStrategy` implements every method in `ClassifierStrategy` (fit, predict, predict_proba, score, save/load helpers).
- The strategy **never hardcodes defaults**. All hyperparameters flow from Hydra config so YAML remains the single source of truth (see [`conf/classifier/xgboost.yaml`](../../src/antibody_training_esm/conf/classifier/xgboost.yaml)).
- `classifier_factory.create_classifier()` chooses the strategy based on `classifier.type`. Example:
  ```python
  classifier_cfg = OmegaConf.to_container(cfg.classifier, resolve=True)
  classifier = create_classifier(classifier_cfg)
  ```
- `trainer.py` detects `XGBoostStrategy` and saves both the pickle artifact (`.pkl`) and the **native `.xgb` file plus JSON metadata** for production.

---

## 3. Configuration & Usage

### Hydra Config

`src/antibody_training_esm/conf/classifier/xgboost.yaml`:
```yaml
type: xgboost
n_estimators: 100
max_depth: 6
learning_rate: 0.3
subsample: 1.0
colsample_bytree: 1.0
reg_alpha: 0.0
reg_lambda: 1.0
random_state: ${training.random_state}
objective: binary:logistic
```

To train with XGBoost:
```bash
uv run antibody-train classifier=xgboost  # uses defaults above

# Override hyperparameters inline
uv run antibody-train \
  classifier=xgboost \
  classifier.n_estimators=200 \
  classifier.max_depth=8

# Hyperparameter sweep
uv run antibody-train --multirun \
  classifier=xgboost \
  classifier.learning_rate=0.1,0.3 \
  classifier.subsample=0.8,1.0
```

The inference CLI requires **no changes**—it reads classifier metadata from the checkpoint and reconstructs the correct strategy automatically.

### CLI Outputs

After training, checkpoints live under `experiments/checkpoints/<model>/<classifier>/` (see `docs/developer-guide/directory-organization.md`). For XGBoost each run emits:
- `<dataset>_<fragment>_<model>_xgboost.pkl` – Standard pipeline checkpoint (embedding cache + classifier)
- `<dataset>_<fragment>_<model>_xgboost.xgb` – Native booster (pickle-free)
- `<dataset>_<fragment>_<model>_xgboost.json` – Hyperparameter metadata for production reloads

Example snippet from `trainer.py` (`save_classifier()`):
```python
if isinstance(classifier, XGBoostStrategy):
    classifier.classifier.save_model(xgb_path)
    saved_paths["xgb"] = str(xgb_path)
```

---

## 4. Testing & Validation

| Layer        | Tests                                                                           |
|--------------|---------------------------------------------------------------------------------| 
| Unit         | `tests/unit/core/strategies/test_xgboost_strategy.py` (fit/predict/proba/save)  |
| Integration  | `tests/integration/test_xgboost_integration.py` (trainer + Hydra configs)       |
| Lightweight  | `tests/integration/test_xgboost_e2e_lightweight.py` (Boughter subset)           |
| Audit        | `docs/archive/audits/2025-11-19-xgboost-branch-audit.md` (518 tests, 90% cov)   |

To run targeted suites:
```bash
uv run pytest tests/unit/core/strategies/test_xgboost_strategy.py
uv run pytest tests/integration/test_xgboost_integration.py -m "not gpu"
```

XGBoost unit/integration suites run inside `make test` / `make all`. Use
`make test-all` if you need the full pytest run in one command.

---

## 5. Extending or Debugging

1. **Adding new hyperparameters**
   - Update `conf/classifier/xgboost.yaml` and pass through to `XGBoostStrategy.__init__`.
   - Ensure you persist/load the parameter via `to_dict()` / `from_dict()` to keep `.json` metadata in sync.

2. **GPU vs CPU**
   - XGBoost automatically detects CUDA if compiled with GPU support. Control via the `tree_method` or `device` entries in the config.

3. **Serialization**
   - Prefer `.xgb + JSON` for production because it avoids Python pickle compatibility risks.
   - Pickle artifact is kept for parity with Logistic Regression and for backward compatibility with existing notebooks.

4. **Debugging training**
   - Reuse the `tests/integration/test_xgboost_integration.py::test_end_to_end_training(tmp_path)` fixture to reproduce issues with a small synthetic dataset.
   - Enable verbose logging via `hydra.verbose=true` or `HYDRA_FULL_ERROR=1` for config tracing.

---

## 6. Historical Documents

The original planning artifacts now live in the archive for posterity:
- `docs/archive/plans/xgboost-integration-spec.md`
- `docs/archive/plans/xgboost-api-design.md`
- `docs/archive/plans/xgboost-test-plan.md`
- `docs/archive/plans/xgboost-implementation-status.md`

Consult those if you need deep rationale, but treat this file as the **source of truth** for the implemented XGBoost backend.
