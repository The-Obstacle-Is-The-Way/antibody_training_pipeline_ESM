# Training Pipeline Investigation (2025-11-11)

_Author: Internal QA agent_  
_Scope: `antibody_training_pipeline_ESM` training CLI (Hydra integration + caching)_

## TL;DR
- `antibody-train model=esm2_650m …` silently falls back to the default `esm1v` backbone whenever the CLI wrapper (`antibody_training_esm/cli/train.py`) invokes the Hydra-decorated entry point. Running the trainer module directly (`python -m antibody_training_esm.core.trainer …`) honors overrides, so the wrapper is discarding Hydra config-group choices.  
  _Evidence:_ `outputs/novo_replication/2025-11-11_21-50-06/.hydra/overrides.yaml` contains `model=esm2_650m`, while the rendered config (`.hydra/config.yaml`) still shows `facebook/esm1v_t33_650M_UR90S_1`.
- Embedding cache files (`embeddings_cache/train_7efc852ce835_embeddings.pkl`) are keyed only by sequence hash, so different PLMs reuse the same cache. Training AntiBERTa or ESM2 reuses ESM-1v embeddings and produces bogus comparisons.
- Hydra emits automatic-schema-matching deprecation warnings on every run (`'model/esm1v' is validated against ConfigStore schema with the same name…`). This stems from registering structured configs with the same names as YAML files; Hydra 1.2 will treat this as an error.
- Log directory creation bug already patched (added `log_file.parent.mkdir()` inside the Hydra code path), but it’s worth tracking because previous runs failed silently when Hydra wrote logs relative to the run dir.

The following sections document each issue, reproduction steps, and recommended fixes.

---

## 1. Hydra Config-Group Overrides Ignored in CLI Wrapper

### Symptoms
- `antibody-train model=esm2_650m classifier=logreg` still loads `facebook/esm1v_t33_650M_UR90S_1`.
- `.hydra/overrides.yaml` shows the correct override, yet the composed config (`.hydra/config.yaml`) and runtime logs use the default backbone.
- Overriding individual fields (e.g., `model.name=facebook/esm2_t33_650M_UR50D`) **does** work, so the bug is specific to config-group overrides.

### Reproduction
1. Run `antibody-train --cfg job model=esm2_650m`.
2. Inspect `outputs/novo_replication/<timestamp>/.hydra/config.yaml` → `model.name` remains `facebook/esm1v_t33_650M_UR90S_1`.
3. Run `python -m antibody_training_esm.core.trainer --cfg job model=esm2_650m` → config now shows `facebook/esm2_t33_650M_UR50D`.

### Root Cause
Hydra config-group overrides are processed only when the decorated `@hydra.main` function is invoked directly as the module's entry point. Our console script (`antibody-train` → `antibody_training_esm.cli.train:main`) wraps the Hydra-decorated `core.trainer:main()` through an indirection layer. This causes Hydra to compose defaults first, then pass control to the wrapper, which calls the decorated function—but by that point, config-group overrides have already been skipped. The wrapper also invokes the decorated function via `python -m antibody_training_esm.cli.train`, which exhibits the same behavior (defaults win, overrides are recorded but not applied). Running `python -m antibody_training_esm.core.trainer` directly bypasses the wrapper and allows Hydra to process overrides correctly.

### Fix Recommendation
1. Change the `antibody-train` console script in `pyproject.toml` to point directly at the Hydra-decorated function:
   ```toml
   [project.scripts]
   antibody-train = "antibody_training_esm.core.trainer:main"
   ```
   or,
2. Keep the wrapper but decorate it with Hydra as well, passing overrides through explicitly (`hydra.main(...)(cli_main)`), ensuring Hydra owns argument parsing.
3. Add an integration test that composes the config via the CLI entrypoint and asserts `cfg.model.name` matches the override.

Until this is fixed, instruct users to run `python -m antibody_training_esm.core.trainer` when they need config-group overrides.

---

## 2. Embedding Cache Reused Across Different PLMs

### Symptoms
- After training ESM-1v once, running ESM2 without deleting `embeddings_cache/train_*.pkl` reuses the same embeddings (`hash: 7efc852ce835`), producing identical metrics irrespective of backbone.
- Cache filenames only encode the dataset split hash; the backbone name, revision, pooling strategy, and sequence length are absent.

### Root Cause
`get_or_create_embeddings()` builds the cache key from `hashlib.sha256("".join(sequences))` (trainer.py:305), hashing only the concatenated sequences. The cache filename includes the dataset split name and sequence hash (e.g., `train_7efc852ce835_embeddings.pkl`), but omits the model name, revision, pooling strategy, and max sequence length. Any dataset reuse hits the same file even if the embedding extractor changes. AntiBERTa, ESM2, or future custom PLMs would all reuse the ESM-1v embeddings unless the user manually clears the cache.

### Fix Recommendation
- Expand the cache key to include:
  - `embedding_extractor.model_name`
  - `embedding_extractor.revision`
  - `embedding_extractor.max_length`
  - Any pooling/head parameters that change the embedding tensor.
- Example:
  ```python
  cache_key = hashlib.sha256(
      f"{model_name}|{revision}|{max_length}|{''.join(sequences)}".encode("utf-8")
  ).hexdigest()[:12]  # Truncate to 12 chars like current implementation
  filename = f"{split}_{cache_key}_embeddings.pkl"
  ```
- Store model metadata inside the pickle and assert it matches the current extractor before reusing.

Without this change, benchmarking different PLMs is meaningless because every run silently reuses the first cache.

---

## 3. Hydra Automatic Schema Matching Deprecation Warnings

### Symptoms
Every CLI run logs four warnings like:
```
'model/esm1v' is validated against ConfigStore schema with the same name.
This behavior is deprecated in Hydra 1.1 and will be removed in Hydra 1.2.
```

### Root Cause
We register structured configs with the same names as YAML files:
```python
cs.store(group="model", name="esm1v", node=ModelConfig)
```
and also ship `conf/model/esm1v.yaml`. Hydra 1.0 allowed this implicit matching; 1.1 warns, and 1.2 will error.

### Fix Recommendation
Follow Hydra’s migration guide:
1. Rename schema registrations (e.g., `name="esm_schema"`) and set `@dataclass` defaults accordingly **or**
2. Delete the schema registration and rely on YAML-only configs **or**
3. Keep structured configs but rename YAML files (e.g., `model/base.yaml`, `model/esm1v.yaml` + `defaults` selecting the schema).

Whichever approach we choose, silence the warnings before we upgrade Hydra or the CLI will break.

---

## 4. Log File Creation (Already Patched but Track It)

### Context
Earlier today the Hydra code path failed when writing `logs/training.log` because the directory didn’t exist. We patched this by adding:
```python
log_file.parent.mkdir(parents=True, exist_ok=True)
```
inside the Hydra block (`core/trainer.py:171-175`). Keep this regression test around; without it Hydra runs in production will crash if the log dir is absent.

---

## Action Items
1. **CLI Override Bug**
   - Rewire the console script to call the Hydra entry point directly or redecorate the wrapper.
   - Add integration tests for `model=esm2_650m` and `classifier=xgboost` overrides via the CLI.
2. **Embedding Cache**
   - Incorporate model metadata into the cache key and stored metadata.
   - Add unit tests covering cache reuse with different backbones.
3. **Hydra Warnings**
   - Rename structured config registrations or drop them to comply with the 1.1+ rules.
   - Document the upgrade path in `docs/developer-guide`.
4. **Docs**
   - Update training instructions to mention the current workaround (`python -m antibody_training_esm.core.trainer …`) until the CLI fix lands.

Once these are addressed we can proceed with the Week 1 benchmark confident that Hydra overrides and caching behave correctly.

---

## Implementation Priority

**CRITICAL (blocks correct benchmarking):**
1. **Embedding Cache Fix** - Without this, all backbone comparisons are meaningless
   - Impact: HIGH - produces incorrect benchmark results
   - Effort: LOW - 30 min code change + unit test
   - Location: `trainer.py:305` - update hash to include model metadata

**HIGH (UX regression, but has workaround):**
2. **CLI Override Bug** - Breaks expected Hydra behavior
   - Impact: MEDIUM - users must use workaround `python -m` invocation
   - Effort: LOW - change one line in `pyproject.toml` + integration test
   - Location: `pyproject.toml:76` - point directly to Hydra entry point

**MEDIUM (cosmetic, but will error in future Hydra versions):**
3. **Hydra Schema Warnings** - Deprecation warnings on every run
   - Impact: LOW - warnings only, but will break in Hydra 1.2+
   - Effort: MEDIUM - requires understanding Hydra 1.1+ patterns
   - Location: Multiple - wherever structured configs are registered

**TRACKING ONLY (already fixed):**
4. **Log Directory Creation** - Already patched, keep regression test
   - Status: FIXED in `trainer.py:175`
   - Regression test: Verify log directory creation in CI

---

## Validation Complete

This document has been validated against the actual codebase (2025-11-11). All code locations, line numbers, and behaviors have been verified from first principles by examining the source code directly.
