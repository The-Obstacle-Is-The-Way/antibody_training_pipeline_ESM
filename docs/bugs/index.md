# Bug Backlog (Audit) — 2025-12-14

This doc captures potential bugs and high-risk issues found during a codebase audit.

> **All P1/P2 bugs have been validated from first principles and have TDD specs.**
> See: [`docs/specs/index.md`](../specs/index.md)

## Priority Scale

- **P0**: Critical — data loss, security issue in default usage, or core pipeline unusable.
- **P1**: High — major feature broken or produces incorrect results silently.
- **P2**: Medium — edge-case crashes, incomplete "end-to-end" flows, significant perf traps.
- **P3**: Low — papercuts, doc drift, non-default footguns.
- **P4**: Trivial — cleanup, warning noise, minor refactors.

## Audit Snapshot (What Was Run)

- `make lint`, `make typecheck`, `make test` (fast suite): ✅ pass
- `make test-e2e` (env-gated): ✅ pass (some expected skips)
- `make docs-build`: ✅ builds, many warnings
- `uv run bandit -r src/antibody_training_esm`: ✅ no findings, but warning noise

## Validation Status (2025-12-14)

| Bug | Validation | Spec |
|-----|------------|------|
| P1.1 | ✅ CONFIRMED by code review | [Spec](../specs/P1_1_classifier_type_vs_strategy.md) |
| P1.2 | ✅ CONFIRMED by code review | [Spec](../specs/P1_2_npz_loses_embedding_model_type.md) |
| P2.1-P2.4 | ✅ ALL CONFIRMED | [Spec](../specs/P2_prediction_and_cache_bugs.md) |

---

## P0 — Critical

- None found during this pass (tests + typecheck passed).

---

## Research — Active Investigations

### R1 — Jain Parity Reverse Engineering (59/27 vs Novo's 57/29)

> **Research Spec:** [jain_parity_reverse_engineering.md](./jain_parity_reverse_engineering.md)
> **Data Inventory:** [jain_parity_data_inventory.md](./jain_parity_data_inventory.md)

#### Problem
Our P5e-S2 preprocessing produces 59 specific / 27 non-specific, but Novo's Figure S14A shows 57 specific / 29 non-specific. We are off by 2 antibodies.

| Metric | Ours | Novo | Delta |
|--------|------|------|-------|
| Confusion Matrix | `[[40, 19], [10, 17]]` | `[[40, 17], [10, 19]]` | FP/TP differ by 2 |
| Accuracy | 66.28% | 68.6% | -2.32pp |
| Label Split | 59/27 | 57/29 | ±2 |

#### Key Insight
TN=40 and FN=10 match exactly. The discrepancy is entirely in FP/TP — we have 2 specific antibodies that Novo classifies as non-specific.

#### Research Goals
1. Identify the unknown QC step Novo uses to get from ~116 to 86 antibodies
2. Find which 2 antibodies need reclassification (specific → non-specific)
3. Validate that the solution is biologically principled (not cherry-picking)

#### Data Inventory (from [jain_parity_data_inventory.md](./jain_parity_data_inventory.md))

| Stage | Count | Specific | Non-Specific | File |
|-------|-------|----------|--------------|------|
| After ELISA 1-3 removal | 116 | 94 | 22 | `jain_ELISA_ONLY_116.csv` |
| After reclassification (5) | 116 | 89 | 27 | (computed) |
| After removal (30) — **OURS** | 86 | 59 | 27 | `jain_86_novo_parity.csv` |
| **NOVO TARGET** | 86 | **57** | **29** | Figure S14A |

#### Experimental Phases
1. **Phase 1:** Data preparation — ✅ Complete (see data inventory)
2. **Phase 2:** Systematic permutation search — test all C(59,2) = 1,711 pairs
3. **Phase 3:** Biological validation — evaluate plausibility of candidate pairs
4. **Phase 4:** Alternative strategies — test different ranking/removal methods

#### Branch Strategy
- **Stable reference:** `investigate/jain-parity-verification` (current)
- **Experiments:** Create `experiment/jain-parity-permutations` for testing

#### Status
- [x] Document the discrepancy (GitHub Issue #33)
- [x] Create research spec with hypotheses and experimental protocol
- [x] Create data inventory with all 89 specific antibodies and biophysical data
- [ ] Commit stable investigation branch
- [ ] Create experiment branch for permutation testing
- [ ] Execute Phase 2: Permutation search
- [ ] Execute Phase 3: Biological validation
- [ ] Update preprocessing pipeline with correct methodology

---

## P1 — High

### P1.1 — Hydra classifier selection mismatch (`classifier.type` vs `classifier.strategy`)

> **Spec:** [P1_1_classifier_type_vs_strategy.md](../specs/P1_1_classifier_type_vs_strategy.md)

#### Impact
- `classifier=xgboost` appears to work (configs exist), but the training pipeline can silently fall back to Logistic Regression.
- This is especially dangerous because it looks "successful" while training the wrong model.

#### Evidence
- Hydra configs use `type: ...` (e.g. `src/antibody_training_esm/conf/classifier/xgboost.yaml`).
- Pydantic schema expects `classifier.strategy` (`src/antibody_training_esm/models/config.py`).
- Trainer passes `strategy=...` into `BinaryClassifier(...)` (`src/antibody_training_esm/core/trainer.py`).
- Factory only switches on `config["type"]` (`src/antibody_training_esm/core/classifier_factory.py`).
- Repo-level comment claims this works: `pyproject.toml` under `[project.scripts]`.

#### Likely Symptom / Repro
- Run training with `classifier=xgboost` and inspect saved artifacts: you'd expect a `.xgb` to be created (XGBoost), but you'll get LogReg-style artifacts (e.g. `.npz`) and/or LogReg logs.

#### Suggested Fix
- Unify on **one** key for classifier selection across Hydra + Pydantic + factory:
  - Option A (minimal): treat `classifier.type` as the source of truth and add a Pydantic alias so `strategy` and `type` both work.
  - Option B: rename YAML configs to `strategy:` and teach `create_classifier(...)` to check `strategy` too.
- Add an integration test that composes Hydra config with `classifier=xgboost` and asserts the trained strategy is XGBoost.

---

### P1.2 — NPZ+JSON "production" load path loses embedding model type (breaks AMPLIFY/biophysical)

> **Spec:** [P1_2_npz_loses_embedding_model_type.md](../specs/P1_2_npz_loses_embedding_model_type.md)

#### Impact
- `.npz` + `*_config.json` artifacts cannot be reliably loaded for non-ESM backbones:
  - **biophysical**: will attempt to load a HuggingFace model named `"biophysical"` (fails).
  - **AMPLIFY**: will reconstruct as ESM by default (feature-dimension mismatch or wrong model).

#### Evidence
- `ModelArtifactMetadata.from_classifier(...)` does **not** persist the embedding extractor type (`esm`/`amplify`/`biophysical`) (`src/antibody_training_esm/models/artifact.py`).
- `ModelArtifactMetadata.to_classifier_params()` reconstructs a `BinaryClassifier(...)` without passing `model_type`, so `BinaryClassifier` defaults to `"esm"` (`src/antibody_training_esm/models/artifact.py`, `src/antibody_training_esm/core/classifier.py`).
- `load_model_from_npz(...)` depends on `metadata.to_classifier_params()` (`src/antibody_training_esm/core/training/serialization.py`).

#### Likely Symptom / Repro
- Train a biophysical model, then try to predict using the `.npz` path:
  - `uv run antibody-predict classifier.path=.../model.npz classifier.config_path=.../model_config.json model=biophysical`
  - Expected: uses biophysical features; Actual: tries to load ESM from HF or crashes.

#### Suggested Fix
- Extend metadata to include a dedicated field for the embedding extractor type (e.g. `embedding_model_type: Literal["esm","amplify","biophysical"]`) and round-trip it through `to_classifier_params()`.
- Add regression tests that:
  - Save + load NPZ for `model=biophysical` and `model=amplify_350m`.
  - Assert the reconstructed classifier uses the right embedding extractor and predicts without feature-dim mismatch.

---

## P2 — Medium

> **All P2 specs:** [P2_prediction_and_cache_bugs.md](../specs/P2_prediction_and_cache_bugs.md)

### P2.1 — `Predictor.embedder` recreation ignores biophysical models

#### Impact
- If a biophysical classifier is loaded and the device is overridden/mismatched, the embedder "recreate" path can incorrectly instantiate an `ESMEmbeddingExtractor` instead of a biophysical extractor.

#### Evidence
- Embedder recreation handles `"amplify"` explicitly, otherwise defaults to ESM; no `"biophysical"` branch (`src/antibody_training_esm/core/prediction.py`).

#### Suggested Fix
- Add a biophysical branch that rebuilds `BiophysicalEmbeddingExtractor`, or skip recreation entirely for biophysical.

---

### P2.2 — Predictor does not support pickle-free XGBoost artifacts (`.xgb` + JSON)

#### Impact
- Training can serialize XGBoost to `.xgb` (pickle-free), but inference can't load it without using the pickle fallback.

#### Evidence
- Training writes `.xgb` for XGBoost (`src/antibody_training_esm/core/training/serialization.py`).
- `Predictor.classifier` only supports `.npz` (+ JSON) or pickle/joblib, not `.xgb` (`src/antibody_training_esm/core/prediction.py`).

#### Suggested Fix
- Add a `load_model_from_xgb(...)` path (similar to NPZ) and teach `Predictor` to recognize `.xgb`.

---

### P2.3 — Embedding cache key builds a giant string of all sequences (perf/memory risk)

#### Impact
- `sequences_str = "|".join(sequences)` scales with total dataset length; for large datasets this can be slow and memory-intensive (and can blow up for 100k+ sequences).

#### Evidence
- Cache key construction concatenates the full corpus (`src/antibody_training_esm/core/training/cache.py`).

#### Suggested Fix
- Hash incrementally (streaming) instead of joining into one huge string.

---

### P2.4 — Training pipeline doesn’t use `data.test_file` (docs imply test-set evaluation)

#### Impact
- `TrainingPipelineConfig` requires `test_file`, and docs show "Test Set (Jain)" metrics as part of training output, but training currently only loads/evaluates training data + CV.

#### Evidence
- `train_pipeline(...)` only calls `load_data(config)` (train) and never loads `config.data.test_file` (`src/antibody_training_esm/core/trainer.py`).
- Training docs describe test-set metrics in "Training Output" (`docs/user-guide/training.md`).

#### Suggested Fix
- Either:
  - Implement test-set evaluation inside `train_pipeline(...)`, or
  - Update docs/config schema to make it clear testing is a separate CLI (`antibody-test`) and `test_file` isn’t required for training.

---

## P3 — Low

### P3.1 — Default model revisions are unpinned (`revision: main`)

#### Impact
- "Same config" may not reproduce the same embeddings/results over time if upstream model cards change.

#### Evidence
- `revision: main` in default ESM configs (`src/antibody_training_esm/conf/model/esm1v.yaml`, `src/antibody_training_esm/conf/model/esm2_650m.yaml`).

#### Suggested Fix
- Pin to known-good SHAs (like AMPLIFY config does), or document that ESM defaults are not pinned.

---

### P3.2 — MkDocs build produces many warnings (broken links + nav drift)

#### Impact
- Docs build succeeds, but link integrity is noisy; broken links reduce usability and can hide real doc regressions.

#### Evidence
- `make docs-build` emits many warnings about missing targets and pages not included in nav.

#### Suggested Fix
- Decide which docs should be in nav vs archive-only, then fix/convert problematic relative links (often needs absolute repo links).

---

### P3.3 — `antibody-preprocess` CLI is a stub (not end-to-end)

#### Impact
- Users may assume preprocessing is supported via CLI, but it only prints guidance.

#### Evidence
- CLI explicitly states it is not implemented (`src/antibody_training_esm/cli/preprocess.py`).

#### Suggested Fix
- Either implement a Hydra-driven preprocess pipeline (preferred per repo guidelines), or rename/reframe the CLI to avoid implying functionality.

---

## P4 — Trivial / Cleanup

### P4.1 — Bandit `# nosec ...` annotations generate warning noise

#### Impact
- `bandit` run is noisy due to `# nosec` comments that include extra words; this can mask real findings in CI output.

#### Evidence
- `uv run bandit -r src/antibody_training_esm` warns about "Test in comment: ...".

#### Suggested Fix
- Standardize to `# nosec: B615` (or `# nosec B615`) with no extra tokens.

---

### P4.2 — E2E skip reasons appear stale/misleading

#### Impact
- Skipped tests mention "trainer not implemented" even though trainer exists; this can confuse future debugging.

#### Evidence
- `make test-e2e` shows skip reasons in `tests/e2e/test_train_pipeline.py`.

#### Suggested Fix
- Update skip conditions/messages to match current pipeline status and gating env vars.

---

### P4.3 — `Predictor` uses `.__code__` introspection to detect `assay_type`

#### Impact
- Works for pure-Python callables, but can break for non-Python implementations of `predict` (future-proofing issue).

#### Evidence
- `src/antibody_training_esm/core/prediction.py` checks `self.classifier.predict.__code__.co_varnames`.

#### Suggested Fix
- Use `inspect.signature(...)` with a try/except, or just try calling with `assay_type` and fall back on `TypeError`.

