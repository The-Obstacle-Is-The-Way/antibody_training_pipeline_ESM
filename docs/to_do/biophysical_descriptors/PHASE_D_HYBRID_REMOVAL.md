# Phase D: Hybrid Model Removal (Scope Creep Cleanup)

**Date**: 2025-11-28
**Status**: COMPLETED
**Parent**: [BIOPHYSICAL_IMPLEMENTATION_SPECS.md](BIOPHYSICAL_IMPLEMENTATION_SPECS.md)

---

## 1. Executive Summary

**Phase C was a mistake.** It implemented a "hybrid" model (ESM + Biophysical) that:

1. **Is NOT in the Novo Nordisk paper** - They ran Track A (PLM) and Track B (descriptors) as **separate parallel experiments**, never combined
2. **Has no scientific justification** - Adding 3 features to 1280 is noise
3. **Doesn't help Novo parity** - We're trying to replicate the paper, not invent new models
4. **Is scope creep** - We got carried away without checking if it made sense

This document specifies the cleanup to restore alignment with Sakhnini et al. 2025.

---

## 2. What Novo Nordisk Actually Did

From the paper (Figure 2C, Section 2.5):

| Model | Features | 10-fold CV Accuracy |
|-------|----------|---------------------|
| ESM-1v VH LogReg | 1280-d embeddings | ~71% |
| All 68 Descriptors | 68-d biophysical | ~68% |
| Top 5 Descriptors | 5-d biophysical | ~67% |
| Top 2 Descriptors | 2-d biophysical | ~66% |
| pI Alone | 1-d biophysical | 65.2% |

**These are SEPARATE MODELS being compared as bar charts. They are NOT combined.**

The paper's purpose was to answer:
- "Can we predict non-specificity with PLMs?" → Yes, 71%
- "What biophysical properties drive non-specificity?" → pI is dominant
- "How do interpretable baselines compare?" → Close (65-68% vs 71%)

**The paper NEVER asks: "What if we fuse ESM + biophysical?"** Because that's not scientifically interesting - ESM likely already encodes pI implicitly.

---

## 3. What We Built (The Mistake)

### Phase C Added:
- `src/antibody_training_esm/conf/features/hybrid.yaml` - Config to enable hybrid mode
- `src/antibody_training_esm/conf/features/default.yaml` - Flag `use_biophysical: false`
- `trainer.py:166-181` - Concatenation logic for hybrid features
- `tests/integration/test_hybrid_pipeline.py` - Tests for hybrid mode
- `src/antibody_training_esm/models/config.py` - `FeaturesConfig` Pydantic model

### The Problem:
```python
# trainer.py lines 166-181 (THE SLOP)
if config.features.use_biophysical:
    from antibody_training_esm.core.biophysical import BiophysicalExtractor
    bio_extractor = BiophysicalExtractor()
    X_bio = bio_extractor.extract_batch_features(X_train)
    X_train_embedded = np.concatenate([X_train_embedded, X_bio], axis=1)
    # Result: (n, 1283) instead of (n, 1280)
```

This creates a model that:
- Is neither Track A nor Track B
- Has no corresponding baseline in the paper
- Cannot be validated against any published result

---

## 4. What We Should Keep

### Track A (ESM-only) - KEEP
- `antibody-train` CLI with default config
- ESM-1v embeddings → LogReg → ~71% CV
- This matches Novo paper exactly

### Track B (Biophysical-only) - KEEP
- `reproduce_track_b.py` standalone script
- BiophysicalExtractor (3 descriptors) → LogReg → ~63% CV
- This provides interpretability (pI coefficient visible)
- Partial parity with paper (3/68 descriptors, missing Schrödinger)

### BiophysicalExtractor module - KEEP
- `src/antibody_training_esm/core/biophysical.py`
- Used by Track B standalone
- May be useful for future research (not hybrid, but standalone analysis)

---

## 5. Removal Plan

### 5.1 Files to DELETE

```
src/antibody_training_esm/conf/features/hybrid.yaml
src/antibody_training_esm/conf/features/default.yaml
tests/integration/test_hybrid_pipeline.py
```

After deleting both yaml files, the entire `conf/features/` directory can be removed.

### 5.2 Files to MODIFY

**`src/antibody_training_esm/core/trainer.py`**:
```python
# DELETE lines 166-181 (the hybrid concatenation block)
# if config.features.use_biophysical:
#     ...
```

**`src/antibody_training_esm/models/config.py`**:
```python
# DELETE the FeaturesConfig class (lines 283-297):
class FeaturesConfig(BaseModel):
    """..."""
    use_biophysical: bool = Field(...)

# DELETE the features field from TrainingPipelineConfig (line 312):
features: FeaturesConfig = Field(default_factory=FeaturesConfig)
```

**`src/antibody_training_esm/conf/config.yaml`**:
```yaml
# REMOVE the features config group from defaults
defaults:
  - model: esm1v
  - classifier: logreg
  - data: boughter_jain
  # - features: default  # DELETE THIS LINE
  - hardware: default
  - hydra: default
```

**`src/antibody_training_esm/cli/reproduce_track_b.py`**:
```python
# UPDATE comments that reference "Phase C hybrid pipeline"
# Example: "The Phase C hybrid pipeline (trainer.py) defaults to NO scaling"
# Change to: "The main pipeline (trainer.py) does not use scaling"
```

### 5.3 FALSE POSITIVES (Do NOT modify)

**`src/antibody_training_esm/datasets/boughter.py`**:
- Contains "hybrid translation strategy" - this refers to DNA→protein translation
- UNRELATED to Phase C hybrid model
- Leave unchanged

### 5.4 Documentation Updates

- Update `CLAUDE.md` to remove `features=hybrid` mentions
- Update `docs/` if any hybrid references exist
- Archive Phase C spec with "DEPRECATED" header

---

## 6. Validation After Removal

```bash
# Verify Track A still works
uv run antibody-train

# Verify Track B still works
uv run python -m antibody_training_esm.cli.reproduce_track_b

# Run full test suite
make test

# Verify no references to hybrid remain
grep -r "hybrid" src/
grep -r "use_biophysical" src/
```

---

## 7. Lessons Learned

1. **Always check the paper before implementing** - Phase C was built without verifying Novo actually did this
2. **Scope creep is real** - "Hey let's also try combining them" led to dead-end code
3. **First principles matter** - Would fusing 3 features into 1280 actually help? (No)
4. **Parity means parity** - If we claim "Novo replication", we should replicate, not extend

---

## 8. Acceptance Criteria

- [ ] `src/antibody_training_esm/conf/features/` directory deleted (both yaml files)
- [ ] `tests/integration/test_hybrid_pipeline.py` deleted
- [ ] Hybrid concatenation code removed from `trainer.py` (lines 166-181)
- [ ] `FeaturesConfig` class removed from `models/config.py`
- [ ] `features` field removed from `TrainingPipelineConfig` in `models/config.py`
- [ ] `features: default` removed from `conf/config.yaml` defaults
- [ ] Comments in `reproduce_track_b.py` updated (remove Phase C references)
- [ ] All tests pass (`make test`)
- [ ] Track A (`antibody-train`) works unchanged
- [ ] Track B (`reproduce_track_b.py`) works unchanged
- [ ] No grep hits for "use_biophysical" in `src/` (except boughter.py false positive)
- [ ] Phase C spec marked as DEPRECATED (already done)