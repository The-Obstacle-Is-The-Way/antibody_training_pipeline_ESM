# Bug Fix Specifications

This directory contains TDD specifications for validated bugs from the codebase audit.

## Audit Summary (2025-12-14)

All bugs listed in `docs/bugs/index.md` were validated from first principles by examining source code.

| Bug ID | Priority | Status | Root Cause | Spec |
|--------|----------|--------|------------|------|
| P1.1 | **HIGH** | ✅ CONFIRMED | Hydra `type:` vs Pydantic `strategy:` | [P1_1_classifier_type_vs_strategy.md](P1_1_classifier_type_vs_strategy.md) |
| P1.2 | **HIGH** | ✅ CONFIRMED | NPZ metadata missing `embedding_model_type` | [P1_2_npz_loses_embedding_model_type.md](P1_2_npz_loses_embedding_model_type.md) |
| P2.1 | MEDIUM | ✅ CONFIRMED | Predictor ignores biophysical in embedder recreation | [P2_prediction_and_cache_bugs.md](P2_prediction_and_cache_bugs.md#p21-predictor-embedder-recreation-ignores-biophysical) |
| P2.2 | MEDIUM | ✅ CONFIRMED | Predictor can't load `.xgb` artifacts | [P2_prediction_and_cache_bugs.md](P2_prediction_and_cache_bugs.md#p22-predictor-doesnt-support-xgb-artifacts) |
| P2.3 | MEDIUM | ✅ CONFIRMED | Cache key `"|".join(sequences)` memory risk | [P2_prediction_and_cache_bugs.md](P2_prediction_and_cache_bugs.md#p23-cache-key-builds-giant-string-memory-risk) |
| P2.4 | MEDIUM | ✅ CONFIRMED | Training ignores `test_file` config | [P2_prediction_and_cache_bugs.md](P2_prediction_and_cache_bugs.md#p24-training-pipeline-doesnt-use-test_file) |

## Validation Methodology

Each bug was validated by:

1. **Reading the actual source code** - Not just the audit report
2. **Tracing data flow** - From config to execution
3. **Identifying root cause** - Why the bug occurs
4. **Writing TDD tests** - Tests that will fail before the fix
5. **Documenting the fix** - Precise code changes needed

## Specification Format

Each spec follows this structure:

```
# Bug ID: Title

## Root Cause Analysis
- What happens
- Why it happens
- Data flow diagram

## Affected Files
- Table of files and their roles

## TDD Test Specifications
- Test code that validates the fix
- Tests should FAIL before fix, PASS after

## Implementation Fix
- Precise code changes
- Options if multiple approaches exist

## Acceptance Criteria
- Checklist for done
```

## Fix Priority Order

Recommended order based on impact and dependency:

### Phase 1: Critical Fixes (P1)
1. **P1.1** - Classifier type/strategy mismatch (silent wrong model)
2. **P1.2** - NPZ model type (breaks AMPLIFY/biophysical)

### Phase 2: Medium Fixes (P2)
3. **P2.1** - Predictor biophysical support (depends on P1.2)
4. **P2.2** - XGBoost loading (depends on P1.1 working)
5. **P2.3** - Cache memory optimization (independent)
6. **P2.4** - test_file usage clarification (docs/design decision)

### Phase 3: Low Priority (P3-P4)
- See `docs/bugs/index.md` for P3/P4 items
- Mostly doc fixes and cleanup

## Running Tests

After implementing fixes, run:

```bash
# Run all tests including new bug-fix tests
make test

# Run specific test file
uv run pytest tests/unit/models/test_config_classifier_type.py -v
uv run pytest tests/unit/models/test_artifact_model_type.py -v
uv run pytest tests/integration/test_classifier_selection.py -v
```

## Contributing

When adding new specs:

1. Follow the template format
2. Include TDD tests that fail before the fix
3. Reference affected files with line numbers
4. Update this index

---

**Spec Directory Created:** 2025-12-14
**Last Updated:** 2025-12-14
