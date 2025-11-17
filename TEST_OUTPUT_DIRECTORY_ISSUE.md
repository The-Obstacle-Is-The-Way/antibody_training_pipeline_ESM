# Test Output Directory Issue - Classifier Shortname Resolves to `unknown`

**Date:** 2025-11-16
**Status:** OPEN (blocking perfect benchmark organization)

## Summary
Running `uv run antibody-test` after the cleanup stores dataset-specific artifacts under
`experiments/benchmarks/esm1v/unknown/<dataset>/`. The classifier segment resolves to `unknown`
instead of `logreg`, even though the trained model is LogisticRegression. Root-level aggregated files
still appear under `experiments/benchmarks/` as expected.

## Observed Behavior
```
experiments/benchmarks/
├── README.md
├── confusion_matrix_boughter_vh_esm1v_logreg_VH_only_shehata.png         # aggregated output (expected)
├── detailed_results_boughter_vh_esm1v_logreg_VH_only_shehata_*.yaml      # aggregated output
├── predictions_boughter_vh_esm1v_logreg_VH_only_shehata_*.csv            # aggregated output
├── test_20251116_192311.log                                              # CLI log (dataset aggregate)
├── esm1v/
│   └── unknown/
│       └── VH_only_shehata/
│           ├── confusion_matrix_boughter_vh_esm1v_logreg_VH_only_shehata.png
│           ├── detailed_results_boughter_vh_esm1v_logreg_VH_only_shehata_*.yaml
│           └── predictions_boughter_vh_esm1v_logreg_VH_only_shehata_*.csv
└── novo_parity/
```
Hierarchical directories exist, but the classifier folder is `unknown/` instead of `logreg/`.

## Expected Behavior
Per `src/antibody_training_esm/core/directory_utils.py:get_hierarchical_test_results_dir`,
results should land in:
```
experiments/benchmarks/esm1v/logreg/<dataset>/
```
matching the model checkpoint hierarchy (`experiments/checkpoints/esm1v/logreg/`).

## Root Cause
`src/antibody_training_esm/cli/test.py::_compute_output_directory` reads the model metadata file
(`experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg_config.json`) and passes
`model_config.get("classifier", {})` to `extract_classifier_shortname`. Our `save_model()` helper writes
JSON metadata that lacks a `classifier` block entirely, so that call always receives `{}`:
```
{
  "model_type": "LogisticRegression",
  ...
  "device": "cpu"
}
```
Consequently `classifier_config = {}` and `extract_classifier_shortname()` returns `"unknown"`.

Note: The flat files (`confusion_matrix_*`, `detailed_results_*`, `predictions_*`, `test_*.log`) live in
`experiments/benchmarks/` regardless of hierarchical routing—`ModelTester` writes aggregated summaries to
the base output directory by design. Those files are correct; the only mis-placed items are the per-model
subdirectories that should be `logreg/` instead of `unknown/`.

## Recommended Fix
1. Update `save_model()` in `src/antibody_training_esm/core/trainer.py` to include the classifier
   configuration when writing `<model>_config.json`, e.g.:
   ```json
   {
     "model_name": "facebook/esm1v_t33_650M_UR90S_1",
     "classifier": {"type": "logistic_regression", "C": 1.0, ...},
     ...
   }
   ```
   (Hydra config already has `config["classifier"]`; serialize that block.)
2. Ensure the JSON also records `model_name` (not just `esm_model`) so `_compute_output_directory`
   can always find the canonical HuggingFace identifier.
3. Retrain or regenerate metadata for existing checkpoints (or provide migration script) so stored
   JSON files include the `classifier` block.

## Validation Steps Once Fixed
1. Retrain production model or patch metadata file by hand.
2. Run `uv run antibody-test --config configs/testing/jain_p5e_s2.yaml` and check that dataset
   artifacts now live in `experiments/benchmarks/esm1v/logreg/VH_only_jain_86_p5e_s2/`.
3. Repeat for Shehata/Harvey to ensure all outputs consistently use `logreg`.

---
**Impact:** Purely organizational—metrics are correct, but benchmark directories no longer match the
canonical `{model}/{classifier}/{dataset}` spec. Fix before tagging the validated release so the repo
stays clean.

---

## CURRENT STATE ASSESSMENT (2025-11-16 Post-Cleanup Validation)

### Harvey Status
- ⏳ **STILL RUNNING** - Background process extracting embeddings for 141k sequences
- Log: `experiments/runs/tests/harvey/test_20251116_192348.log`
- Will output to: `experiments/runs/tests/harvey/esm1v/unknown/VHH_only_harvey/` (bug confirmed)

### Directory Structure Confusion - IS THIS PROFESSIONAL?

**Current Reality:**
```
experiments/
├── runs/                           # Ad-hoc/sandbox outputs (Hydra multirun, test configs)
│   └── tests/
│       ├── jain/                   # Custom config specified output_dir
│       │   ├── esm1v/unknown/VH_only_jain_86_p5e_s2/  ⚠️ BUG: should be logreg/
│       │   ├── confusion_matrix_*.png
│       │   └── test_*.log
│       └── harvey/                 # CLI flag --output-dir specified
│           └── esm1v/unknown/VHH_only_harvey/         ⚠️ BUG: should be logreg/
│
├── benchmarks/                     # Default test output location (canonical results)
│   ├── esm1v/unknown/VH_only_shehata/                 ⚠️ BUG: should be logreg/
│   ├── confusion_matrix_*.png
│   └── test_*.log
│
├── checkpoints/
│   └── esm1v/logreg/              ✅ CORRECT hierarchy for models
└── cache/                          ✅ Embedding caches (correct)
```

**The Two Problems:**
1. **Outputs scattered across `runs/` vs `benchmarks/`** - Is this intentional design or confusion?
2. **ALL test outputs have `unknown/` bug** - Missing classifier metadata in model JSON

### Model Metadata Inspection

**Current JSON** (`experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg_config.json`):
```json
{
  "model_type": "LogisticRegression",
  "C": 1.0,
  "penalty": "l2",
  "solver": "lbfgs",
  ...
  "esm_model": "facebook/esm1v_t33_650M_UR90S_1"
}
```
❌ **MISSING:** `classifier` block with `type` field
❌ **MISSING:** `model_name` field (has `esm_model` instead)

**What `test.py` expects:**
```json
{
  "model_name": "facebook/esm1v_t33_650M_UR90S_1",
  "classifier": {
    "type": "logistic_regression",
    "C": 1.0,
    ...
  }
}
```

### Validation Uncertainty - CAN WE TRUST THESE RESULTS?

**Jain validation (66.28% accuracy):**
- ✅ Timestamp: 2025-11-16 19:22:58 (fresh, not cached)
- ✅ Files match today's run
- ⚠️ **BUT:** Outputs in `runs/tests/jain/esm1v/unknown/` - wrong hierarchy
- ❓ **Question:** Should we delete and re-validate with FIXED structure to prove it works?

**Shehata validation (52.51% accuracy):**
- ✅ Metrics look correct
- ⚠️ Outputs in `benchmarks/esm1v/unknown/` - wrong hierarchy

**Harvey validation:**
- ⏳ Still running, will have same `unknown/` bug

---

## DEEPMIND-STYLE REMEDIATION PLAN

### Option A: "Fix First, Then Validate Clean" (RECOMMENDED)
**Philosophy:** Don't validate with broken infrastructure. Fix the bug, then get pristine results.

#### Phase 1: Fix Model Metadata (15 min)
1. **Read** `src/antibody_training_esm/core/trainer.py::save_model()` to understand current JSON serialization
2. **Update** `save_model()` to include:
   - `model_name` field (not just `esm_model`)
   - `classifier` block with `type`, `C`, `penalty`, etc.
3. **Patch existing checkpoint metadata** or retrain model:
   - Quick fix: Manually edit `boughter_vh_esm1v_logreg_config.json` to add missing fields
   - Clean fix: Re-run `uv run antibody-train` to regenerate checkpoint with new JSON format
4. **Verify** JSON has correct structure: `cat experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg_config.json`

#### Phase 2: Clean Slate - Delete All Test Outputs (2 min)
```bash
# Kill Harvey background process
pkill -f "antibody-test.*harvey"

# Delete all validation outputs (they have the bug anyway)
rm -rf experiments/runs/tests/jain
rm -rf experiments/runs/tests/harvey
rm -rf experiments/benchmarks/esm1v/unknown
rm -f experiments/benchmarks/confusion_matrix_*
rm -f experiments/benchmarks/detailed_results_*
rm -f experiments/benchmarks/predictions_*
rm -f experiments/benchmarks/test_*.log
```

#### Phase 3: Standardize Output Directory (5 min)
**Decision:** Use `experiments/benchmarks/` for ALL canonical test results
- Update `configs/testing/jain_p5e_s2.yaml`: Change `output_dir: experiments/benchmarks`
- Use default output dir for Shehata/Harvey (already `experiments/benchmarks/`)
- Reserve `experiments/runs/` ONLY for Hydra training multirun outputs

#### Phase 4: Re-Validate All Three Datasets (30 min)
```bash
# Jain - Novo parity benchmark
uv run antibody-test --config configs/testing/jain_p5e_s2.yaml

# Shehata - PSR test
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --data data/test/shehata/fragments/VH_only_shehata.csv

# Harvey - Nanobody test (141k sequences, takes ~20 min)
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --data data/test/harvey/fragments/VHH_only_harvey.csv
```

#### Phase 5: Verify Clean Structure (2 min)
```bash
# Expected output
tree experiments/benchmarks/esm1v/logreg -L 2
# Should show:
# experiments/benchmarks/esm1v/logreg/
# ├── VH_only_jain_86_p5e_s2/
# ├── VH_only_shehata/
# └── VHH_only_harvey/
```

**Total Time:** ~55 minutes
**Outcome:** Clean, professional directory structure with verified results

---

### Option B: "Ship It Now, Fix Later" (NOT RECOMMENDED)
- Keep current outputs as-is
- Document `unknown/` as known issue
- Tag v0.5.0 with caveat
- Fix in v0.5.1

**Why this is bad:**
- Violates "canonical structure" design principle
- Makes benchmarks harder to navigate
- First impression of repo is "messy"
- No confidence that validation actually worked correctly

---

## DECISION CRITERIA

**Go with Option A if:**
- ✅ You want Google DeepMind-level rigor
- ✅ You value clean, navigable benchmark outputs
- ✅ You want to PROVE validation works end-to-end
- ✅ 55 minutes is acceptable for peace of mind

**Go with Option B if:**
- ❌ You're okay with technical debt
- ❌ You trust the metrics despite wrong directory names
- ❌ You want to ship v0.5.0 TODAY

---

## RECOMMENDATION

**STOP. FIX. VALIDATE CLEAN.**

The current `unknown/` bug undermines confidence in the entire validation process. Even though the
metrics are likely correct, the broken directory structure suggests the pipeline isn't fully debugged.

**Next Steps:**
1. Wait for Harvey to finish (or kill it - we'll re-run anyway)
2. Implement Option A: Fix metadata, delete outputs, re-validate clean
3. Document final validated structure in `VALIDATION_ROADMAP.md`
4. Tag v0.5.0 with confidence

---
**Updated:** 2025-11-16 Post-Cleanup Assessment
**Verdict:** Current validation outputs are technically correct but organizationally broken. Re-validate after fix.
