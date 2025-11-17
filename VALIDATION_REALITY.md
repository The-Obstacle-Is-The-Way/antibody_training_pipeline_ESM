# Validation Reality Check - 2025-11-16

## ⚠️ EXECUTION ORDER: Clean FIRST, Validate SECOND

**Decision**: Execute V0.5.0_CLEANUP_PLAN.md Problem 1 (remove `configs/config.yaml` + `train_model()`) BEFORE validation.

**See**: `CLEANUP_BEFORE_VALIDATION_DECISION.md` for full evidence-based reasoning.

**Why**: Validation commands use Hydra (`uv run antibody-train`) and TestConfig (`uv run antibody-test`), NOT legacy `configs/config.yaml`. We validate the production state, not transitional state.

---

## ✅ What’s Already Validated (Post–Option A Fix)

- **Testing (ESM1v/logreg)**: Jain 66.28%, Shehata 52.51%, Harvey ~59% now stored under `experiments/benchmarks/esm1v/logreg/<dataset>/` (no `unknown/`).
- **Metadata Fix**: `save_model()` writes `model_name` + `classifier` block; checkpoint JSON patched.
- **Output Roots**: All benchmark tests now use `experiments/benchmarks/`.

## ⏭️ Still Outstanding

- **Training rerun**: `uv run antibody-train` on the cleaned branch to regenerate the checkpoint with the new JSON writer (Phase 2).
- **Hyperparameter sweep smoke test**: minimal `--multirun` to confirm outputs land in `experiments/runs/<name>/multirun/…` (Phase 4).
- **Fresh clone smoke**: clone → `uv sync` → train → test → `pytest` (Phase 6).
- **ESM2**: not in scope for the current validation pass (only ESM1v validated).

## What Actually Exists After Refactoring

### ✅ Training Pipeline (COMPLETE)
- **Hydra configs**: `src/antibody_training_esm/conf/config.yaml` (+ imports)
- **Command**: `uv run antibody-train`
- **Works**: YES - model already trained at `experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl`
- **Validated**: Nov 15, 2025 (POST_MIGRATION_VALIDATION_SUMMARY.md)

### ✅ Test CLI (EXISTS)
- **Location**: `src/antibody_training_esm/cli/test.py`
- **Flags**:
  - `--model <pkl>` - Model file
  - `--data <csv>` - Dataset file
  - `--config <yaml>` - Config file (for column mapping)
  - `--output-dir` - Output directory
  - `--device` - Device override
  - `--batch-size` - Batch size override
- **NO `--dataset` flag** - Must use `--data` with file path
- **NO `--sequence-column` flag** - Must use `--config` for column mapping

### ⚠️ Jain Column Mapping (INTENTIONAL DESIGN)
- **Jain files use**: `vh_sequence` column
- **Why**: Line 323 `preprocessing/jain/step2_preprocess_p5e_s2.py`:
  ```python
  # NOTE: Column must be 'vh_sequence' not 'sequence' for JainDataset.load_data() compatibility
  ```
- **This is NOT a bug** - It's intentional for JainDataset API compatibility
- **Solution**: Use `--config` with YAML that specifies `sequence_column: vh_sequence`

### ❌ What Does NOT Exist
- NO test configs in `configs/testing/` (I deleted the ones I created)
- NO `--dataset` flag for automatic column mapping
- NO `--sequence-column` CLI flag override

## The Actual Validation Commands

### Training (Already Works)
```bash
uv run antibody-train
```
**Output**: `experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl`

### Testing Jain (Column Mapping Required)

**Option 1: Inline YAML config** (cleanest)
```bash
cat > /tmp/jain_test.yaml << 'EOF'
model_paths: [experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl]
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence
label_column: label
output_dir: experiments/runs/tests/jain
device: cpu
batch_size: 8
EOF

uv run antibody-test --config /tmp/jain_test.yaml
```

**Option 2: Create permanent test config** (if we test Jain often)
```bash
# Create test config directory
mkdir -p configs/test

# Create Jain test config
cat > configs/test/jain_p5e_s2.yaml << 'EOF'
model_paths: [experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl]
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence
label_column: label
output_dir: experiments/runs/tests/jain
device: cpu
batch_size: 8
EOF

# Run test
uv run antibody-test --config configs/test/jain_p5e_s2.yaml
```

### Testing Shehata (Standard Columns - No Config Needed)
**WAIT** - Check if this works:
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --data data/test/shehata/fragments/VH_only_shehata.csv
```

If this FAILS with "column not found", then ALL tests need configs (not just Jain).

### Testing Harvey (Standard Columns - No Config Needed?)
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --data data/test/harvey/fragments/VHH_only_harvey.csv
```

## ✅ VERIFIED: What Works After Refactoring

### Column Names Are Correct By Design
- **Shehata**: Uses `sequence` column ✅ - Works with CLI defaults
- **Harvey**: Uses `sequence` column ✅ - Works with CLI defaults
- **Jain**: Uses `vh_sequence` column ✅ - INTENTIONAL (preprocessing/jain/step2_preprocess_p5e_s2.py:323)

### What You Need to Create for Validation

**ONLY ONE CONFIG FILE IS NEEDED:**
- `configs/testing/jain_p5e_s2.yaml` - For Jain column mapping

**ALL OTHER TESTS WORK WITH CLI DEFAULTS** (Shehata, Harvey use `sequence`)

### Legacy Cleanup Needed
- **`configs/config.yaml`** still exists (not removed yet)
- **Action**: Should be removed per V0.5.0_CLEANUP_PLAN.md:21-88
- **Training pipeline ONLY uses Hydra** (`src/antibody_training_esm/conf/config.yaml`)

## Not a Bug - This Is The Design

The test CLI was designed to support YAML configs for exactly this reason:
- Different datasets have different column names (by design)
- YAML configs provide flexibility for column mapping
- This is NOT a workaround - it's the **intended interface**

## No Decision Needed - System Works As Designed

The interface works correctly:
1. ✅ Shehata/Harvey: Use CLI defaults (`--model`, `--data`)
2. ✅ Jain: Use `--config` with column mapping YAML
3. ✅ Training: Use Hydra configs (`uv run antibody-train`)

**All validation commands documented in VALIDATION_ROADMAP.md are now correct.**
