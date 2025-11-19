# Dataset Column Naming Investigation - Root Cause Analysis

**Date:** 2025-11-18
**Triggered by:** CLI refactoring validation (end-to-end test)
**Status:** ✅ **NOT A BUG - WORKING AS DESIGNED**
**Severity:** 🟢 **LOW** - Documentation/UX clarity issue, not a code bug

---

## Executive Summary

**Conclusion:** The CLI defaulting to `sequence` column while canonical files use `vh_sequence` is **100% INTENTIONAL DESIGN**, not a bug.

**Evidence:**
1. ✅ Training configs explicitly set `sequence_column: vh_sequence` for canonical files
2. ✅ Testing CLI explicitly defaults to `sequence_column: sequence` for fragment files
3. ✅ Documentation shows canonical files used via config files, not direct CLI flags
4. ✅ Fragment files created by preprocessing have standardized `sequence` column

**Root Cause:** Design pattern mismatch between two valid use cases:
- **Canonical files** (`canonical/`): Research-quality datasets with original column names (`vh_sequence`, `vl_sequence`) → Used via config files or Python code
- **Fragment files** (`fragments/`): Standardized test files with uniform column names (`sequence`) → Used via CLI direct flags

**Impact:** ⚠️ **User confusion** - Not immediately clear which files work with which interface
**Fix:** ✅ **Documentation improvement** (not code changes)

---

## Investigation Timeline

### 1. Initial Observation

**Symptom:**
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv

# ERROR: Sequence column 'sequence' not found in dataset.
# Available columns: ['id', 'vh_sequence', 'label']
```

**Initial Hypothesis:** CLI refactoring introduced a regression

### 2. Deep Investigation - Column Naming

**Fragment files (standardized for CLI):**
```bash
$ head -1 data/test/jain/fragments/VH_only_jain.csv
id,sequence,label,elisa_flags,source  # ← Uses 'sequence' ✅
```

**Canonical files (research-quality with original names):**
```bash
$ head -1 data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv
id,vh_sequence,label  # ← Uses 'vh_sequence' (original column name)
```

**Discovery:** Two different column naming conventions exist **by design**

### 3. Historical Analysis - Was This Always The Case?

**Training config (Hydra):**
```yaml
# src/antibody_training_esm/conf/data/boughter_jain.yaml (line 11)
sequence_column: vh_sequence  # ← EXPLICIT config for canonical files
```

**Testing CLI default:**
```python
# src/antibody_training_esm/cli/testing/config.py (line 16)
sequence_column: str = "sequence"  # ← Default for fragment files
```

**Result:** This was ALWAYS the design! Not introduced by refactoring.

### 4. Documentation Review

**Canonical file README:**
```markdown
# data/test/jain/canonical/README.md (lines 80-85)

# Load canonical benchmark
df = pd.read_csv('data/test/jain/canonical/jain_86_novo_parity.csv')
sequences = df['vh_sequence'].tolist()  # ← Manual column access

# NOT meant for direct CLI usage with --data flag!
```

**CLI documentation fix (commit f543490 - Nov 9, 2025):**
```diff
- uv run antibody-test --dataset jain --fragment VH  # WRONG (never existed)
+ uv run antibody-test --model model.pkl --data file.csv  # CORRECT
```

**Finding:** Documentation was CORRECTED to remove references to non-existent `--dataset` flag. Canonical files were never meant for direct CLI `--data` usage.

### 5. Design Pattern Analysis

**Two valid use cases identified:**

#### Use Case 1: CLI Direct Flags (Fragment Files)
```bash
# For quick tests with fragment files
uv run antibody-test \
  --model model.pkl \
  --data data/test/jain/fragments/VH_only_jain.csv  # Uses 'sequence' ✅
```

**Requirements:**
- Fragment file with standardized `sequence` column
- CLI defaults to `sequence_column: "sequence"`
- No config file needed

#### Use Case 2: Config Files (Canonical Files)
```bash
# For reproducible benchmarks with canonical files
uv run antibody-test --config test_jain.yaml
```

**test_jain.yaml:**
```yaml
model_paths: [experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl]
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence  # ← Override for canonical file
label_column: label
```

**Requirements:**
- Canonical file with original column names (`vh_sequence`)
- Config file explicitly sets `sequence_column: vh_sequence`
- Flexible for any column naming convention

#### Use Case 3: Python Code (Canonical Files)
```python
# For custom analysis scripts
df = pd.read_csv('data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv')
sequences = df['vh_sequence'].tolist()  # Manual column access
```

---

## Architectural Justification

### Why Two Column Naming Conventions?

**1. Canonical Files - Research Integrity**
- **Goal:** Preserve original dataset structure
- **Audience:** Researchers, reproducibility audits
- **Column names:** Original (`vh_sequence`, `vl_sequence`, `psr`, `ac_sins`)
- **Usage:** Config files, Python scripts, training pipelines

**Rationale:**
- Maintains provenance with published papers
- Allows rich biophysical metadata
- Enables cross-dataset comparisons with original column names

**2. Fragment Files - CLI Convenience**
- **Goal:** Standardized interface for quick testing
- **Audience:** CLI users, automated workflows
- **Column names:** Standardized (`id`, `sequence`, `label`, `source`)
- **Usage:** Direct CLI `--data` flag

**Rationale:**
- Uniform interface across all datasets (Jain, Harvey, Shehata)
- No config file needed for simple tests
- Fragment-specific (VH-only, H-CDR3, etc.) with consistent naming

---

## Impact Assessment

### ❌ What is NOT Broken

1. **CLI refactoring** - Perfectly preserves old behavior ✅
2. **Column name handling** - Works exactly as designed ✅
3. **Training pipeline** - Uses correct `vh_sequence` via config ✅
4. **Testing with config files** - Column override works ✅
5. **Fragment file testing** - Standardized `sequence` works ✅

### ⚠️ User Experience Gap

**Problem:** Not immediately clear which files work with which interface

**Confusion points:**
1. Users try to use canonical files with CLI direct flags → Error
2. No clear documentation of "canonical vs fragment" file usage patterns
3. Error message mentions `sequence` column but doesn't explain design

**Evidence of confusion:**
- CLAUDE.md line 60 shows outdated `--dataset` flag (now fixed)
- No README explaining canonical vs fragment file purposes
- User (me during validation) initially thought this was a bug

---

## File Artifact Created During Investigation

**File:** `data/test/jain/canonical/VH_only_jain_86_novo_parity_fragment.csv`
**Status:** ⚠️ **UNTRACKED** - Created during investigation, NOT needed
**Purpose:** Temporary workaround to test CLI with canonical data
**Action:** 🗑️ **DELETE** - Not part of intended design

**Why it's not needed:**
```bash
# Instead of fragment-style canonical file, just use config:
cat > test_jain_canonical.yaml << EOF
model_paths: [experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl]
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence
label_column: label
output_dir: ./experiments/benchmarks
EOF

uv run antibody-test --config test_jain_canonical.yaml  # ✅ Works perfectly
```

---

## Recommendations

### Immediate (P0) - Documentation Clarity

**1. Add "File Types Guide" to dataset READMEs:**

```markdown
## File Organization

### Canonical Files (`canonical/`)
**Purpose:** Research-quality datasets with original column names
**Usage:** Config files, Python scripts, training pipelines
**Columns:** Original (`vh_sequence`, `vl_sequence`, biophysical properties)

**Example:**
\`\`\`bash
# Create config file
cat > test_config.yaml << EOF
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence  # Override for original column names
EOF

# Run test
uv run antibody-test --config test_config.yaml
\`\`\`

### Fragment Files (`fragments/`)
**Purpose:** Standardized test files for CLI convenience
**Usage:** Direct CLI `--data` flag
**Columns:** Standardized (`id`, `sequence`, `label`, `source`)

**Example:**
\`\`\`bash
# Direct CLI usage (no config needed)
uv run antibody-test \\
  --model model.pkl \\
  --data data/test/jain/fragments/VH_only_jain.csv
\`\`\`
```

**2. Improve error message in CLI:**

```python
# src/antibody_training_esm/cli/testing/data.py (line 36)

if sequence_col not in df.columns:
    # Detect if user is trying to use canonical file with CLI flags
    if "vh_sequence" in df.columns or "vl_sequence" in df.columns:
        raise ValueError(
            f"Canonical file detected with '{list(df.columns)}' columns.\n"
            f"Canonical files use original column names and require a config file.\n\n"
            f"Two options:\n"
            f"  1. Use fragment file: data/test/[dataset]/fragments/VH_only_[dataset].csv\n"
            f"  2. Create config file with: sequence_column: vh_sequence\n\n"
            f"See: data/test/[dataset]/canonical/README.md for details."
        )
    else:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
```

### Short-term (P1) - CLI Enhancement

**Add column name override flags:**

```python
# src/antibody_training_esm/cli/test.py

parser.add_argument(
    '--sequence-column',
    default='sequence',
    help='Column name for sequences (default: sequence). '
         'Use "vh_sequence" for canonical files.'
)

parser.add_argument(
    '--label-column',
    default='label',
    help='Column name for labels (default: label)'
)
```

**Usage:**
```bash
# Now canonical files work with direct CLI flags
uv run antibody-test \
  --model model.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  --sequence-column vh_sequence
```

### Long-term (P2) - Dataset Standardization

**Consider:** Create fragment-style canonical files for convenience

**Pros:**
- Canonical data + standardized column names
- Works with both CLI and config
- No column override needed

**Cons:**
- Duplicates data (storage cost)
- Two sources of truth per dataset
- Extra maintenance burden

**Decision:** NOT RECOMMENDED - Config files are sufficient

---

## Testing Protocol

### Verify Both Use Cases Work

**Test 1: Fragment files with CLI direct flags** ✅
```bash
uv run antibody-test \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/fragments/VH_only_jain.csv

# Expected: Works (may have NaN labels for held-out sequences)
```

**Test 2: Canonical files with config file** ✅
```bash
cat > test_canonical.yaml << EOF
model_paths: [experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl]
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence
label_column: label
EOF

uv run antibody-test --config test_canonical.yaml

# Expected: 66.28% accuracy (Novo parity)
```

**Test 3: Training with canonical files** ✅
```bash
uv run antibody-train  # Uses conf/data/boughter_jain.yaml

# Config has: sequence_column: vh_sequence
# Expected: Training succeeds with canonical files
```

---

## Conclusion

**Status:** ✅ **WORKING AS DESIGNED - NOT A BUG**

**What happened:**
1. CLI defaults to `sequence` column (for fragment files)
2. Canonical files use `vh_sequence` column (original dataset format)
3. This is INTENTIONAL - two valid use cases with different file types

**What was discovered:**
1. Design pattern exists but is under-documented
2. Error messages don't explain file type difference
3. No clear guide on "canonical vs fragment" usage

**What should be fixed:**
1. ✅ **Documentation** - Add file types guide to READMEs
2. ✅ **Error messages** - Detect canonical files and suggest config approach
3. ✅ **CLI enhancement** - Added `--sequence-column` override flag (Completed 2025-11-18)

**What should NOT be changed:**
1. ❌ Column naming in canonical files (preserve research integrity)
2. ❌ Column naming in fragment files (preserve CLI convenience)
3. ❌ Default CLI behavior (fragment files are the standard CLI use case)

---

**Validated by:** Claude Code (Sonnet 4.5)
**Investigation Date:** 2025-11-18
**Verdict:** Design is sound, documentation improved, CLI enhanced.
**Next Steps:**
1. Delete temporary fragment file: `VH_only_jain_86_novo_parity_fragment.csv` (Done)
2. Add file types guide to dataset READMEs (Done)
3. Improve error messages in CLI data loader (Done)
4. Added `--sequence-column` CLI flag for convenience (Done)
