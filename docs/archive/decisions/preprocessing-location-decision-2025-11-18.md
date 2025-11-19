# Preprocessing Directory Structure - Architectural Decision

**Date:** 2025-11-18
**Status:** ✅ FINAL DECISION - Keep at Root (Permanent)
**Decision:** Preprocessing stays at root. Move to `src/` ONLY if packaging/distribution requires it.
**Rationale:** One-time ETL scripts (Factory) should NOT be bundled with runtime code (Product)

---

## Executive Summary

**Current Status:** ✅ CORRECT ARCHITECTURE (9/10 for research repos with one-time ETL)
**Industry Standard:** ⭐ src/package/preprocessing/ (10/10 for libraries with runtime data utilities)
**Our Context:** Research repository with dataset-specific one-time ETL scripts
**Decision:** Keep at root permanently. Moving to `src/` would be architecturally incorrect for our use case.

---

## Why Keep at Root (Permanent Decision)

### 1. Factory vs Product Separation

**Core Principle:** Separate "data manufacturing" (Factory) from "data usage" (Product)

- **`src/antibody_training_esm/`** = **Product**
  - Training and inference library
  - What you would `pip install` and import at runtime
  - Contains: Models, data loaders, classifiers, evaluation

- **`preprocessing/`** = **Factory**
  - One-time data manufacturing scripts
  - Converts raw paper data (Excel files, PDFs) → canonical CSVs
  - **Never needed at runtime or in production**
  - Contains: Dataset-specific ETL pipelines

**If you move preprocessing to `src/`:**
- ❌ You bundle the construction equipment (cranes, cement mixers) inside the finished building
- ❌ You bloat the library with dependencies like `openpyxl` (Excel parsing) that are dead code in production
- ❌ You mix "data creation" (run once) with "data usage" (run repeatedly)

**Analogy:**
- ❌ **Wrong**: Shipping a car with the entire factory inside the trunk
- ✅ **Right**: Factory (preprocessing/) builds the car, then stays behind. Car (src/) goes to customer.

### 2. ETL vs Runtime Data Loading - The Critical Distinction

**Our preprocessing is ONE-TIME ETL (Extract, Transform, Load):**
- `preprocessing/jain/step1_convert_excel_to_csv.py` - Parses Jain et al. 2023 Excel file (one-time)
- `preprocessing/boughter/stage1_dna_translation.py` - Translates DNA → protein (one-time)
- `preprocessing/harvey/step1_convert_raw_csvs.py` - Merges Harvey dataset CSVs (one-time)

**Runtime data loading DOES live in `src/` (correctly):**
- `src/antibody_training_esm/data/loaders.py` - Loads CSVs during training (runtime)
- `src/antibody_training_esm/datasets/jain.py` - Dataset class for training loop (runtime)

**Key Question:** "Will this code run during model training or inference?"
- ✅ **YES** → Belongs in `src/`
- ❌ **NO (one-time only)** → Belongs at root

### 3. Research Reproducibility

**Top-level visibility signals scientific importance:**
- Data transformation methodology is as critical as model architecture
- Preprocessing scripts are part of the **scientific contribution**, not just implementation details
- Makes ETL logic discoverable for peer review and reproduction studies

**If buried in `src/antibody_training_esm/preprocessing/`:**
- Signals: "Just utility code, not important"
- Harder to find for researchers reviewing methodology

### 4. TensorFlow/PyTorch Comparison - Why It Doesn't Apply

**Common Misconception:** "TensorFlow has `official/nlp/data/` inside the package, so we should too."

**Critical Analysis:**
- TensorFlow's `data/` contains **runtime data loaders** (tokenizers, data pipelines for training)
- Our `preprocessing/` contains **one-time ETL scripts** (Excel parsers, QC filters)

**Analogy:**
- TensorFlow pattern: Kitchen equipment (oven, mixer) - used repeatedly during cooking
- Our pattern: Food processing factory - used once to produce ingredients, then ingredients go to kitchen

**Verdict:** Industry patterns for runtime utilities ≠ our one-time ETL scripts.

---

## Current Structure (Permanent)

```
antibody_training_pipeline_ESM/
├── preprocessing/          # ← At ROOT (current)
│   ├── boughter/
│   ├── jain/
│   ├── harvey/
│   └── shehata/
├── src/antibody_training_esm/
│   ├── data/              # Runtime loaders
│   ├── datasets/          # Dataset classes
│   └── ...
└── data/                  # Actual data files
```

**Import Pattern:**
```python
from preprocessing.jain.step1_convert_excel_to_csv import calculate_flags
```

---

## Alternative Structure (Optional, NOT Recommended for Our Use Case)

```
antibody_training_pipeline_ESM/
├── src/antibody_training_esm/
│   ├── preprocessing/      # ← INSIDE package (future)
│   │   ├── __init__.py
│   │   ├── boughter/
│   │   ├── jain/
│   │   ├── harvey/
│   │   └── shehata/
│   ├── data/              # Runtime loaders
│   ├── datasets/          # Dataset classes
│   └── ...
└── data/                  # Actual data files
```

**Import Pattern (Alternative):**
```python
from antibody_training_esm.preprocessing.jain.step1_convert_excel_to_csv import calculate_flags
```

**When to Use This Structure (RARE):**

Move preprocessing to `src/` ONLY if:

1. **Publishing as PyPI Library** (not just research repo)
   - You want users to `pip install antibody-training-esm` and import preprocessing utilities
   - The preprocessing code will be used by OTHER projects at runtime
   - Example: "We built a reusable antibody sequence validator library"

2. **Preprocessing Becomes Runtime Utilities**
   - The "one-time ETL" scripts evolve into reusable data transformation libraries
   - Other researchers need to import your preprocessing functions
   - Currently: ❌ NOT the case (scripts are dataset-specific, one-time)

3. **Corporate Packaging Requirements**
   - Company policy mandates everything in one package
   - Deployment infrastructure requires single installable unit

**DO NOT move if:**
- ✅ This is primarily a research repository (our case)
- ✅ Preprocessing is one-time dataset-specific ETL (our case)
- ✅ Scripts parse specific paper data formats (our case)
- ✅ No plan to distribute preprocessing as reusable library (our case)

**Verdict for This Project:** ❌ **Don't move.** Our preprocessing is classic one-time ETL.

---

## 🎯 THE HONEST TRUTH FROM 2025 PROFESSIONAL PATTERNS

### What Pros ACTUALLY Do (Web Search Evidence):

| Pattern | Example | Location | Frequency |
|---------|---------|----------|-----------|
| **Inside Package** | TensorFlow: `official/nlp/data/` | `src/package/preprocessing/` | **MOST COMMON** ✅ |
| **Inside Package** | Cookiecutter: `src/ml/data/` | `src/package/data/` | **MOST COMMON** ✅ |
| **At Root** | Research repos, data pipelines | `preprocessing/` or `preprocess/` | **VALID** ⚠️ |
| **Wrong** | (None found) | `scripts/preprocessing/` | **NEVER** ❌ |

---

## Comparison: Root vs src/ for One-Time ETL Scripts

### Current Structure (At Root) - Our Choice ✅

**Strengths:**
- ✅ **Correct separation:** Factory (data creation) ≠ Product (data usage)
- ✅ **Clear visibility:** Easy to find for researchers reviewing methodology
- ✅ **Lean package:** `src/` stays focused on runtime code only
- ✅ **No bloat:** Avoids bundling Excel parsers and one-time scripts in pip package
- ✅ **Dataset-centric:** Each dataset owns its complete ETL pipeline
- ✅ **Research-appropriate:** Emphasizes data lineage and scientific reproducibility

**Trade-offs:**
- ⚠️ Import path is outside main package: `from preprocessing.jain import ...`
- ⚠️ Not the "everything in one package" style (but that's intentional)

### Alternative Structure (Inside src/) - NOT Recommended ❌

**Strengths:**
- ✅ Prettier import path: `from antibody_training_esm.preprocessing import ...`
- ✅ Everything in one package (if that's a requirement)

**Weaknesses:**
- ❌ **Violates Factory/Product separation:** Bundles data manufacturing with data usage
- ❌ **Bloats runtime package:** Includes Excel parsers and one-time ETL in pip package
- ❌ **Mixes concerns:** One-time ETL (run once) lives with runtime code (run repeatedly)
- ❌ **Misleading signal:** Buries scientific methodology as "just utility code"
- ❌ **Wrong pattern:** TensorFlow's `data/` is for runtime loaders, not one-time ETL

**Cost to switch:**
- 4-6 hours refactoring for architectural downgrade

---

## Refactoring Checklist (Optional - NOT Recommended Unless Required)

**⚠️ WARNING:** Only use this checklist if you have a SPECIFIC requirement to move preprocessing to `src/`. For our research use case, this is NOT recommended.

### Valid Reasons to Move (RARE):
- [ ] Publishing as PyPI library where preprocessing utilities will be imported by other projects
- [ ] Preprocessing evolved from one-time ETL to reusable runtime utilities
- [ ] Corporate packaging policy requires everything in one package

### DO NOT Move If (Our Case):
- [x] This is a research repository (not a general-use library)
- [x] Preprocessing is one-time dataset-specific ETL
- [x] Scripts parse specific paper data formats (Jain Excel, Harvey CSVs)
- [x] No plan to distribute preprocessing as reusable library
- [x] Want to keep Factory (data creation) separate from Product (data usage)

---

## Refactoring Steps (OPTIONAL - Only If Required)

**⚠️ SKIP THIS SECTION** unless you determined above that moving is necessary.

**Reminder:** For research repos with one-time ETL, keeping preprocessing at root is the CORRECT architecture.

**If you still need to proceed** (e.g., corporate requirement), here's the full checklist:

### Phase 1: Move Files (30 minutes)
```bash
# Create new location
mkdir -p src/antibody_training_esm/preprocessing

# Move dataset directories
mv preprocessing/boughter src/antibody_training_esm/preprocessing/
mv preprocessing/jain src/antibody_training_esm/preprocessing/
mv preprocessing/harvey src/antibody_training_esm/preprocessing/
mv preprocessing/shehata src/antibody_training_esm/preprocessing/

# Move README and __init__.py
mv preprocessing/README.md src/antibody_training_esm/preprocessing/
mv preprocessing/__init__.py src/antibody_training_esm/preprocessing/

# Remove old directory
rmdir preprocessing/
```

### Phase 2: Update Imports (1-2 hours)

**Files to Update:**

1. **Internal Imports (1 file):**
   - `src/antibody_training_esm/preprocessing/jain/validate_conversion.py:26`
   ```python
   # OLD:
   from preprocessing.jain.step1_convert_excel_to_csv import ...

   # NEW:
   from antibody_training_esm.preprocessing.jain.step1_convert_excel_to_csv import ...
   ```

2. **CLI References (1 file):**
   - `src/antibody_training_esm/cli/preprocess.py` (4 path references)
   ```python
   # OLD:
   "jain": "preprocessing/jain/step2_preprocess_p5e_s2.py"

   # NEW:
   "jain": "src/antibody_training_esm/preprocessing/jain/step2_preprocess_p5e_s2.py"
   ```

3. **Validation Scripts (1 file):**
   - `scripts/validation/validate_jain_csvs.py` (3 path references)

4. **Data Manifests (1 file):**
   - `data/test/jain/fragments/manifest.yml:7`
   ```yaml
   # OLD:
   script: preprocessing/jain/step3_extract_fragments.py

   # NEW:
   script: src/antibody_training_esm/preprocessing/jain/step3_extract_fragments.py
   ```

### Phase 3: Update Tests (30 minutes)

**Files to Update (6 files):**
- `tests/unit/cli/test_preprocess.py` (4 path assertions)
- `tests/e2e/test_reproduce_novo.py` (2 paths in skip messages)
- `tests/integration/test_boughter_embedding_compatibility.py` (2 references)
- `tests/integration/test_harvey_embedding_compatibility.py` (2 references)
- `tests/integration/test_jain_embedding_compatibility.py` (check for references)
- `tests/integration/test_shehata_embedding_compatibility.py` (check for references)

### Phase 4: Update Documentation (2-3 hours)

**Core Documentation (6 files):**
- `CLAUDE.md` (15+ references)
- `GEMINI.md` (15+ references)
- `README.md` (check for references)
- `USAGE.md` (check for references)
- `ARCHITECTURAL_FIXES_PLAN.md` (update if still relevant)
- `PREPROCESSING_SCRIPTS_REFACTORING_SPEC.md` (mark as completed)

**Dataset Documentation (40+ files):**
- `docs/datasets/boughter/*.md`
- `docs/datasets/jain/*.md`
- `docs/datasets/harvey/*.md`
- `docs/datasets/shehata/*.md`

**User/Developer Guides (15+ files):**
- `docs/user-guide/*.md`
- `docs/developer-guide/*.md`
- `docs/research/*.md`

### Phase 5: Verification (30 minutes)

```bash
# All imports work
python3 -c "from antibody_training_esm.preprocessing.jain import step1_convert_excel_to_csv"

# All tests pass
uv run pytest

# CLI still works
uv run antibody-preprocess --help

# Preprocessing scripts run
uv run python -m antibody_training_esm.preprocessing.jain.step2_preprocess_p5e_s2

# Documentation builds (if applicable)
# Check all links in markdown files
```

---

## Alternative: Keep Current Structure Permanently

**If you decide current structure is good enough:**

1. Add to `preprocessing/README.md`:
   ```markdown
   ## Why Separate from src/?

   **TL;DR:** Preprocessing pipelines are dataset-centric ETL scripts,
   not core ML runtime code.

  **Design Decision:** We keep preprocessing/ at the project root to:
   - Emphasize separation between "data creation" and "model training"
   - Maintain dataset-centric organization (all Jain code in one place)
   - Allow non-Python users to easily discover preprocessing scripts
   - Follow research codebase patterns for data-heavy projects

   **Trade-off:** This is "VALID" (9/10 professional) but not "MOST COMMON"
   (10/10). For production deployment, consider moving to
   `src/antibody_training_esm/preprocessing/`.

  For rationale, see `docs/archive/decisions/preprocessing-location-decision-2025-11-18.md` (canonical ADR).
   ```

---

## Quick Actions Kept from Earlier Spec (can be done anytime)

- Document PYTHONPATH assumption in `preprocessing/README.md` (run scripts from project root; `uv run` already sets PYTHONPATH accordingly).
- Optional: remove `sys.path.insert` hack in `preprocessing/harvey/test_psr_threshold.py` (low priority cleanup).

2. Add to `CLAUDE.md`:
   ```markdown
   ## Preprocessing Directory Location

   **Note:** Preprocessing pipelines live at project root (`preprocessing/`),
   not inside `src/`. This is a conscious design decision for dataset-centric
   organization. See `docs/archive/decisions/preprocessing-location-decision-2025-11-18.md` (canonical ADR) for the full rationale.
   ```

---

## Decision Matrix

| Criterion | Keep at Root | Move to src/ | Winner |
|-----------|--------------|--------------|--------|
| **2025 Industry Standard** | Valid (9/10) | Standard (10/10) | src/ |
| **Ease of Discovery** | Excellent | Good | Root |
| **Research Friendliness** | Excellent | Good | Root |
| **Production Readiness** | Good | Excellent | src/ |
| **Refactoring Effort** | Zero | 4-6 hours | Root |
| **Import Path Clarity** | Good | Excellent | src/ |
| **Dataset-Centric Design** | Excellent | Good | Root |
| **Risk of Breaking Changes** | Zero | Medium | Root |

**Recommendation:** ✅ **Keep at root PERMANENTLY** (correct architecture for one-time ETL)

---

## Status Tracking

### Completed ✅
- [x] Added missing `preprocessing/boughter/__init__.py` (2025-11-18)
- [x] Verified all datasets have `__init__.py`
- [x] Tested imports work correctly
- [x] Created architectural decision document
- [x] Validated decision from first principles (Factory vs Product separation)
- [x] Confirmed with 2025 industry patterns (runtime utilities ≠ one-time ETL)
- [x] Updated `ARCHITECTURAL_FIXES_PLAN.md` to note preprocessing stays at root

### Optional Future Tasks (Only If Requirements Change) 📋
- [ ] **IF** converting to PyPI library: Consider moving to src/ (unlikely)
- [ ] **IF** preprocessing becomes runtime utilities: Refactor location (not our case)

### Documentation Improvements (Low Priority) 🤔
- [ ] Add "Why at root?" section to `preprocessing/README.md`
- [ ] Expand CLAUDE.md note about preprocessing location

---

## References

- **Industry Research:** Web search on 2025-11-18 confirmed TensorFlow, PyTorch, Cookiecutter patterns
- **Critical Analysis (2025-11-18):** Agent feedback applying first principles:
  - Factory vs Product separation principle
  - ETL vs Runtime utilities distinction
  - TensorFlow pattern (runtime loaders) ≠ Our pattern (one-time ETL)
  - Research reproducibility requirements
- **Analysis Document:** `PREPROCESSING_SCRIPTS_REFACTORING_SPEC.md` (detailed 45-min analysis, archived/deleted)
- **Architectural Plan:** `ARCHITECTURAL_FIXES_PLAN.md` (updated to reflect permanent decision)
- **TensorFlow Example:** `tensorflow/models/official/nlp/data/` (runtime loaders, not one-time ETL)
- **Cookiecutter Example:** `src/ml/data/make_dataset.py` (template for general libraries)

---

## Final Recommendation

**Decision (2025-11-18):**

✅ **Keep at root PERMANENTLY** (`preprocessing/`)
❌ **Do NOT move to src/** (would be architectural downgrade)
🎯 **Priority:** Focus on actual P1 issues (logging, file splitting, etc.)

**Rationale:**

1. **Correct Separation of Concerns:**
   - Factory (data manufacturing) ≠ Product (data usage)
   - One-time ETL scripts should NOT be bundled with runtime code
   - Keeps `src/` lean and focused

2. **Research Context:**
   - This is a research repository, not a general-use library
   - Data transformation methodology deserves top-level visibility
   - Preprocessing is scientific contribution, not just utility code

3. **TensorFlow Comparison Doesn't Apply:**
   - Their `data/` = runtime loaders (used repeatedly)
   - Our `preprocessing/` = one-time ETL (run once)
   - Different use cases require different patterns

4. **Engineering Pragmatism:**
   - Moving would cost 4-6 hours for no functional benefit
   - Would bloat pip package with Excel parsers and dataset-specific ETL
   - Better to spend time on actual improvements (logging, etc.)

**When to Reconsider (RARE):**
- Only if converting to PyPI library where preprocessing utilities are imported by other projects
- Only if preprocessing evolves from one-time ETL to runtime utilities
- NOT for "production deployment" or "publication" (current structure is already correct)

---

**Last Updated:** 2025-11-18
**Document Type:** Architectural Decision Record (ADR)
**Decision Status:** ✅ FINAL - Keep at Root (Permanent)
**Estimated Refactoring Cost (if needed):** 4-6 hours
