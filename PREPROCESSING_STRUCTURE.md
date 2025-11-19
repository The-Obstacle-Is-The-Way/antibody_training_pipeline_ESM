# Preprocessing Directory Structure - Tech Debt Note

**Date:** 2025-11-18
**Status:** 📋 TECH DEBT - Refactor Later
**Priority:** P2 (Medium - Quality of Life Improvement)
**Effort:** 4-6 hours
**Impact:** Code quality improvement, no functional change

---

## Executive Summary

**Current Status:** ✅ VALID (9/10 professional)
**Future Goal:** ⭐ MOST PROFESSIONAL (10/10)
**Decision:** Keep current structure NOW, refactor later when prepping for production/publication

---

## Current Structure (VALID - Keep for Now)

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

## Recommended Future Structure (MOST PROFESSIONAL)

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

**Import Pattern (Future):**
```python
from antibody_training_esm.preprocessing.jain.step1_convert_excel_to_csv import calculate_flags
```

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

## Why Current Structure is VALID (Keep for Now)

### Pros:
- ✅ **Clear visibility** (easy to find)
- ✅ **Semantic separation** (data ETL ≠ ML code)
- ✅ **Works for research/data-heavy projects**
- ✅ **Document analysis confirms this is acceptable**
- ✅ **Each dataset owns complete pipeline** (dataset-centric design)
- ✅ **Separation of concerns:**
  - `preprocessing/` = Run-once ETL scripts to CREATE data
  - `src/antibody_training_esm/data/` = Runtime data LOADING
  - `src/antibody_training_esm/datasets/` = Dataset classes

### Cons:
- ⚠️ **Not the "pure Python package" style**
- ⚠️ **Import path:** `from preprocessing.jain import ...` (outside main package)
- ⚠️ **Less conventional** for production ML packages

---

## Why Future Structure is BETTER (Refactor Later)

### Pros:
- ✅ **Matches TensorFlow/Cookiecutter 2025 patterns**
- ✅ **Import path:** `from antibody_training_esm.preprocessing.jain import ...` (cleaner)
- ✅ **Everything in one installable package**
- ✅ **More "production" feel**
- ✅ **Better for pip/uv distribution**
- ✅ **Aligns with Google DeepMind engineering standards**

### Cons:
- ⚠️ **Harder to discover for non-Python users**
- ⚠️ **Philosophically mixes "run-once ETL" with "runtime code"**
- ⚠️ **Requires 4-6 hours refactoring effort**

---

## Refactoring Checklist (Future Work)

### When to Refactor:
- [ ] Preparing for production deployment
- [ ] Preparing for publication/open source release
- [ ] When team size grows beyond 3-4 people
- [ ] When other major refactoring is already planned
- [ ] When adding CI/CD for automated preprocessing

### DO NOT Refactor If:
- [ ] Actively shipping features
- [ ] Under tight deadlines
- [ ] Only internal research use
- [ ] Team prefers current structure

---

## Refactoring Steps (4-6 Hours Total)

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

   For rationale, see `PREPROCESSING_STRUCTURE.md` in project root.
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
   organization. See `PREPROCESSING_STRUCTURE.md` for full rationale.
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

**Recommendation:** Keep at root NOW, move to src/ LATER (before production)

---

## Status Tracking

### Completed ✅
- [x] Added missing `preprocessing/boughter/__init__.py` (2025-11-18)
- [x] Verified all datasets have `__init__.py`
- [x] Tested imports work correctly
- [x] Created this tech debt document

### Pending (Do Before Production) 📋
- [ ] Move `preprocessing/` to `src/antibody_training_esm/preprocessing/`
- [ ] Update all imports (15+ files)
- [ ] Update all tests (6+ files)
- [ ] Update all documentation (60+ files)
- [ ] Verify all preprocessing scripts still work
- [ ] Update CI/CD if needed

### Optional (If Keeping at Root) 🤔
- [ ] Add "Why at root?" section to `preprocessing/README.md`
- [ ] Add note to `CLAUDE.md` explaining decision
- [ ] Update `ARCHITECTURAL_FIXES_PLAN.md` to reflect decision

---

## References

- **Industry Research:** Web search on 2025-11-18 confirmed TensorFlow, PyTorch, Cookiecutter all prefer `src/package/preprocessing/`
- **Analysis Document:** `PREPROCESSING_SCRIPTS_REFACTORING_SPEC.md` (detailed 45-min analysis)
- **Architectural Plan:** `ARCHITECTURAL_FIXES_PLAN.md` (mentions this as medium priority)
- **TensorFlow Example:** `tensorflow/models/official/nlp/data/`
- **Cookiecutter Example:** `src/ml/data/make_dataset.py`

---

## Final Recommendation

**Current Decision (2025-11-18):**

✅ **Keep at root** (`preprocessing/`)
📅 **Refactor later** when prepping for production
🎯 **Priority:** Focus on fixing print statements, logging, and other P1 issues first

**Rationale:** Current structure is VALID and works well for research. Refactoring to `src/` provides marginal benefit (9→10) at moderate cost (4-6 hours). Better to invest that time in:
1. Migrating 799 print() statements to logging (P1, 4-6 hours)
2. Splitting overly long files (P1, 3-4 hours)
3. Centralizing hardcoded paths (P1, 2 hours)
4. Other functional improvements

**When to Revisit:** Before production deployment, publication, or major open source release.

---

**Last Updated:** 2025-11-18
**Next Review:** Before production deployment
**Owner:** Tech debt backlog
**Estimated Effort:** 4-6 hours when prioritized
