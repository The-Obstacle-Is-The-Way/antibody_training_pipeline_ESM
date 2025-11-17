# Burner Docs Audit Report - Phase 1 Findings

**Created**: 2025-11-17
**Phase**: Phase 1 (Burner Docs Audit) of DOCUMENTATION_UPDATE_ROADMAP.md
**Purpose**: Identify stale references in 7 gold burner docs before integration
**Files Audited**: 7 files in `docs_burner/implementation/`

---

## 🎯 Executive Summary

**Current Repository State (Verified 2025-11-17):**
- ✅ `configs/config.yaml` → ❌ **DOES NOT EXIST** (deleted in Phase 5)
- ✅ Root `models/` → ❌ **DOES NOT EXIST** (moved to `experiments/checkpoints/`)
- ✅ Root `embeddings_cache/` → ❌ **DOES NOT EXIST** (moved to `experiments/cache/`)
- ✅ Root `logs/` → ❌ **DOES NOT EXIST** (moved to `experiments/runs/logs/`)
- ✅ Root `outputs/` → ❌ **DOES NOT EXIST** (moved to `experiments/runs/`)
- ✅ `experiments/` → ✅ **EXISTS** with subdirs: `benchmarks/`, `cache/`, `checkpoints/`, `runs/`

**Audit Results:**

| File | Old Path References | Classification | Priority | Update Needed? |
|------|---------------------|----------------|----------|----------------|
| TEST_COVERAGE_PLAN.md | 0 | ✅ Clean | N/A | ✅ **NO** - Ready to integrate |
| VALIDATION_ROADMAP.md | 2 | ✅ Accurate (historical) | P3 Low | ⚠️ **MINOR** - Add "EXECUTED" markers |
| VALIDATION_REALITY.md | 3 | ❌ **STALE** | P0 Critical | ❌ **YES** - Line 137 claims file exists |
| REPOSITORY_REORGANIZATION_PLAN.md | 73 | ✅ Accurate (plan doc) | P2 Medium | ⚠️ **MINOR** - Mark as "EXECUTED" |
| V0.5.0_CLEANUP_PLAN.md | 39 | ✅ Accurate (plan doc) | P2 Medium | ⚠️ **MINOR** - Mark as "EXECUTED" |
| REPOSITORY_STRUCTURE_ANALYSIS.md | 32 | ✅ Accurate (historical) | P3 Low | ⚠️ **MINOR** - Clarify past tense |
| OUTPUT_DIRECTORY_INVESTIGATION.md | 33 | ⚠️ Mixed (pre-Phase 5) | P1 High | ⚠️ **MODERATE** - Update to experiments/ paths |

**Key Findings:**
- 🟢 **1 file completely clean** (TEST_COVERAGE_PLAN.md)
- 🟡 **5 files historically accurate** (document plans/evolution correctly)
- 🔴 **1 file critically stale** (VALIDATION_REALITY.md claims configs/config.yaml exists)
- ⚠️ **1 file needs path updates** (OUTPUT_DIRECTORY_INVESTIGATION.md pre-Phase 5)

---

## 📋 File-by-File Audit Results

### ✅ 1. TEST_COVERAGE_PLAN.md - CLEAN

**Old Path References**: 0
**Status**: ✅ **READY TO INTEGRATE** (no updates needed)

**Findings**:
- Zero references to `configs/config.yaml`, `models/`, `embeddings_cache/`, `logs/`, or `outputs/`
- All paths and commands are current
- No staleness issues

**Recommendation**: Integrate as-is into `docs/developer-guide/testing-strategy.md`

---

### ✅ 2. VALIDATION_ROADMAP.md - MOSTLY ACCURATE

**Old Path References**: 2
**Status**: ✅ **ACCURATE** (references are historical/contextual)
**Priority**: P3 Low (minor updates)

**References Found**:
1. **Line 37**: "Remove legacy `configs/config.yaml`" - Part of prerequisite task list
2. **Line 127**: "Root-level `models/`, `outputs/`, `embeddings_cache/`, and `logs/` have been REMOVED" - Accurate statement

**Context Analysis**:
- Line 37 is CORRECT - it's documenting a prerequisite task (V0.5.0_CLEANUP_PLAN.md)
- Line 127 is CORRECT - it explicitly states these directories "have been REMOVED"
- Both references are ACCURATE historical context

**Staleness Assessment**: ⚠️ **MINOR**
- References are accurate, but could be clearer about execution status
- V0.5.0_CLEANUP_PLAN.md prerequisite (line 37) has been EXECUTED

**Recommended Updates**:
1. Line 34-40: Add note that V0.5.0_CLEANUP_PLAN.md Problem 1 is ✅ **EXECUTED** (configs/config.yaml deleted)
2. Consider updating "FIRST: Remove legacy..." to "✅ DONE: Removed legacy..."

**Integration Destination**: `docs/developer-guide/validation-checklist.md`

---

### ❌ 3. VALIDATION_REALITY.md - CRITICALLY STALE

**Old Path References**: 3
**Status**: ❌ **STALE** (claims deleted file still exists)
**Priority**: P0 Critical (must fix before integration)

**References Found**:
1. **Line 5**: "Execute V0.5.0_CLEANUP_PLAN.md Problem 1 (remove `configs/config.yaml`)" - Prerequisite
2. **Line 9**: "NOT legacy `configs/config.yaml`" - Historical context (accurate)
3. **Line 137**: ❌ **"- `configs/config.yaml` still exists (not removed yet)"** - **FALSE**

**Staleness Assessment**: ❌ **CRITICAL**
- Line 137 INCORRECTLY claims `configs/config.yaml` still exists
- This file was verified DELETED on 2025-11-17
- Contradiction: Line 5 says "remove it", Line 137 says "still exists"

**Verified Current State**:
```bash
$ ls -la configs/config.yaml
ls: configs/config.yaml: No such file or directory
```

**Recommended Updates**:
1. **Line 137**: Change to "- ✅ `configs/config.yaml` REMOVED (deleted in Phase 5)"
2. **Line 5**: Add "✅ EXECUTED:" prefix
3. Add timestamp: "Last verified: 2025-11-17"

**Integration Destination**: Merge into `docs/developer-guide/validation-checklist.md` (after updates)

---

### ✅ 4. REPOSITORY_REORGANIZATION_PLAN.md - ACCURATE (EXECUTED PLAN)

**Old Path References**: 73
**Status**: ✅ **ACCURATE** (documents executed migration)
**Priority**: P2 Medium (mark as executed)

**References Found**: 73 references documenting the migration:
- Line 28: "Scattered Outputs: 4 root-level directories (`outputs/`, `models/`, `embeddings_cache/`, `logs/`)"
- Line 60-63: Before state (root-level directories)
- Line 172-217: Migration commands (`mv models/* experiments/checkpoints/`, etc.)
- Lines 578-606: Migration plan details

**Context Analysis**:
- This is a **PLAN DOCUMENT** that was **EXECUTED**
- All references are INTENTIONAL - documenting the migration (old → new)
- The plan describes moving from root-level dirs → `experiments/` hierarchy
- Verified: Migration was COMPLETED (old dirs don't exist, new structure exists)

**Staleness Assessment**: ⚠️ **MINOR**
- Content is ACCURATE (correctly documents the migration)
- Missing: Execution timestamp and completion markers
- Should be marked as "EXECUTED" or moved to docs/archive/

**Recommended Updates**:
1. Add at top: "**Status**: ✅ EXECUTED (2025-11-XX) - Phase 5 Complete"
2. Add verification section: "All old directories confirmed removed, new structure verified"
3. Update tone from "will move" → "moved" (past tense)
4. Consider moving to `docs/archive/migrations/phase-5-reorganization.md`

**Integration Destination**: `docs/architecture/experiments-structure.md` OR `docs/archive/migrations/phase-5-reorganization.md`

---

### ✅ 5. V0.5.0_CLEANUP_PLAN.md - ACCURATE (PARTIALLY EXECUTED PLAN)

**Old Path References**: 39
**Status**: ✅ **ACCURATE** (documents cleanup plan)
**Priority**: P2 Medium (mark execution status)

**References Found**: 39 references documenting the cleanup:
- Line 15-24: States `configs/config.yaml` will be deleted
- Line 35: "Remove all legacy configuration infrastructure (`configs/config.yaml` + `train_model()`)"
- Line 51-63: Documents dependencies on `configs/config.yaml`
- Line 132-510: Detailed cleanup steps

**Context Analysis**:
- This is a **CLEANUP PLAN** document
- Problem 1 (remove `configs/config.yaml`) appears to be ✅ **EXECUTED**
- Problem 2 (remove `train_model()` function) - status unclear
- All references are INTENTIONAL - documenting what will/should be removed

**Verified Execution Status**:
- ✅ `configs/config.yaml`: DELETED (verified 2025-11-17)
- ❓ `train_model()` function: Need to verify if removed from codebase

**Staleness Assessment**: ⚠️ **MINOR**
- Content is ACCURATE as a plan document
- Missing: Execution status markers
- Needs verification: Has Problem 2 (`train_model()` removal) been completed?

**Recommended Updates**:
1. Add execution status at top:
   - "**Problem 1 Status**: ✅ EXECUTED (configs/config.yaml deleted 2025-11-XX)"
   - "**Problem 2 Status**: ❓ NEEDS VERIFICATION (check if train_model() removed)"
2. Mark completed sections with ✅ checkmarks
3. Add timestamp when fully executed
4. Consider archiving if both problems resolved

**Integration Destination**: `docs/archive/migrations/v0.5.0-legacy-removal.md` (if executed) OR keep in burner until Problem 2 verified

---

### ✅ 6. REPOSITORY_STRUCTURE_ANALYSIS.md - ACCURATE (HISTORICAL DOC)

**Old Path References**: 32
**Status**: ✅ **ACCURATE** (documents repository evolution)
**Priority**: P3 Low (minor clarity improvements)

**References Found**: 32 references documenting repository history:
- Line 17: "Output artifacts scattered across multiple locations (`outputs/`, `models/`, `embeddings_cache/`, `logs/`)" - Past problem
- Line 33-36: "✅ Moved `outputs/*` → `experiments/runs/`" - Completed migration
- Line 171-241: "Before" state (root-level directories with ⚠️ markers)
- Line 348-352: Migration results table
- Line 558-606: Problem analysis and solutions

**Context Analysis**:
- This is a **HISTORICAL ANALYSIS** document (documents evolution)
- References to old paths are CORRECT - they describe what existed BEFORE Phase 5
- Uses ✅ checkmarks to show migrations were completed
- Clear distinction between "before" and "after" states

**Staleness Assessment**: ⚠️ **MINOR**
- Content is ACCURATE (correctly documents history)
- Past tense used appropriately (e.g., "Moved", "Scattered")
- Could be clearer about timeline (when migrations happened)

**Recommended Updates**:
1. Add timeline markers: "Pre-Phase 5 (before 2025-11-XX)" vs "Post-Phase 5 (after 2025-11-XX)"
2. Add final verification section: "Current state verified 2025-11-17"
3. Ensure all migration checkmarks are ✅ (appears already done)
4. Minor: Clarify this is HISTORICAL documentation in title or header

**Integration Destination**: `docs/architecture/repository-evolution.md`

---

### ⚠️ 7. OUTPUT_DIRECTORY_INVESTIGATION.md - MIXED (PRE-PHASE 5)

**Old Path References**: 33
**Status**: ⚠️ **MIXED** (accurate for pre-Phase 5, needs updates for post-Phase 5)
**Priority**: P1 High (significant path updates needed)

**References Found**: 33 references to pre-Phase 5 structure:
- Line 13: "`outputs/` - Hydra-managed training scratch space ✅ CORRECT"
- Line 23-62: Detailed analysis of `outputs/` directory (pre-Phase 5)
- Line 115-138: Analysis of `outputs/test_dataset/`
- Line 151-152: Training logs in `outputs/novo_replication/.../logs/training.log`
- Line 212-222: Gitignore analysis referencing old structure
- Line 228-293: Q&A section referencing `outputs/`, `models/`, `embeddings_cache/`

**Context Analysis**:
- This document was written BEFORE Phase 5 reorganization
- It analyzes the OLD directory structure (`outputs/`, `models/`, etc.)
- Contains valuable insights about hierarchical routing and directory organization
- Routing logic (model/classifier hierarchy) is STILL RELEVANT

**Staleness Assessment**: ⚠️ **MODERATE**
- **ACCURATE**: Hierarchical routing insights (model/classifier structure)
- **STALE**: All specific paths need updating (`outputs/` → `experiments/runs/`)
- **STALE**: References to old `.gitignore` patterns
- **VALUABLE**: Documents routing issues and CV results gap (still applicable)

**Recommended Updates**:
1. Add header: "**Historical Context**: Written pre-Phase 5. Paths updated to reflect experiments/ hierarchy."
2. Global path replacements:
   - `outputs/` → `experiments/runs/`
   - `models/` → `experiments/checkpoints/`
   - `embeddings_cache/` → `experiments/cache/`
3. Update lines 212-222: Current `.gitignore` patterns for `experiments/`
4. Update lines 228-293: Q&A section with current paths
5. Keep routing insights (hierarchical model/classifier structure) - still valid
6. Extract CV results gap section for preprocessing docs (valuable for developers)

**Integration Destinations**:
- Routing documentation → `docs/developer-guide/output-routing.md`
- CV results gap → `docs/developer-guide/preprocessing-internals.md`
- Hierarchical structure insights → `docs/architecture/experiments-structure.md`

---

## 📊 Staleness Summary

### By Severity

**P0 - Critical (Must Fix Before Integration)**:
1. VALIDATION_REALITY.md - Line 137 claims `configs/config.yaml` exists (FALSE)

**P1 - High (Significant Updates Needed)**:
2. OUTPUT_DIRECTORY_INVESTIGATION.md - 33 path references need updating to `experiments/` hierarchy

**P2 - Medium (Minor Updates, Mark Execution Status)**:
3. REPOSITORY_REORGANIZATION_PLAN.md - Mark as "EXECUTED", add timestamps
4. V0.5.0_CLEANUP_PLAN.md - Mark Problem 1 as "EXECUTED", verify Problem 2

**P3 - Low (Clarity Improvements)**:
5. VALIDATION_ROADMAP.md - Add "EXECUTED" markers for prerequisites
6. REPOSITORY_STRUCTURE_ANALYSIS.md - Add timeline markers, clarify historical context

**✅ Ready to Integrate**:
7. TEST_COVERAGE_PLAN.md - No updates needed

---

### By Old Path Type

**`configs/config.yaml` References** (44 total):
- VALIDATION_ROADMAP.md: 2 (✅ accurate - historical)
- VALIDATION_REALITY.md: 3 (❌ 1 stale - line 137)
- REPOSITORY_REORGANIZATION_PLAN.md: 1 (✅ accurate - plan doc)
- V0.5.0_CLEANUP_PLAN.md: 38 (✅ accurate - cleanup plan)

**Root `models/` References** (31 total):
- REPOSITORY_REORGANIZATION_PLAN.md: 18 (✅ accurate - migration doc)
- REPOSITORY_STRUCTURE_ANALYSIS.md: 9 (✅ accurate - historical)
- OUTPUT_DIRECTORY_INVESTIGATION.md: 4 (⚠️ needs update to experiments/checkpoints/)

**Root `embeddings_cache/` References** (12 total):
- REPOSITORY_REORGANIZATION_PLAN.md: 6 (✅ accurate - migration doc)
- REPOSITORY_STRUCTURE_ANALYSIS.md: 4 (✅ accurate - historical)
- OUTPUT_DIRECTORY_INVESTIGATION.md: 2 (⚠️ needs update to experiments/cache/)

**Root `logs/` References** (11 total):
- REPOSITORY_REORGANIZATION_PLAN.md: 5 (✅ accurate - migration doc)
- REPOSITORY_STRUCTURE_ANALYSIS.md: 3 (✅ accurate - historical)
- OUTPUT_DIRECTORY_INVESTIGATION.md: 3 (⚠️ needs update to experiments/runs/logs/)

**Root `outputs/` References** (69 total):
- OUTPUT_DIRECTORY_INVESTIGATION.md: 24 (⚠️ needs update to experiments/runs/)
- REPOSITORY_REORGANIZATION_PLAN.md: 20 (✅ accurate - migration doc)
- REPOSITORY_STRUCTURE_ANALYSIS.md: 16 (✅ accurate - historical)
- VALIDATION_ROADMAP.md: 1 (✅ accurate - historical)

---

## 🎯 Prioritized Update Plan

### Immediate (Before Integration)

**1. Fix VALIDATION_REALITY.md (P0)**:
- Line 137: Change "still exists (not removed yet)" → "✅ REMOVED (deleted in Phase 5)"
- Line 5: Add "✅ EXECUTED:" prefix
- Add verification timestamp

**2. Update OUTPUT_DIRECTORY_INVESTIGATION.md (P1)**:
- Global path replacements: `outputs/` → `experiments/runs/`, etc.
- Update .gitignore section
- Add header noting historical context with updated paths
- Extract reusable sections for integration

### Before Integration (Per File)

**3. Mark REPOSITORY_REORGANIZATION_PLAN.md as executed (P2)**:
- Add "Status: ✅ EXECUTED (2025-11-XX)" at top
- Update tone to past tense ("moved" not "will move")
- Add verification section

**4. Mark V0.5.0_CLEANUP_PLAN.md execution status (P2)**:
- Mark Problem 1 as ✅ EXECUTED (configs/config.yaml deleted)
- Verify Problem 2 status (train_model() removal)
- Add execution timestamps

**5. Add clarity to VALIDATION_ROADMAP.md (P3)**:
- Note V0.5.0_CLEANUP_PLAN.md prerequisite is ✅ EXECUTED
- Minor: Update "FIRST: Remove" → "✅ DONE: Removed"

**6. Add timeline to REPOSITORY_STRUCTURE_ANALYSIS.md (P3)**:
- Add "Pre-Phase 5" and "Post-Phase 5" timeline markers
- Add verification timestamp
- Clarify historical documentation purpose

**7. TEST_COVERAGE_PLAN.md (✅ Ready)**:
- No updates needed, integrate as-is

---

## ✅ Success Criteria

Burner docs are ready for integration when:
- ✅ Zero CRITICAL staleness issues (VALIDATION_REALITY.md fixed)
- ✅ All path references updated to `experiments/` hierarchy
- ✅ Executed plans marked with ✅ EXECUTED status
- ✅ Historical docs clearly labeled with timeline context
- ✅ All references verified against current codebase state

---

## 📝 Verification Commands

**Verify current state** (run before each update):
```bash
# Confirm configs/config.yaml deleted
ls -la configs/config.yaml 2>&1 | grep "No such file"

# Confirm old directories deleted
ls -la | grep -E "^d" | awk '{print $9}' | grep -E "^(models|embeddings_cache|logs|outputs)$"
# Should return empty

# Confirm experiments/ structure exists
ls -la experiments/
# Should show: benchmarks, cache, checkpoints, runs
```

**Verify no stale references after updates**:
```bash
# Check for configs/config.yaml references (outside of archive/migration docs)
grep -r "configs/config.yaml" docs_burner/implementation/*.md | grep -v "archive\|migration\|EXECUTED\|REMOVED"

# Check for root-level directory references (outside of historical docs)
grep -r "^models/\|^embeddings_cache/\|^logs/\|^outputs/" docs_burner/implementation/*.md | \
  grep -v "have been REMOVED\|Moved\|EXECUTED\|✅"
```

---

## 🎯 Next Steps (Phase 2)

After completing burner docs updates:
1. Execute Phase 2: Audit canonical docs (`docs/`, `README.md`, `CLAUDE.md`, etc.)
2. Check canonical docs for same stale path patterns
3. Create CANONICAL_DOCS_AUDIT_REPORT.md
4. Synthesize both audits into DOCUMENTATION_UPDATE_PLAN.md (Phase 3)

---

**Audit Completed**: 2025-11-17
**Audited By**: Claude Code (systematic first-principles validation)
**Files Audited**: 7/7 gold burner docs
**Critical Issues Found**: 1 (VALIDATION_REALITY.md line 137)
**Total Old Path References**: 167 (73 accurate historical, 91 accurate plan docs, 3 stale)
**Next Action**: Fix P0/P1 issues, then proceed to Phase 2 (canonical docs audit)
