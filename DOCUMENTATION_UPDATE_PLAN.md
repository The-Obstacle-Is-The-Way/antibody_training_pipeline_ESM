# Documentation Update Plan - Phase 3

**Created**: 2025-11-17
**Status**: ✅ **PLAN READY** - Consolidated from Phase 1 + Phase 2 findings
**Purpose**: Systematic file-by-file update plan with priorities, effort estimates, and verification
**Philosophy**: Fix P0 first (user-breaking), then P1 (historical clarifications), then P2 (integration)

---

## 🎯 Executive Summary

**Total Files to Update**: 17 files (7 P0 + 3 P1 + 7 P2 burner integrations)
**Total Stale References**: 64+ (17 P0 + 7 P1 + 23+ P2 + 17 burner)
**Estimated Total Effort**: 6-8 hours (3h P0 + 30m P1 + 2-3h P2/burner integration)
**Urgency**: **IMMEDIATE** - P0 files break user quickstart workflows

---

## 📊 Consolidated Findings Summary

### Phase 1 - Burner Docs (COMPLETED ✅)
- 7 gold burner docs audited
- 167 old path references analyzed
- All P0/P1 issues FIXED
- Ready for integration (Phase 5)

### Phase 2 - Canonical Docs (COMPLETED ✅)
- 194 canonical docs audited
- 47+ stale references found
- 7 P0 critical files identified
- Pending systematic fixes (Phase 4)

---

## 🚨 **PHASE 4A: P0 CRITICAL FIXES (IMMEDIATE)**

**Target**: Fix user-facing documentation with WRONG commands/paths
**Completion Time**: 2-3 hours
**Approach**: One file at a time, verify each fix, atomic commits

---

### **FILE 1: docs/user-guide/getting-started.md** ⚠️ HIGHEST PRIORITY

**Priority**: 🔴 P0-CRITICAL
**Impact**: **BREAKS QUICKSTART WORKFLOW** - First-run user experience
**Stale References**: 5 lines
**Estimated Effort**: 30 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 73 | "Saves trained model to \`models/\`" | "Saves trained model to \`experiments/checkpoints/\`" | Check training output |
| 88 | "Model saved to: models/boughter_vh_esm1v_logreg.pkl" | "Model saved to: experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl" | Run training, verify path |
| 102 | `ls -lh models/*.pkl` | `ls -lh experiments/checkpoints/**/*.pkl` | Test command works |
| 105 | `ls -lh embeddings_cache/` | `ls -lh experiments/cache/` | Test command works |
| 108 | `ls -lh logs/` | `ls -lh experiments/runs/*/logs/` | Test command works |

#### Verification Commands
```bash
# After fixing, verify all commands work:
ls -lh experiments/checkpoints/**/*.pkl
ls -lh experiments/cache/
ls -lh experiments/runs/*/logs/

# Verify old paths don't exist:
ls models/ 2>&1 | grep "No such file"
ls embeddings_cache/ 2>&1 | grep "No such file"
ls logs/ 2>&1 | grep "No such file"
```

#### Commit Message
```
docs: Fix getting-started.md paths for Phase 5 hierarchy

- Update model save path: models/ → experiments/checkpoints/
- Update cache path: embeddings_cache/ → experiments/cache/
- Update logs path: logs/ → experiments/runs/*/logs/
- Update verification commands with correct paths

Fixes: P0 critical - breaks quickstart workflow
Verified: All commands tested and working
```

---

### **FILE 2: USAGE.md**

**Priority**: 🔴 P0-CRITICAL
**Impact**: User command examples fail with "file not found"
**Stale References**: 3 lines
**Estimated Effort**: 20 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 206 | `antibody-test --model models/antibody_classifier.pkl` | `antibody-test --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl` | Run command, verify works |
| 209 | `models/model1.pkl models/model2.pkl` | `experiments/checkpoints/{...}/model1.pkl ...` | Check model paths |
| 261 | `with open('models/antibody_classifier.pkl', 'rb')` | `with open('experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl', 'rb')` | Test Python code |

#### Verification Commands
```bash
# Verify model path examples work:
ls -la experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl

# Test actual command:
uv run antibody-test --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl --dataset jain
```

#### Commit Message
```
docs: Fix USAGE.md model paths for Phase 5 hierarchy

- Update antibody-test examples: models/ → experiments/checkpoints/
- Update Python code examples with correct model paths
- Use realistic model filenames from hierarchical structure

Fixes: P0 critical - command examples fail
Verified: Commands tested and working
```

---

### **FILE 3: README.md**

**Priority**: 🔴 P0-CRITICAL
**Impact**: Root documentation references wrong directory
**Stale References**: 1 line
**Estimated Effort**: 10 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 162 | `outputs/` | `experiments/runs/` | Verify directory exists |

#### Verification Commands
```bash
# Verify new path exists:
ls -la experiments/runs/

# Verify old path doesn't exist:
ls outputs/ 2>&1 | grep "No such file"
```

#### Commit Message
```
docs: Fix README.md outputs/ path for Phase 5

- Update outputs/ → experiments/runs/

Fixes: P0 - root documentation accuracy
Verified: experiments/runs/ exists
```

---

### **FILE 4: docs/user-guide/troubleshooting.md**

**Priority**: 🔴 P0-HIGH
**Impact**: Troubleshooting commands fail
**Stale References**: 5 lines
**Estimated Effort**: 25 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 264 | `rm -rf embeddings_cache/` | `rm -rf experiments/cache/` | Test with dummy dir |
| 360 | `ls -lh embeddings_cache/` | `ls -lh experiments/cache/` | Test command |
| 406 | `rm -rf embeddings_cache/` | `rm -rf experiments/cache/` | Test with dummy dir |
| 425 | `rm -rf embeddings_cache/` | `rm -rf experiments/cache/` | Test with dummy dir |
| 1050 | `ls -lh configs/ models/ data/train/ data/test/` | `ls -lh experiments/ data/train/ data/test/` | Test command |

#### Verification Commands
```bash
# Test cache commands:
ls -lh experiments/cache/
# (Don't actually rm -rf, just verify path correct)

# Test diagnostic command:
ls -lh experiments/ data/train/ data/test/
```

#### Commit Message
```
docs: Fix troubleshooting.md paths for Phase 5

- Update cache clear: embeddings_cache/ → experiments/cache/
- Update diagnostic: configs/ models/ → experiments/
- All troubleshooting commands now reference correct paths

Fixes: P0 high - troubleshooting commands fail
Verified: All paths exist and commands work
```

---

### **FILE 5: docs/user-guide/training.md**

**Priority**: 🔴 P0-MEDIUM
**Impact**: Training guide config examples wrong
**Stale References**: 1 line
**Estimated Effort**: 10 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 372 | `model_save_dir: "models/"` | `model_save_dir: "experiments/checkpoints/"` | Check config example |

#### Verification Commands
```bash
# Verify directory exists:
ls -la experiments/checkpoints/
```

#### Commit Message
```
docs: Fix training.md model_save_dir config example

- Update model_save_dir: models/ → experiments/checkpoints/

Fixes: P0 medium - config example uses wrong path
Verified: experiments/checkpoints/ exists
```

---

### **FILE 6: docs/developer-guide/directory-organization.md**

**Priority**: 🔴 P0-HIGH
**Impact**: Developers misunderstand directory structure
**Stale References**: 1 line
**Estimated Effort**: 15 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 32 | "Structure: \`models/{model_shortname}/{classifier_type}/\`" | "Structure: \`experiments/checkpoints/{model_shortname}/{classifier_type}/\`" | Verify actual structure |

#### Verification Commands
```bash
# Verify actual structure:
ls -la experiments/checkpoints/
ls -la experiments/checkpoints/esm1v/logreg/
```

#### Commit Message
```
docs: Fix directory-organization.md model structure path

- Add experiments/checkpoints/ prefix to hierarchical structure
- Structure now: experiments/checkpoints/{model}/{classifier}/

Fixes: P0 high - architecture docs incorrect
Verified: Structure matches actual layout
```

---

### **FILE 7: docs/developer-guide/architecture.md**

**Priority**: 🔴 P0-HIGH
**Impact**: Core architecture doc wrong
**Stale References**: 1 line
**Estimated Effort**: 10 minutes

#### Updates Needed

| Line | Current | Fixed | Verification |
|------|---------|-------|--------------|
| 157 | "Trained models saved as .pkl files in \`models/\`" | "Trained models saved as .pkl files in \`experiments/checkpoints/\`" | Verify actual location |

#### Verification Commands
```bash
# Verify models are in experiments/checkpoints/:
find experiments/checkpoints/ -name "*.pkl" | head -5
```

#### Commit Message
```
docs: Fix architecture.md model save location

- Update models/ → experiments/checkpoints/

Fixes: P0 high - core architecture doc accuracy
Verified: Models actually saved to experiments/checkpoints/
```

---

## **P0 Summary**

| File | Priority | Lines | Effort | Status |
|------|----------|-------|--------|--------|
| docs/user-guide/getting-started.md | P0-CRITICAL | 5 | 30min | ⏸️ Pending |
| USAGE.md | P0-CRITICAL | 3 | 20min | ⏸️ Pending |
| README.md | P0-CRITICAL | 1 | 10min | ⏸️ Pending |
| docs/user-guide/troubleshooting.md | P0-HIGH | 5 | 25min | ⏸️ Pending |
| docs/user-guide/training.md | P0-MEDIUM | 1 | 10min | ⏸️ Pending |
| docs/developer-guide/directory-organization.md | P0-HIGH | 1 | 15min | ⏸️ Pending |
| docs/developer-guide/architecture.md | P0-HIGH | 1 | 10min | ⏸️ Pending |
| **TOTAL** | | **17** | **~2h** | |

---

## 🟡 **PHASE 4B: P1 CLARIFICATIONS (NEXT)**

**Target**: Add Phase 5 migration notes to historical docs
**Completion Time**: 30 minutes
**Approach**: Add inline annotations, header notes

---

### **FILE 8: CHANGELOG.md**

**Priority**: 🟡 P1-HIGH
**Impact**: Historical context confusion
**Stale References**: 4 lines
**Estimated Effort**: 15 minutes

#### Updates Needed

Add inline migration notes:
- Line 23: "outputs/{experiment.name}/" → Add "(now experiments/runs/, moved in v0.5.0)"
- Line 40: "outputs/" → Add "(moved to experiments/runs/ in v0.5.0)"
- Line 41: "outputs/.../logs/" → Add "(now experiments/runs/.../logs/)"
- Line 247: "embeddings_cache/" → Add "(now experiments/cache/)"

#### Commit Message
```
docs: Add Phase 5 migration notes to CHANGELOG

- Annotate historical paths with current locations
- Clarify outputs/ → experiments/runs/ migration (v0.5.0)
- Clarify embeddings_cache/ → experiments/cache/ migration

Fixes: P1 high - historical context clarity
```

---

### **FILE 9: docs/archive/TRAINING_SETUP_STATUS.md**

**Priority**: 🟡 P1-MEDIUM
**Impact**: Low (archived doc)
**Stale References**: 2 lines
**Estimated Effort**: 10 minutes

#### Updates Needed

Add header note:
```markdown
> **Historical Note**: This document predates Phase 5 reorganization (2025-11-16).
> Paths referenced here have been moved:
> - `embeddings_cache/` → `experiments/cache/`
> - `models/` → `experiments/checkpoints/`
```

#### Commit Message
```
docs: Add Phase 5 note to archived TRAINING_SETUP_STATUS

- Add header with path migration context
- Document embeddings_cache/ and models/ moves

Fixes: P1 medium - archived doc clarity
```

---

### **FILE 10: docs/archive/CODEBASE_REORGANIZATION_PLAN.md**

**Priority**: 🟡 P1-MEDIUM
**Impact**: Low (archived migration plan)
**Stale References**: 1 line
**Estimated Effort**: 5 minutes

#### Updates Needed

Add header note:
```markdown
> **Historical Note**: This reorganization plan predates Phase 5 (2025-11-16).
> The `models/` directory was subsequently moved to `experiments/checkpoints/` in Phase 5.
```

#### Commit Message
```
docs: Add Phase 5 note to archived CODEBASE_REORGANIZATION_PLAN

- Clarify this plan predates Phase 5
- Note models/ → experiments/checkpoints/ migration

Fixes: P1 medium - archived plan context
```

---

## **P1 Summary**

| File | Priority | Lines | Effort | Status |
|------|----------|-------|--------|--------|
| CHANGELOG.md | P1-HIGH | 4 | 15min | ⏸️ Pending |
| docs/archive/TRAINING_SETUP_STATUS.md | P1-MEDIUM | 2 | 10min | ⏸️ Pending |
| docs/archive/CODEBASE_REORGANIZATION_PLAN.md | P1-MEDIUM | 1 | 5min | ⏸️ Pending |
| **TOTAL** | | **7** | **30min** | |

---

## 🟢 **PHASE 4C: P2 INTEGRATION UPDATES (LATER)**

**Target**: Update to-be-integrated docs during Phase 5 (burner integration)
**Completion Time**: 1 hour
**Approach**: Update during integration process

---

### **FILE 11: docs/to-be-integrated/output_pipeline_architecture.md**

**Priority**: 🟢 P2-MEDIUM
**Stale References**: 20+ lines
**Estimated Effort**: 30 minutes

#### Updates Needed

Systematic replacement:
- `outputs/` → `experiments/runs/`
- `test_results/` → `experiments/benchmarks/`
- Update all directory structure examples

#### Approach

Update during burner doc integration (Phase 5).

---

### **FILE 12: docs/to-be-integrated/ESM2_FEATURE.md**

**Priority**: 🟢 P2-LOW
**Stale References**: 1 line
**Estimated Effort**: 5 minutes

#### Updates Needed

- Line 89: `outputs/<timestamp>/training.log` → `experiments/runs/{experiment}/{timestamp}/logs/training.log`

---

### **FILE 13: docs/archive/CLEANUP_COMPLETE_SUMMARY.md**

**Priority**: 🟢 P2-LOW
**Stale References**: 2 lines
**Estimated Effort**: 5 minutes

#### Updates Needed

Add note: "(test_results/ moved to experiments/benchmarks/ in Phase 5)"

---

## **P2 Summary**

| File | Priority | Lines | Effort | Status |
|------|----------|-------|--------|--------|
| docs/to-be-integrated/output_pipeline_architecture.md | P2-MEDIUM | 20+ | 30min | ⏸️ Phase 5 |
| docs/to-be-integrated/ESM2_FEATURE.md | P2-LOW | 1 | 5min | ⏸️ Phase 5 |
| docs/archive/CLEANUP_COMPLETE_SUMMARY.md | P2-LOW | 2 | 5min | ⏸️ Phase 5 |
| **TOTAL** | | **23+** | **40min** | |

---

## 📋 **Execution Checklist**

### Phase 4A: P0 Critical Fixes (IMMEDIATE)

- [ ] **FILE 1**: docs/user-guide/getting-started.md (30min)
- [ ] **FILE 2**: USAGE.md (20min)
- [ ] **FILE 3**: README.md (10min)
- [ ] **FILE 4**: docs/user-guide/troubleshooting.md (25min)
- [ ] **FILE 5**: docs/user-guide/training.md (10min)
- [ ] **FILE 6**: docs/developer-guide/directory-organization.md (15min)
- [ ] **FILE 7**: docs/developer-guide/architecture.md (10min)

**Total**: ~2 hours

### Phase 4B: P1 Clarifications

- [ ] **FILE 8**: CHANGELOG.md (15min)
- [ ] **FILE 9**: docs/archive/TRAINING_SETUP_STATUS.md (10min)
- [ ] **FILE 10**: docs/archive/CODEBASE_REORGANIZATION_PLAN.md (5min)

**Total**: ~30 minutes

### Phase 4C: P2 Integration Updates (During Phase 5)

- [ ] **FILE 11**: docs/to-be-integrated/output_pipeline_architecture.md (30min)
- [ ] **FILE 12**: docs/to-be-integrated/ESM2_FEATURE.md (5min)
- [ ] **FILE 13**: docs/archive/CLEANUP_COMPLETE_SUMMARY.md (5min)

**Total**: ~40 minutes

---

## 🎯 **Success Criteria**

Documentation update is COMPLETE when:

### P0 Success Criteria
- ✅ All quickstart commands in getting-started.md work
- ✅ All command examples in USAGE.md reference correct paths
- ✅ README.md references experiments/runs/ not outputs/
- ✅ Troubleshooting commands work with experiments/ hierarchy
- ✅ Training guide config examples use experiments/checkpoints/
- ✅ Architecture docs reflect experiments/ structure

### P1 Success Criteria
- ✅ CHANGELOG has migration notes for historical entries
- ✅ Archived docs have Phase 5 context headers
- ✅ No confusion about pre-Phase 5 vs post-Phase 5 paths

### P2 Success Criteria
- ✅ To-be-integrated docs updated during Phase 5 integration
- ✅ All references to test_results/ updated to experiments/benchmarks/
- ✅ All references to outputs/ updated to experiments/runs/

### Global Success Criteria
- ✅ Zero references to configs/config.yaml (except historical notes)
- ✅ Zero references to root-level models/, embeddings_cache/, logs/, outputs/
- ✅ All directory structure examples show experiments/ hierarchy
- ✅ All command examples verified working
- ✅ Git commits: atomic per file, clear messages

---

## 📊 **Final Summary**

| Phase | Files | Stale Refs | Effort | Priority |
|-------|-------|------------|--------|----------|
| **Phase 4A: P0** | 7 | 17 | 2h | 🔴 IMMEDIATE |
| **Phase 4B: P1** | 3 | 7 | 30m | 🟡 NEXT |
| **Phase 4C: P2** | 3 | 23+ | 40m | 🟢 LATER |
| **TOTAL** | **13** | **47+** | **~3h** | |

**Plus Phase 5 (Burner Integration)**: 7 burner docs → canonical docs (~2-3h)

**Grand Total Effort**: ~6-8 hours for 100% documentation accuracy

---

## 🚀 **Next Action**

**START PHASE 4A NOW** - Fix P0 critical files:

```bash
# Execute fixes in order:
# 1. docs/user-guide/getting-started.md (HIGHEST PRIORITY)
# 2. USAGE.md
# 3. README.md
# 4. docs/user-guide/troubleshooting.md
# 5. docs/user-guide/training.md
# 6. docs/developer-guide/directory-organization.md
# 7. docs/developer-guide/architecture.md

# After each fix:
# - Verify commands work
# - Mark checkbox complete
# - Commit with provided message template
```

---

**Plan Status**: ✅ **READY FOR EXECUTION**
**Created By**: Claude Code (synthesized from Phase 1 + Phase 2 audits)
**Next Step**: Execute Phase 4A (P0 critical fixes)
**Target**: Fix user-breaking documentation IMMEDIATELY
