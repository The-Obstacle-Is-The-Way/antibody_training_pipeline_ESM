# Documentation Update Roadmap - Systematic Audit & Integration Plan

**Created**: 2025-11-17
**Status**: 📋 **PLANNING** - Ready for systematic execution
**Purpose**: Ensure 100% documentation accuracy with current codebase (post-Phase 5)
**Philosophy**: Measure twice, cut once (Rob C. Martin discipline)

---

## 🎯 **Objective**

Systematically audit and update ALL documentation to match the current repository state:
- ✅ Phase 5 reorganization COMPLETE (experiments/ consolidation)
- ✅ Legacy `configs/config.yaml` DELETED
- ✅ Root-level output directories MOVED to `experiments/`

**No half-measures, no stale references, 100% accuracy.**

---

## 📊 **Current State (2025-11-17)**

### ✅ **What Exists NOW**

**Directory Structure:**
```
experiments/
├── benchmarks/        # Published results (versioned)
├── cache/             # Embeddings cache (gitignored)
├── checkpoints/       # Trained models (gitignored)
└── runs/              # Hydra outputs (gitignored)
```

**Configuration:**
- `src/antibody_training_esm/conf/config.yaml` ✅ (Hydra config)
- `configs/config.yaml` ❌ **DELETED** (legacy, removed in Phase 5)

### ❌ **What NO LONGER Exists**

**Root-level directories (MOVED in Phase 5):**
- `models/` → `experiments/checkpoints/`
- `embeddings_cache/` → `experiments/cache/`
- `logs/` → `experiments/runs/logs/`
- `outputs/` → `experiments/runs/`

**Legacy files (DELETED):**
- `configs/config.yaml` (legacy training config)

---

## ⚠️ **The Problem: Intermittent Staleness**

Burner docs were written at different times during migration/cleanup. They may reference:
- ❌ Files that no longer exist (`configs/config.yaml`)
- ❌ Old directory structures (`models/`, `embeddings_cache/`, `logs/`)
- ❌ Issues that are already fixed (KNOWN_ISSUES.md)
- ❌ Pre-Phase 5 paths

**Canonical docs may ALSO be stale** if they were written before Phase 5.

**THIS IS WHY WE AUDIT BEFORE INTEGRATION.**

---

## 📋 **Systematic Execution Plan**

### **Phase 0: Baseline Inventory** (30 min) - **NEXT**

Create complete inventory of ALL documentation files.

**Tasks:**
```bash
# 1. List all markdown files in repository
find . -name "*.md" -type f | grep -v "node_modules\|\.git" | sort > /tmp/all_docs.txt

# 2. Categorize by location
echo "=== ROOT DOCS ===" && ls -1 *.md
echo "=== BURNER DOCS (GOLD) ===" && ls -1 docs_burner/implementation/*.md
echo "=== CANONICAL DOCS ===" && find docs -name "*.md" | wc -l
echo "=== EXPERIMENTS DOCS ===" && find experiments/benchmarks -name "*.md" | wc -l
```

**Deliverable**: `DOCUMENTATION_INVENTORY.md` with complete file list + categorization

---

### **Phase 1: Audit Burner Docs (Gold Files)** (2-3 hours)

Audit 7 gold burner docs for stale references.

**Files to Audit:**
1. `VALIDATION_ROADMAP.md`
2. `REPOSITORY_REORGANIZATION_PLAN.md`
3. `TEST_COVERAGE_PLAN.md`
4. `V0.5.0_CLEANUP_PLAN.md`
5. `REPOSITORY_STRUCTURE_ANALYSIS.md`
6. `VALIDATION_REALITY.md`
7. `OUTPUT_DIRECTORY_INVESTIGATION.md`

**Audit Checklist (per file):**
- [ ] Check for references to `configs/config.yaml` (DELETED)
- [ ] Check for references to `models/` at root (should be `experiments/checkpoints/`)
- [ ] Check for references to `embeddings_cache/` at root (should be `experiments/cache/`)
- [ ] Check for references to `logs/` at root (should be `experiments/runs/logs/`)
- [ ] Check for references to `outputs/` at root (should be `experiments/runs/`)
- [ ] Verify all file paths exist in current codebase
- [ ] Verify all commands still work
- [ ] Mark stale sections with `⚠️ STALE:` tag

**Method:**
```bash
# For each file:
# 1. Read entire file
# 2. Grep for old paths
grep -n "configs/config.yaml\|models/\|embeddings_cache/\|logs/\|outputs/" <file>
# 3. Verify each path reference exists
# 4. Document findings in audit report
```

**Deliverable**: `BURNER_DOCS_AUDIT_REPORT.md` with:
- File-by-file staleness findings
- Specific lines that need updating
- Priority (P0: critical, P1: high, P2: low)

---

### **Phase 2: Audit Canonical Docs** (3-4 hours)

Audit main documentation for Phase 5 compatibility.

**Categories:**

#### **2.1: Root Documentation** (30 min)
- `README.md`
- `CLAUDE.md`
- `USAGE.md`
- `ROADMAP.md`

**Check for:**
- Directory structure examples (should show `experiments/` hierarchy)
- Command examples (should reference new paths)
- Installation/setup instructions

---

#### **2.2: Developer Guides** (1 hour)
- `docs/developer-guide/architecture.md`
- `docs/developer-guide/development-workflow.md`
- `docs/developer-guide/directory-organization.md` ⚠️ **HIGH PRIORITY**
- `docs/developer-guide/testing-strategy.md`
- `docs/developer-guide/ci-cd.md`
- `docs/developer-guide/docker.md`

**Check for:**
- Directory structure diagrams
- Output path references
- Example commands

---

#### **2.3: User Guides** (1 hour)
- `docs/user-guide/getting-started.md`
- `docs/user-guide/installation.md`
- `docs/user-guide/training.md` ⚠️ **HIGH PRIORITY**
- `docs/user-guide/testing.md` ⚠️ **HIGH PRIORITY**
- `docs/user-guide/preprocessing.md`

**Check for:**
- Quickstart commands
- File path examples
- Expected output locations

---

#### **2.4: Research Documentation** (30 min)
- `docs/research/novo-parity.md`
- `docs/research/methodology.md`
- `docs/research/benchmark-results.md`
- `docs/research/assay-thresholds.md`

**Check for:**
- Model path references
- Test result paths
- Reproducibility commands

---

#### **2.5: Dataset Documentation** (30 min)
- `docs/datasets/boughter/`
- `docs/datasets/jain/`
- `docs/datasets/harvey/`
- `docs/datasets/shehata/`

**Check for:**
- Data file paths
- Preprocessing output paths

---

**Deliverable**: `CANONICAL_DOCS_AUDIT_REPORT.md` with:
- File-by-file audit results
- Categorized by doc type (root, developer, user, research, datasets)
- Priority ranking
- Estimated effort per file

---

### **Phase 3: Create Update Plan** (1 hour)

Synthesize audit findings into prioritized update plan.

**Deliverable**: `DOCUMENTATION_UPDATE_PLAN.md` with:

| File | Status | Updates Needed | Priority | Effort | Owner |
|------|--------|----------------|----------|--------|-------|
| README.md | ⚠️ Stale | Update directory structure | P0 | 30min | - |
| CLAUDE.md | ⚠️ Stale | Update paths, commands | P0 | 45min | - |
| docs/developer-guide/directory-organization.md | ❌ Critical | Complete rewrite | P0 | 2h | - |
| ... | ... | ... | ... | ... | ... |

**Priority Levels:**
- **P0 - Critical**: User-facing docs with stale commands (breaks quickstart)
- **P1 - High**: Developer docs with stale paths (breaks development workflow)
- **P2 - Medium**: Research/dataset docs (informational only)
- **P3 - Low**: Historical/archived docs (reference only)

---

### **Phase 4: Execute Updates** (4-6 hours)

Work through update plan systematically.

**Execution Rules:**
1. ✅ **One file at a time** (no batch updates)
2. ✅ **Verify after each update** (test commands, check paths)
3. ✅ **Mark complete in plan** (track progress)
4. ✅ **Commit frequently** (small, atomic commits)

**Update Template:**
```bash
# For each file:
# 1. Read current version
# 2. Identify stale references (from audit)
# 3. Update with current paths/commands
# 4. Verify accuracy (test commands if applicable)
# 5. Mark complete in DOCUMENTATION_UPDATE_PLAN.md
# 6. Commit: "docs: Update <file> with Phase 5 paths"
```

**Commit Message Format:**
```
docs: Update <filename> with Phase 5 paths

- Replace models/ with experiments/checkpoints/
- Replace embeddings_cache/ with experiments/cache/
- Replace configs/config.yaml with src/antibody_training_esm/conf/config.yaml
- Verify all commands work

Fixes: Stale path references from pre-Phase 5 state
```

---

### **Phase 5: Integrate Burner Docs** (2-3 hours)

After burner docs are audited and corrected, integrate into canonical docs.

**Integration Plan:**

| Burner Doc | Destination | Action | Effort |
|------------|-------------|--------|--------|
| VALIDATION_ROADMAP.md | docs/developer-guide/validation-checklist.md | Extract checklist, merge | 1h |
| REPOSITORY_REORGANIZATION_PLAN.md | docs/architecture/experiments-structure.md | Document Phase 5 | 45min |
| TEST_COVERAGE_PLAN.md | docs/developer-guide/testing-strategy.md | Append coverage section | 30min |
| V0.5.0_CLEANUP_PLAN.md | docs/archive/migrations/v0.5.0-legacy-removal.md | Archive migration plan | 15min |
| REPOSITORY_STRUCTURE_ANALYSIS.md | docs/architecture/repository-evolution.md | Document evolution | 30min |
| VALIDATION_REALITY.md | Merge into validation-checklist.md | Merge as "Reality Check" | 15min |
| OUTPUT_DIRECTORY_INVESTIGATION.md | Extract routing docs → preprocessing-internals.md | Extract useful sections | 30min |

**After integration:**
```bash
# Move integrated files to trash
mv docs_burner/implementation/<file> docs_burner/trash/integrated/

# Keep trash folder for reference (not committed to git)
```

---

### **Phase 6: Final Verification** (1 hour)

Verify 100% documentation accuracy.

**Verification Checklist:**

#### **6.1: Path Audit**
```bash
# Find any remaining old path references
grep -r "configs/config\.yaml" docs/ --include="*.md" || echo "✅ CLEAN"
grep -r "models/" docs/ --include="*.md" | grep -v "experiments/checkpoints\|ESM models\|# models" || echo "✅ CLEAN"
grep -r "embeddings_cache/" docs/ --include="*.md" | grep -v "experiments/cache" || echo "✅ CLEAN"
grep -r "^logs/" docs/ --include="*.md" | grep -v "experiments/runs/logs" || echo "✅ CLEAN"
```

#### **6.2: Command Verification**
- [ ] Test quickstart commands from README.md
- [ ] Test training commands from USAGE.md
- [ ] Test testing commands from docs/user-guide/testing.md
- [ ] Verify all example commands work

#### **6.3: Link Verification**
```bash
# Check for broken internal links
# (optional: use markdown-link-check)
npx markdown-link-check docs/**/*.md
```

**Deliverable**: `DOCUMENTATION_VERIFICATION_REPORT.md` with:
- All verification checks passed ✅
- Any remaining issues documented
- Sign-off: "Documentation is 100% accurate as of 2025-11-17"

---

## 📊 **Success Criteria**

Documentation update is COMPLETE when:

- ✅ Zero references to `configs/config.yaml` in non-archived docs
- ✅ Zero references to root-level `models/`, `embeddings_cache/`, `logs/` (except in migration docs)
- ✅ All directory structure examples show `experiments/` hierarchy
- ✅ All command examples use current paths
- ✅ All quickstart commands verified working
- ✅ Burner docs integrated into canonical docs
- ✅ Audit reports document complete process
- ✅ Final verification report signed off

---

## 🗂️ **Deliverables (7 documents)**

1. ✅ **DOCUMENTATION_INVENTORY.md** (Phase 0) - COMPLETE (212 files cataloged)
2. ✅ **BURNER_DOCS_AUDIT_REPORT.md** (Phase 1) - COMPLETE (7 files audited, 167 old path refs analyzed)
3. ✅ **CANONICAL_DOCS_AUDIT_REPORT.md** (Phase 2) - COMPLETE (194 files audited, 47+ stale refs, 7 P0 critical)
4. ⏸️ **DOCUMENTATION_UPDATE_PLAN.md** (Phase 3) - Pending
5. ⏸️ **DOCUMENTATION_VERIFICATION_REPORT.md** (Phase 6) - Pending
6. ✅ **This roadmap** (tracking progress) - COMPLETE
7. ⏸️ **Commit history** (atomic updates) - Ongoing

---

## 📈 **Progress Tracking**

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| Phase 0: Inventory | ✅ **COMPLETE** | 100% | DOCUMENTATION_INVENTORY.md created (212 files cataloged) |
| Phase 1: Burner Audit | ✅ **COMPLETE** | 100% | BURNER_DOCS_AUDIT_REPORT.md created (7/7 audited, all P0/P1 issues fixed) |
| Phase 2: Canonical Audit | ✅ **COMPLETE** | 100% | CANONICAL_DOCS_AUDIT_REPORT.md created (194 files audited, 47+ stale refs) |
| Phase 3: Update Plan | ✅ **COMPLETE** | 100% | DOCUMENTATION_UPDATE_PLAN.md created (synthesized Phase 1+2) |
| Phase 4: Execute Updates | 🔄 **IN PROGRESS** | 70% | Phase 4A P0 fixes COMPLETE (7/7 files), P1/P2 pending |
| Phase 5: Integration | ⏸️ Pending | 0% | Integrate burner docs |
| Phase 6: Verification | ⏸️ Pending | 0% | Final checks |

**Estimated Total Effort**: 12-16 hours
**Target Completion**: TBD (start Phase 0 now)

---

## 🎯 **Next Action**

**START PHASE 0** - Create documentation inventory:

```bash
# Execute baseline inventory
find . -name "*.md" -type f | grep -v "node_modules\|\.git\|\.venv" | sort > /tmp/all_docs.txt
wc -l /tmp/all_docs.txt
```

Then create `DOCUMENTATION_INVENTORY.md` with categorized file list.

---

## 📝 **Notes**

**Why This Approach?**
- ✅ **Systematic**: Phase-by-phase, no skipping steps
- ✅ **Verifiable**: Audit reports document what was checked
- ✅ **Trackable**: Progress tracking in this roadmap
- ✅ **Reversible**: Small commits, easy to rollback
- ✅ **Complete**: 100% accuracy, no half-measures

**Philosophy**: Rob C. Martin discipline - measure twice, cut once. Audit BEFORE integration, verify AFTER updates.

---

**Roadmap Status**: ✅ **READY FOR EXECUTION**
**Created By**: Claude Code (validated from first principles)
**Approved By**: Senior Developer (awaiting sign-off)
**Next Step**: Execute Phase 0 (Baseline Inventory)
