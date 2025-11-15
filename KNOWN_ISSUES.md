# Known Issues & Historical Notes

**Status**: Documentation of known issues, broken states, and technical debt
**Created**: 2025-11-15
**Purpose**: Track issues that are intentionally not fixed (with justification)

---

## Issue 1: Legacy Config Broken (configs/config.yaml)

**Status**: ⚠️ KNOWN BROKEN - Will not fix (scheduled for deletion in v0.5.0)

**Issue**:
```bash
# configs/config.yaml line 20 references non-existent file:
test_file: ./data/test/jain/canonical/VH_only_jain_P5e_S2.csv

# File does not exist:
$ ls data/test/jain/canonical/VH_only_jain_P5e_S2.csv
ls: No such file or directory

# Actual file (created during Phase 1 migration):
data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv  ✅ EXISTS
```

**Root Cause**:
- During Phase 1 migration (test_datasets → data/test), Jain canonical files were created with lowercase `p5e_s2` and `_86_` identifier
- Legacy `configs/config.yaml` was not updated to match (references capital `P5e` without `_86_`)
- Hydra config (`src/antibody_training_esm/conf/data/boughter_jain.yaml`) was correctly updated

**Impact**:
- **HIGH** for users of legacy config (training will fail with FileNotFoundError)
- **NONE** for users of Hydra config (current production system - works correctly)
- All modern code paths use Hydra config, not legacy

**Why Not Fixed**:
1. Legacy config is deprecated (as of v0.4.0)
2. Scheduled for deletion in v0.5.0 (see V0.5.0_CLEANUP_PLAN.md)
3. Hydra config is correct and all production code uses Hydra
4. No users reported issues (no one using legacy config in practice)
5. Fixing dead code is wasted effort

**Workaround**:
Use Hydra config instead:
```bash
# DON'T USE THIS (broken):
python -c "from antibody_training_esm.core.trainer import train_model; train_model('configs/config.yaml')"

# USE THIS (works):
uv run antibody-train  # Uses Hydra config automatically
```

**Timeline**:
- Broken since: Phase 1 migration (commit 288905c, Nov 14 2025)
- Discovered: Nov 15 2025 (during POST_MIGRATION_VALIDATION_PLAN.md review)
- Will be deleted: v0.5.0 (estimated Dec 2025)

**References**:
- V0.5.0_CLEANUP_PLAN.md (Problem 1: Remove configs/config.yaml)
- POST_MIGRATION_VALIDATION_PLAN.md (Check 3: Training Config Validation)
- src/antibody_training_esm/conf/data/boughter_jain.yaml (correct config)

---

## Issue 2: [Template for future issues]

**Status**: [OPEN / KNOWN BROKEN / WONT FIX / FIXED]

**Issue**: [Description]

**Root Cause**: [Why it happened]

**Impact**: [Who is affected]

**Why Not Fixed**: [Justification]

**Workaround**: [Alternative solution]

**Timeline**: [When broken, when fixed/deleted]

**References**: [Related docs]

---

## Maintenance Notes

**When to add issues here**:
- Known bugs that won't be fixed immediately
- Technical debt that's intentionally deferred
- Broken legacy code that's scheduled for deletion
- Breaking changes that need historical context

**When NOT to add issues here**:
- Active bugs being worked on (use GitHub issues)
- Unknown/undocumented issues
- Minor typos or formatting issues

**Review frequency**: Quarterly (or before major releases)

**Cleanup policy**: Delete issues when:
- Code is deleted (e.g., Issue 1 deleted when v0.5.0 ships)
- Issue is actually fixed
- Issue is no longer relevant (dependencies removed, etc.)
