# Archive

This directory contains **historical documentation** from the development process. These documents represent completed work, investigations, and planning efforts that provide context for past decisions.

**Archive Policy:** Documents here are time-bound, completed, and serve as historical reference only. For current documentation, see the main `docs/` directory.

---

## Directory Structure

### `migrations/`
Codebase reorganizations and structural changes:

| File | Date | Description |
|------|------|-------------|
| `CODEBASE_REORGANIZATION_PLAN.md` | 2025-11-05 | v2.0.0 restructuring plan (root files → src/) |
| `TEST_DATASETS_REORGANIZATION_PLAN.md` | 2025-11 | Test dataset directory reorganization |
| `REPOSITORY_MODERNIZATION_PLAN.md` | 2025-11-06 | 2025 tooling upgrade plan (uv, ruff, mypy) |

### `investigations/`
One-off debugging sessions and issue investigations:

| File | Date | Description |
|------|------|-------------|
| `MPS_MEMORY_LEAK_FIX.md` | 2025-11-03 | Apple Silicon MPS memory leak fix (P0 bug) |
| `P0_SEMAPHORE_LEAK.md` | 2025-11-05 | Semaphore leak investigation |
| `SCRIPTS_AUDIT.md` | 2025-11-05 | Script audit report |
| `RESIDUAL_TYPE_ERRORS.md` | 2025-11 | Deferred type errors in preprocessing scripts |

### `plans/`
Completed planning documents:

| File | Date | Description |
|------|------|-------------|
| `CLEANUP_PLAN.md` | 2025-11-05 | Jain dataset cleanup execution plan |
| `STRICT_QC_CLEANUP_PLAN.md` | 2025-11 | Quality control cleanup plan |
| `TRAINING_SETUP_STATUS.md` | 2025-11 | Training setup status report |

### `summaries/`
Completion reports and status summaries:

| File | Date | Description |
|------|------|-------------|
| `CLEANUP_COMPLETE_SUMMARY.md` | 2025-11-05 | Jain cleanup execution summary |
| `FIXES_APPLIED.md` | 2025-11 | Bug fixes and corrections log |
| `PHASE1_TEST_RESULTS.md` | 2025-11 | Phase 1 test results |
| `DOCS_AUDIT_STATUS.md` | 2025-11 | Documentation audit (pre-reorganization) |

---

## Current Active Documentation

For current, evergreen documentation, see:

- **User Guides:** `docs/user-guide/` (pending Phase 3)
- **Developer Guides:** `docs/developer-guide/` (pending Phase 4)
- **Research Notes:** `docs/research/`
- **Dataset Documentation:** `docs/datasets/`
- **Development Documentation:** `docs/development/`

---

## Archive Criteria

A document belongs in `archive/` if it meets ALL of these criteria:

1. ✅ **Complete** - The work described is finished
2. ✅ **Time-bound** - Represents a specific point in time or completed project
3. ✅ **Historical** - Provides context for past decisions but not current operations
4. ✅ **Superseded** - Information may be outdated or replaced by current practices

**Example:** `MPS_MEMORY_LEAK_FIX.md` is archived because the bug is fixed, the investigation is complete, and the fix is merged. The knowledge is valuable for historical context but not needed for daily operations.

---

## Index of All Archived Documents

### By Category

**Migrations (3 docs):**
- Codebase reorganization (v2.0.0 structure)
- Test dataset reorganization
- Repository modernization (2025 tooling)

**Investigations (4 docs):**
- MPS memory leak fix (P0)
- Semaphore leak investigation
- Scripts audit
- Residual type errors (deferred work)

**Plans (3 docs):**
- Jain cleanup execution plan
- Strict QC cleanup plan
- Training setup status

**Summaries (4 docs):**
- Jain cleanup completion summary
- Bug fixes log
- Phase 1 test results
- Documentation audit (pre-reorg)

**Total:** 14 archived documents

---

## Related Documentation

- **Documentation Plan:** `docs/DOCUMENTATION_STRUCTURE_PLAN.md` (active, guides reorganization)
- **Current Development Status:** See `docs/development/` for active work
- **Research Findings:** See `docs/research/` for scientific methodology

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
