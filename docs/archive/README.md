# Archive

This directory contains **historical documentation** from the development process. These documents represent completed work, investigations, and planning efforts that provide context for past decisions.

**Archive Policy:** Documents here are time-bound, completed, and serve as historical reference only. For current documentation, see the main `docs/` directory.

---

## Current Archive Files (Root Level)

**Note:** Subdirectories (`migrations/`, `investigations/`, `plans/`, `summaries/`) are being populated. Most archive documents still live at root level (`docs/archive/*.md`) but are gradually being organized into subdirectories.

### Migrations (4 docs)
Codebase reorganizations and structural changes:

| Current Path | Date | Description |
|--------------|------|-------------|
| `CODEBASE_REORGANIZATION_PLAN.md` | 2025-11-05 | v2.0.0 restructuring plan (root files → src/) |
| `TEST_DATASETS_REORGANIZATION_PLAN.md` | 2025-11 | Test dataset directory reorganization |
| `REPOSITORY_MODERNIZATION_PLAN.md` | 2025-11-06 | 2025 tooling upgrade plan (uv, ruff, mypy) |
| `migrations/v2-structure-migration.md` | 2025-11 | v2.0.0 import structure guide (moved from docs/development/) |

### Investigations (5 docs)
One-off debugging sessions and issue investigations:

| Current Path | Date | Description |
|--------------|------|-------------|
| `MPS_MEMORY_LEAK_FIX.md` | 2025-11-03 | Apple Silicon MPS memory leak fix (P0 bug) |
| `P0_SEMAPHORE_LEAK.md` | 2025-11-05 | Semaphore leak investigation |
| `SCRIPTS_AUDIT.md` | 2025-11-05 | Script audit report |
| `RESIDUAL_TYPE_ERRORS.md` | 2025-11 | Deferred type errors in preprocessing scripts |
| `investigations/p0-blockers.md` | 2025-11 | P0/P1/P2/P3 blocker tracking (moved from docs/development/) |

### Plans (3 docs)
Completed planning documents:

| Current Path | Date | Description |
|--------------|------|-------------|
| `CLEANUP_PLAN.md` | 2025-11-05 | Jain dataset cleanup execution plan |
| `STRICT_QC_CLEANUP_PLAN.md` | 2025-11 | Quality control cleanup plan |
| `TRAINING_SETUP_STATUS.md` | 2025-11 | Training setup status report |

### Summaries (4 docs)
Completion reports and status summaries:

| Current Path | Date | Description |
|--------------|------|-------------|
| `CLEANUP_COMPLETE_SUMMARY.md` | 2025-11-05 | Jain cleanup execution summary |
| `FIXES_APPLIED.md` | 2025-11 | Bug fixes and corrections log |
| `PHASE1_TEST_RESULTS.md` | 2025-11 | Phase 1 test results |
| `DOCS_AUDIT_STATUS.md` | 2025-11 | Documentation audit (pre-reorganization) |

---

## Planned Directory Structure (Phase 6)

After Phase 6 of the documentation reorganization, files will be organized into subdirectories:

```
archive/
├── README.md                     # This file
├── migrations/                   # Codebase reorganizations (4 docs)
│   └── v2-structure-migration.md
├── investigations/               # Debugging sessions (5 docs)
│   └── p0-blockers.md
├── plans/                        # Completed plans (3 docs)
└── summaries/                    # Completion reports (4 docs)
```

---

## Current Active Documentation

For current, evergreen documentation, see:

- **User Guides:** `docs/user-guide/` (pending Phase 3)
- **Developer Guides:** `docs/developer-guide/` (pending Phase 4)
- **Research Notes:** `docs/research/`
- **Dataset Documentation:** `docs/datasets/`
- **Development Documentation:** ~~`docs/development/`~~ (archived as of Phase 6)

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

**Migrations (4 docs):**
- Codebase reorganization (v2.0.0 structure)
- Test dataset reorganization
- Repository modernization (2025 tooling)
- v2.0.0 import structure guide

**Investigations (5 docs):**
- MPS memory leak fix (P0)
- Semaphore leak investigation
- Scripts audit
- Residual type errors (deferred work)
- P0/P1/P2/P3 blocker tracking

**Plans (3 docs):**
- Jain cleanup execution plan
- Strict QC cleanup plan
- Training setup status

**Summaries (4 docs):**
- Jain cleanup completion summary
- Bug fixes log
- Phase 1 test results
- Documentation audit (pre-reorg)

**Total:** 16 archived documents

---

## Related Documentation

- **Documentation Plan:** `docs/DOCUMENTATION_STRUCTURE_PLAN.md` (active, guides reorganization)
- **Current Development Status:** See `docs/development/` for active work
- **Research Findings:** See `docs/research/` for scientific methodology

---

**Last Updated:** 2025-11-10
**Branch:** `docs/canonical-structure`
