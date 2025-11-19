# Utility Scripts

This directory contains standalone utility scripts for migrations, validation, and testing.

## Design Decision: Not a Python Package

**Note:** This directory intentionally does NOT contain `__init__.py` files.

### Why?

- **scripts/** = Run-only utilities (not importable modules)
- **src/antibody_training_esm/** = Importable package (library code)

Scripts are meant to be executed directly (e.g., `python scripts/migrate_*.py`), not imported as modules.

### Directory Structure

```
scripts/
├── README.md                    # This file
├── testing/                     # Educational demos
├── validation/                  # Cross-dataset validation
└── migrate_*.py                 # One-time migration utilities
```

### Running Scripts

```bash
# From project root:
python scripts/migrate_model_directories.py
python scripts/validation/validate_fragments.py

# Or with uv:
uv run python scripts/migrate_model_directories.py
```

### Design Philosophy

**Run-only vs. Importable:**

- **Run-only (scripts/)**: Standalone scripts that are executed directly
  - One-time migrations
  - Ad-hoc validation utilities
  - Testing/debugging helpers
  - NO `__init__.py` (not a package)

- **Importable (src/antibody_training_esm/)**: Reusable library code
  - Core ML pipeline components
  - Dataset loaders
  - Utilities used by multiple modules
  - HAS `__init__.py` (proper package)

This separation keeps the codebase clean and makes it clear which code is meant to be imported vs. executed.

---

**Last Updated:** 2025-11-18
**Status:** ✅ By Design (Not a Bug)
