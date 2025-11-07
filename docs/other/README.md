# Documentation

This directory contains technical documentation for the antibody training pipeline.

---

## 🎯 **FOR NOVO NORDISK PARITY RESULTS (FINAL)**

**The authoritative reverse engineering results are located in:**

📁 **`experiments/novo_parity/`**

**Key documents:**
- **Executive Summary**: `experiments/novo_parity/MISSION_ACCOMPLISHED.md`
- **Technical Details**: `experiments/novo_parity/EXACT_MATCH_FOUND.md`
- **Final Dataset**: `experiments/novo_parity/datasets/jain_86_p5e_s2.csv`

**Result**: [[40, 19], [10, 17]] ✅ **EXACT MATCH** to Novo Nordisk's confusion matrix (66.28% accuracy)

---

## Current Documentation

### Core Technical Docs

- **`excel_to_csv_conversion_methods.md`** - Data processing and conversion methods
- **`FIXES_APPLIED.md`** - Bug fixes and corrections
- **`MPS_MEMORY_LEAK_FIX.md`** - Memory leak fix for MPS backend
- **`TRAINING_SETUP_STATUS.md`** - Model training setup and configuration
- **`METHODOLOGY_AND_DIVERGENCES.md`** - Overall methodology comparison with Novo
- **`ASSAY_SPECIFIC_THRESHOLDS.md`** - ELISA vs PSR threshold handling

### Cleanup & Audit Docs

- **`DOCS_AUDIT_STATUS.md`** - Documentation audit results
- **`CLEANUP_PLAN.md`** - Cleanup execution plan

---

## Archive

The `archive/` directory contains historical documentation from the reverse engineering process:

- **`failed_attempts/`** - Incorrect reverse engineering attempts (pre-P5e)
- **`p5_close_attempt/`** - P5 result (2 cells off from exact match)
- **`key_insights/`** - Important discoveries (e.g., mathematical proof)
- **`preprocessing/`** - 137→116 antibody QC documentation
- **`historical/`** - Pre-parity training results and old analyses

See `archive/README.md` for details.

---

## Dataset-Specific Documentation

- **`boughter/`** - Boughter dataset documentation
- **`harvey/`** - Harvey dataset documentation
- **`jain/`** - Jain dataset documentation (historical - see experiments/novo_parity/ for final)
- **`shehata/`** - Shehata dataset documentation
- **`investigation/`** - Various investigations

---

**Last Updated**: November 3, 2025
**Branch**: `novo-parity-exp-cleaned`
