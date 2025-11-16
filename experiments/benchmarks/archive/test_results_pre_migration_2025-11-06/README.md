# Test Results - Pre-Migration Baseline (Nov 6-12, 2025)

**Archive Date**: 2025-11-15
**Status**: Historical benchmark results preserved for reference

---

## Context

These are historical benchmark results generated **before** the Phase 2 model directory migration (hierarchical model organization). They represent baseline performance metrics for cross-dataset validation.

### Timeline
- **ESM1v Results**: Nov 6, 2025
- **ESM2-650M Results**: Nov 11-12, 2025
- **Migration Date**: Nov 13, 2025 (Phase 2 - hierarchical model directories)
- **Archive Date**: Nov 15, 2025 (Output organization cleanup)

### Models Tested
- **ESM1v** (`facebook/esm1v_t33_650M_UR90S_1`) + Logistic Regression
- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`) + Logistic Regression

### Test Datasets
- **Jain** - Novo Nordisk parity benchmark (86 clinical antibodies)
- **Harvey** - Nanobody PSR assay (141k VHH sequences)
- **Shehata** - B-cell PSR assay (398 antibodies)

---

## Archive Reason

**Model Path Migration**: These results use the **old flat model directory structure**:
```
models/boughter_vh_esm1v_logreg.pkl  # OLD
```

New hierarchical structure (post-migration):
```
models/esm1v/logreg/boughter_vh_esm1v_logreg.pkl  # NEW
```

**Status**: Results remain scientifically valid, but file paths in YAML configs reference pre-migration model locations.

---

## Metrics Summary

### Jain (Novo Parity Benchmark)

**ESM1v + LogReg**:
- Accuracy: 66.28% ✅ (exact Novo Nordisk parity)
- Confusion Matrix: `[[40, 19], [10, 17]]`
- Dataset: 86 antibodies (59 specific / 27 non-specific)

**ESM2-650M + LogReg**:
- Results in `esm2_650m/logreg/VH_only_jain_test_PARITY_86/`

### Harvey (Nanobody PSR Assay)

**ESM1v + LogReg**:
- Results in `esm1v/logreg/harvey/`
- Dataset: 141k VHH sequences

**ESM2-650M + LogReg**:
- Results in `esm2_650m/logreg/VHH_only_harvey/`

### Shehata (PSR Assay)

**ESM1v + LogReg**:
- Results in `esm1v/logreg/shehata/`
- Dataset: 398 antibodies (391 specific / 7 non-specific)

**ESM2-650M + LogReg**:
- Results in `esm2_650m/logreg/VH_only_shehata/`

---

## Directory Structure

```
test_results_pre_migration_2025-11-06/
├── README.md (this file)
├── esm1v/
│   └── logreg/
│       ├── harvey/
│       │   ├── confusion_matrix_VHH_only_harvey.png
│       │   ├── detailed_results_VHH_only_harvey_20251106_223905.yaml
│       │   └── predictions_boughter_vh_esm1v_logreg_VHH_only_harvey_20251106_223905.csv
│       ├── jain/
│       │   ├── confusion_matrix_VH_only_jain_test_PARITY_86.png
│       │   ├── detailed_results_VH_only_jain_test_PARITY_86_20251106_211815.yaml
│       │   └── predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251106_211815.csv
│       └── shehata/
│           ├── confusion_matrix_VH_only_shehata.png
│           ├── detailed_results_VH_only_shehata_20251106_212500.yaml
│           └── predictions_boughter_vh_esm1v_logreg_VH_only_shehata_20251106_212500.csv
│
└── esm2_650m/
    └── logreg/
        ├── VHH_only_harvey/
        │   ├── confusion_matrix_boughter_vh_esm2_650m_logreg_VHH_only_harvey.png
        │   ├── detailed_results_boughter_vh_esm2_650m_logreg_VHH_only_harvey_20251112_051907.yaml
        │   └── predictions_boughter_vh_esm2_650m_logreg_VHH_only_harvey_20251112_051907.csv
        ├── VH_only_jain_test_PARITY_86/
        │   ├── confusion_matrix_boughter_vh_esm2_650m_logreg_VH_only_jain_test_PARITY_86.png
        │   ├── detailed_results_boughter_vh_esm2_650m_logreg_VH_only_jain_test_PARITY_86_20251111_235027.yaml
        │   └── predictions_boughter_vh_esm2_650m_logreg_VH_only_jain_test_PARITY_86_20251111_235027.csv
        └── VH_only_shehata/
            ├── confusion_matrix_boughter_vh_esm2_650m_logreg_VH_only_shehata.png
            ├── detailed_results_boughter_vh_esm2_650m_logreg_VH_only_shehata_20251111_235531.yaml
            └── predictions_boughter_vh_esm2_650m_logreg_VH_only_shehata_20251111_235531.csv
```

---

## Usage Notes

### For Paper Writing
- ✅ Use these metrics as baseline comparisons
- ✅ Confusion matrices remain valid for figures
- ✅ Predictions CSVs can be used for error analysis

### For Reproducibility
- ⚠️ Model paths in YAML configs are outdated
- ⚠️ To regenerate, use new hierarchical model paths
- ✅ All data paths still valid (test datasets unchanged)

### For Future Benchmarks
- ✅ New test results should go in `/test_results/` (now empty)
- ✅ Use hierarchical directory structure: `{model}/{classifier}/{dataset}/`
- ✅ Archive old results when methodology changes

---

## Related Documents

- `POST_MIGRATION_VALIDATION_SUMMARY.md` - Migration validation results
- `OUTPUT_DIRECTORY_INVESTIGATION.md` - Output organization analysis
- `OUTPUT_ORGANIZATION_FINAL_CLEANUP_PLAN.md` - This cleanup's implementation plan

---

**Archived By**: Claude Code (Autonomous Cleanup Agent)
**Archive Purpose**: Preserve historical baselines while cleaning up active working directory
**Status**: ✅ Complete, validated, preserved for reference
