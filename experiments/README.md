# Experiments Archive

This directory contains experimental investigations that were explored but not adopted for the production pipeline.

Experiments are archived here to:
1. **Preserve provenance** - Document what was tried and why
2. **Show scientific rigor** - Demonstrate hypothesis testing and validation
3. **Enable reproducibility** - Allow others to resurrect or learn from experiments
4. **Maintain clarity** - Keep production pipeline clean and unambiguous

---

## Archived Experiments

### strict_qc_2025-11-04

**Status:** ⚠️ ARCHIVED - Hypothesis disproven
**Date:** November 4-6, 2025
**Branch:** leroy-jenkins/full-send

**Hypothesis:** Removing ALL X amino acids (not just in CDRs) would better match Novo Nordisk's methodology and improve model performance.

**What was tested:**
- Created Stage 4 filtering: 914 → 852 sequences (VH fragment)
- Removed 62 sequences with X in framework regions
- Trained model and compared to production (914-sequence) model

**Results:**
- Production (914 seqs): 67.5% ± 8.9% CV accuracy
- Experimental (852 seqs): 66.55% ± 7.07% CV accuracy
- **No statistically significant improvement**

**Outcome:**
- Hypothesis disproven: stricter QC did not improve performance
- Production model externally validated instead:
  - Jain (HIC retention): 66.28% accuracy ✅
  - Shehata (PSR assay): 52.26% accuracy ✅
- Experiment archived, production model deployed

**Key learning:** The 62 sequences with X in frameworks were valid training data, not noise. ESM embeddings already handle ambiguous positions effectively.

**See:** `strict_qc_2025-11-04/EXPERIMENT_README.md` for complete details

**Contents:**
```
strict_qc_2025-11-04/
├── EXPERIMENT_README.md           # Complete experimental report
├── preprocessing/
│   ├── stage4_additional_qc.py    # Stage 4 filtering implementation
│   └── validate_stage4.py         # Validation script
├── data/strict_qc/
│   └── [16 fragment CSV files]    # 852-914 sequences (fragment-dependent)
├── configs/
│   └── config_strict_qc.yaml      # Experimental training config
└── docs/
    ├── BOUGHTER_ADDITIONAL_QC_PLAN.md
    ├── TRAINING_READINESS_CHECK.md
    ├── AUDIT_CORRECTIONS_APPLIED.md
    ├── STALE_REFERENCES_FIX_SUMMARY.md
    ├── TRAIN_DATASETS_ORGANIZATION_PLAN.md
    └── PLAN_AUDIT_SUMMARY.md
```

---

## Future Experiments

If you're considering new experiments, please:

1. **Create a dated subdirectory** (e.g., `experiment_name_YYYY-MM-DD/`)
2. **Write an EXPERIMENT_README.md** documenting:
   - Hypothesis and rationale
   - Methodology and implementation
   - Expected outcomes
   - Actual results
   - Conclusion and decision
3. **Preserve all artifacts** (data, configs, scripts, logs)
4. **Archive when complete** (whether successful or not)
5. **Update this README** with a summary

---

## Contact

Questions about archived experiments? Check the individual EXPERIMENT_README.md files for detailed context.

**Last Updated:** 2025-11-06
**Maintainer:** Clarity Digital Twin Project
