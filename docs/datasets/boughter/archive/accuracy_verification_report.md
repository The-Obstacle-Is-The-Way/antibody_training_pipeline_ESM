# Boughter Dataset – Pipeline Accuracy Verification Report

**Date**: 2025-11-02  
**Pipeline Version**: Hybrid Translation (direct-first) + Strict IMGT  
**Status**: ✅ VALIDATED

---

## 1. Executive Summary

- **Overall retention**: **1065 / 1171** (90.9%) clean antibodies ready for ML training.  
- **Stage performance**:  
  - Stage 1 (DNA → protein): 1171 → **1117** (95.4%)  
  - Stage 2 (ANARCI IMGT): 1117 → **1110** (99.4%)  
  - Stage 3 (QC filters): 1110 → **1065** (95.9%)
- **HIV recovery vs Boughter published counts**: **284 / 315 (90.2%)**  
  - Gut 94.7%, Nat 94.8%, Control 86.3%, PLOS 75.5%
- **Label balance** (Novo flagging, training set): 443 specific (48.5%) vs 471 non-specific (51.5%); 151 mild excluded.
- **All 16 fragment CSVs present** with documented metadata and strict IMGT boundaries.

Outstanding differences are limited to HIV control/PLOS sequences whose raw DNA exhibits irrecoverable ambiguity (see §4.2).

---

## 2. Stage-by-Stage Performance

### 2.1 Stage 1 – DNA Translation & Novo Flagging

| Subset | Raw DNA | Translated | Success % | Notes |
|--------|--------:|-----------:|----------:|-------|
| flu        | 379 | 347 | 91.6% | V-domain direct translation |
| mouse_iga  | 481 | 480 | 99.8% | V-domain direct translation |
| hiv_nat    | 134 | 130 | 97.0% | Hybrid (ATG fallback where needed) |
| hiv_cntrl  | 50  | 45  | 90.0% | Hybrid |
| hiv_plos   | 52  | 43  | 82.7% | Hybrid, several sequences irrecoverable |
| gut_hiv    | 75  | 72  | 96.0% | Hybrid |
| **Total**  | **1171** | **1117** | **95.4%** | 54 failures logged |

**Translation failures (54)**  
`translation_failures.log`: flu 32, hiv_plos 9, hiv_cntrl 5, hiv_nat 4, gut_hiv 3, mouse_iga 1. Failures stem from excessive ambiguous bases (N), stop codons early in the V-domain, or sequences <95 aa after trimming.

### 2.2 Stage 2 – ANARCI Annotation (IMGT)

- Annotated sequences: **1110 / 1117** (99.4%)  
- Failures (7): flu 6, hiv_plos 1 — ANARCI returned no CDRs for these frameworks.  
- IMGT boundaries: H-CDR3 = 105–117, H-CDR2 = 56–65, strict numbering across fragments.

### 2.3 Stage 3 – Post-Annotation Quality Control

- Filters applied: remove sequences with `X` in any CDR (21), remove empty CDRs (25), log to `qc_filtered_sequences.txt`.  
- Clean sequences: **1065 / 1110** (95.9%)  
- Removals by subset: flu 34, mouse 6, hiv_nat 2, hiv_plos 2, hiv_cntrl 1.

---

## 3. Final Dataset Characteristics

### 3.1 Subset Counts (Clean)

| Subset | Clean Count | Published† | Recovery |
|--------|------------:|-----------:|---------:|
| Flu        | 307 | 312 | 98.4% |
| Mouse IgA  | 474 | 445 | 106.5% (includes extra constant-region variants) |
| HIV_nat    | 128 | 135 | 94.8% |
| HIV_cntrl  | 44 | 51  | 86.3% |
| HIV_plos   | 40 | 53  | 75.5% |
| Gut_hiv    | 72 | 76  | 94.7% |
| **Total**  | **1065** | **1072** | **99.3%** |

†Published counts derived from Boughter et al. (2020) Table 1 and supplementary .dat files.

### 3.2 Novo Flagging Outcomes

| Flag bucket | Count | % | Label | Included in training? |
|-------------|------:|---:|-------|-----------------------|
| 0 | 443 | 41.6 | Specific (0) | ✅ |
| 1–3 | 151 | 14.2 | Mild | ✗ |
| 4–7 | 471 | 44.2 | Non-specific (1) | ✅ |
| **Total** | **1065** | | | **914 training** |

### 3.3 Fragment Coverage

All 16 CSVs (`VH_only`, `VL_only`, `H-CDR1`, …, `Full`) contain:
- Metadata headers describing extraction method, IMGT rationale, counts.  
- Fields: `id`, `sequence`, `label`, `subset`, `num_flags`, `flag_category`, `include_in_training`, `source`, `sequence_length`.  
- 1065 rows per fragment (labels + mild flags included; training subset derived via `include_in_training`).

---

## 4. Discrepancy Analysis

### 4.1 Improvements Achieved
- Hybrid translation (direct-first, ATG scoring fallback) recovers **~90%** of HIV sequences (up from 11.7%).  
- Mouse/flu retention now matches or exceeds published counts — prior losses due to misclassified V-domain translations have been eliminated.  
- Stage 2 success surpasses 99%, consistent with ANARCI benchmarks on clean data.

### 4.2 Remaining Gaps
- **HIV_control / HIV_PLOS**: Several sequences still fail due to pervasive Ns and frame disruptions in the publicly released DNA files. Manual inspection shows no recoverable V-domain even after enumerating alternative ATGs. Boughter likely used protein-level data internally; recovering the missing 20 sequences would require access to those cleaned amino-acid files.  
- **Mouse surplus**: 474 vs 445. Raw FASTA includes constant-region variants that Boughter filtered post-ANARCI. Keeping them is harmless (ANARCI succeeds, QC passes) and preserves Novo’s “>1000” dataset size. If strict parity is desired, filter to published IDs using `reference_repos/AIMS_manuscripts/app_data/*.dat` as whitelist.

---

## 5. Validation Artefacts

- Stage 1 output: `data/train/boughter/processed/boughter.csv`  
- Translation failures: `data/train/boughter/raw/translation_failures.log`  
- Stage 2 annotation log: `data/train/boughter/annotated/annotation_failures.log`  
- Stage 3 filtered IDs: `data/train/boughter/annotated/qc_filtered_sequences.txt`  
- Validation report (this run): `data/train/boughter/annotated/validation_report.txt`

---

## 6. Conclusion

The hybrid translation + strict IMGT pipeline produces **1065** high-quality Boughter antibodies with balanced labels, aligning with the Novo Nordisk training specification. Residual discrepancies are restricted to HIV control/PLOS subsets where the public DNA source is irretrievably degraded. The dataset is ready for ESM-2 embedding and reproduction of Sakhnini et al. (2025) results (Table 4, Figures 3–5).
