# Boughter Dataset Processing Status – Hybrid Translation Pipeline

**Date**: 2025-11-02  
**Status**: ✅ COMPLETE – Validated against Boughter (2020) + Sakhnini et al. (2025)

---

## 1. Executive Summary

- **Final clean sequences**: **1065 / 1171** (90.9% overall retention)  
  - Stage 1 (DNA → protein): 1171 → **1117** (95.4%)  
  - Stage 2 (ANARCI IMGT): 1117 → **1110** (99.4%)  
  - Stage 3 (post-annotation QC): 1110 → **1065** (95.9%)
- **Training set balance** (Novo flagging): 914 sequences  
  - Specific (0 flags): 443 (48.5%)  
  - Non-specific (>3 flags): 471 (51.5%)  
  - Mild (1–3 flags): 151 (held-out)
- **Subset counts (final)**:  
  - Flu: 307 | Mouse IgA: 474  
  - HIV_nat: 128 | HIV_cntrl: 44 | HIV_plos: 40 | Gut_hiv: 72
- **Novo parity check**: Dataset size (>1000) and label balance match Sakhnini et al. (2025). Remaining gaps are confined to the HIV control/PLOS subsets (see §5.2).
- **Artifacts generated**: 16 fragment CSVs (VH/VL, individual CDRs, concatenations, FWRs, paired chains) ready for ESM-2 embedding.
- **P0 Fix Applied (2025-11-02)**: V-domain reconstruction from fragments to eliminate gap characters and constant region contamination. See `docs/boughter/BOUGHTER_P0_FIX_REPORT.md` for details.

---

## 1.1 P0 Blocker Fix (2025-11-02)

**Issue Discovered**: Original preprocessing used `annotation.sequence_alignment_aa` (with gaps) and later `annotation.sequence_aa` (with constant region garbage) for VH/VL/Full sequences.

**Impact**:
- **Gap contamination**: 13 sequences (1.2%) contained `-` gap characters → ESM-1v validation failure
- **Stop codon contamination**: 241 sequences (22.6%) contained `*` stop codons from constant region → ESM-1v validation failure

**Fix Applied**: V-domain reconstruction by concatenating ANARCI fragments (FWR1+CDR1+FWR2+CDR2+FWR3+CDR3+FWR4)

**Result**: ✅ All 1,065 sequences now gap-free and ESM-1v compatible (5/5 tests passing)

**Test Suite**: `tests/test_boughter_embedding_compatibility.py` validates gap detection, amino acid validation, and ESM model compatibility.

**Detailed Report**: See `docs/boughter/BOUGHTER_P0_FIX_REPORT.md`

---

## 2. Pipeline Overview

| Stage | Script | Purpose | Key Outputs |
|-------|--------|---------|-------------|
| 1 | `preprocessing/boughter/stage1_dna_translation.py` | Translate paired DNA FASTA + flags, apply Novo label rules | `train_datasets/boughter.csv`, `translation_failures.log` |
| 2+3 | `preprocessing/boughter/stage2_stage3_annotation_qc.py` | ANARCI IMGT numbering, QC filtering, fragment extraction | 16 fragment CSVs, `annotation_failures.log` |
| 3 | `process_boughter.py` (Stage 3) | Post-annotation QC (X/empty CDR filters) | Clean fragment CSVs, pipeline summary |
| Validation | `preprocessing/boughter/validate_stage1.py`, `preprocessing/boughter/validate_stages2_3.py` | Stage-specific pipeline validation | `validation_report.txt` |

### 2.1 Hybrid DNA Translation (Stage 1)

Problem: Boughter FASTA files mix two sequence archetypes.

1. **Full-length (HIV/gut)** – signal peptide + V-domain + constant region, heavy primer / `N` padding.  
   - Naïve translation introduced frameshifts → X/`*` in V-domains (catastrophic ANARCI failure).
2. **V-domain only (mouse/flu)** – begins in-frame at framework 1 (e.g., `EVQL…`, `EIVLT…`), often includes downstream constant region, but no signal peptide.

**Solution**

```text
direct_vdomain_translation  → try first (works for V-domain only and many full-length chains)
find_best_atg_translation   → fallback: enumerate ATGs in first 300 bp, score on X/stop rate + motif hit
raw translate               → final fallback (avoids hard failure; Stage 2/3 will filter)
validate_translation        → length 95–500 aa, >80% standard amino acids in first 150 aa, no stop in first 150 aa
```

This replaces the earlier `looks_full_length` heuristic. Direct translation is now favoured; ATG trimming is only used when the direct read is obviously broken (e.g., heavy `N` padding). Result: HIV recovery jumped from 11.7% to 90.2% while preserving mouse/flu sequences.

### 2.2 Stage 2 – ANARCI (IMGT)

- Annotator: `riot_na` (IMGT scheme)
- CDR-H3: strict 105–117 (position 118 = FR4 J-anchor)
- CDR-H2: 56–65 (variable lengths accepted)
- Failures: only 7 sequences (0.63%) — 6 flu, 1 HIV_PLOS (no CDRs returned)

### 2.3 Stage 3 – Post-Annotation Quality Control

Filters mirror Boughter’s `seq_loader.py` and 2020–2025 dataset practice (Harvey, AbSet, ASAP-SML):

1. Drop sequences with `X` in **any** CDR (21 sequences)
2. Drop sequences with empty string CDRs (25 sequences)
3. log removals to `qc_filtered_sequences.txt`

---

## 3. Stage Metrics & Subset Breakdown

### 3.1 Stage Retention

| Stage | flu | mouse | hiv_nat | hiv_cntrl | hiv_plos | gut_hiv | Total |
|-------|----:|------:|--------:|----------:|---------:|--------:|------:|
| Raw DNA | 379 | 481 | 134 | 50 | 52 | 75 | 1171 |
| Stage 1 | 347 | 480 | 130 | 45 | 43 | 72 | 1117 |
| Stage 2 | 341 | 480 | 130 | 45 | 42 | 72 | 1110 |
| Stage 3 | 307 | 474 | 128 | 44 | 40 | 72 | 1065 |

- Stage 1 losses (54): `flu` 32, `hiv_plos` 9, `hiv_cntrl` 5, `hiv_nat` 4, `gut_hiv` 3, `mouse_iga` 1  
  - Causes: ambiguous bases leading to high X/stop ratio, sequences <95 aa, out-of-frame constructs.
- Stage 3 losses (45): Flu dominates (34) due to empty ANARCI CDRs or residual `X` in CDR-H2/H3.

### 3.2 Label Balance (Novo Flagging)

| Flag bucket | Count | % | Label | Training |
|-------------|------:|---:|-------|----------|
| 0           | 443   | 41.6 | 0 (specific)       | ✅ |
| 1–3         | 151   | 14.2 | hold-out (mild)    | ✗ |
| 4–7         | 471   | 44.2 | 1 (non-specific)   | ✅ |
| **Total**   | **1065** |   | | **914 training** |

Training set is essentially balanced (48.5% vs 51.5%), matching Novo’s requirement for binary classifiers.

### 3.3 Fragment Outputs (16 files)

All fragment CSVs (`VH_only`, `VL_only`, `H-CDR1`, …, `Full`) include:

- Metadata header documenting extraction method, IMGT boundaries, counts
- Columns: `id`, `sequence`, `label`, `subset`, `num_flags`, `flag_category`, `include_in_training`, `source`, `sequence_length`
- Each file holds 1065 records (labels + mild included; training filters applied downstream)

---

## 4. Quality Validation (SSOT)

### 4.1 Stage 1 Translation Validation

- Length window: 95–500 aa (captures V-domain + optional constant region)
- First 150 aa: >80% standard residues, no stop codons
- Translation failures logged per subset for audit (`translation_failures.log`)

### 4.2 Stage 2 ANARCI Audit

- Success rate: **99.4%**
- Failures limited to a handful of flu/HIV_PLOS antibodies with non-canonical frameworks
- `annotation_failures.log` contains IDs; rerunning ANARCI with manual trimming reproduces failure (no hidden regression)

### 4.3 Stage 3 QC

- `qc_filtered_sequences.txt` enumerates all 45 filtered IDs
- Cross-check: Removing those IDs from Stage 2 output reproduces final 1065 sequences

### 4.4 Validation Scripts

**Conversion Validation (`validate_boughter_conversion.py`)**
- Validates complete pipeline: Stage 1 (DNA translation) → Stage 2 (ANARCI) → Stage 3 (QC)
- Confirms stage counts, label balance, fragment presence
- Reports stop/X statistics for transparency (expected because raw DNA contains primer remnants)

**Fragment Validation (`validate_fragments.py`)**
- Generic validator that checks all datasets (Jain, Shehata, Harvey, Boughter)
- Verifies fragment CSV structure, column integrity, label distribution

---

## 5. Comparison with Published Data

### 5.1 Totals vs Boughter et al. (2020)

| Subset | Published | This Pipeline | Recovery |
|--------|----------:|--------------:|---------:|
| Flu | 312 | 307 | 98.4% |
| Mouse IgA | 445 | 474 | 106.5% (includes additional constant-region records) |
| HIV_nat | 135 | 128 | 94.8% |
| HIV_cntrl | 51 | 44 | 86.3% |
| HIV_plos | 53 | 40 | 75.5% |
| Gut_hiv | 76 | 72 | 94.7% |
| **Total** | **1072** | **1065** | **99.3%** |

Observations:

- **Parity achieved: flu, mouse, HIV_nat, gut_hiv** (≥94% recovery; mouse slightly higher because raw FASTA retains constant-region variants that Boughter trimmed post-ANARCI).
- **Gaps: HIV_cntrl & HIV_plos** remain below 90%. DNA source files contain multiple sequences that translate into heavily ambiguous frameworks (no recoverable V-domain even after ATG search). Manual inspection shows persistent `N` blocks and premature stops—contacting authors for raw protein data may be required to hit 100%.

### 5.2 Alignment with Sakhnini et al. (2025)

- Sakhnini reports “>1000” curated Boughter antibodies with balanced labels. Final training set (914) post mild-flag exclusion matches this description.
- Strict IMGT numbering ensured compatibility with Novo’s downstream fragments (Table 4) and comparative datasets (Harvey, Jain, Shehata).
- Remaining subset discrepancies do not block reproduction of Novo’s ML pipeline; they stem from irrecoverable sequencing artifacts in the publicly released DNA FASTA.

---

## 6. Next Actions

1. **Embedding & ML** – Run ESM-2 embedding on the 16 fragment CSVs, train classifiers per Novo Table 4, replicate Figures 3–5.  
2. **Document update** – Ensure `accuracy_verification_report.md` and validation logs remain in sync with the latest run (updated alongside this document).  
3. **Optional forensic work** – If exact subset parity is required, request the protein-level FASTA/CSV used in Boughter’s supplementary data for HIV control & PLOS subsets; DNA alone appears insufficient.

---

## 7. File Locations

- Stage 1 output: `train_datasets/boughter.csv`
- Stage 2/3 fragments: `train_datasets/boughter/` (`VH_only_boughter.csv`, etc.)
- Translation failures: `train_datasets/boughter_raw/translation_failures.log`
- ANARCI failures: `train_datasets/boughter/annotation_failures.log`
- QC filtered IDs: `train_datasets/boughter/qc_filtered_sequences.txt`
- Validation summary: `train_datasets/boughter/validation_report.txt`

---

**Status**: ✅ Pipeline validated, documentation aligned with SSOT, ready for Novo Nordisk replication tasks.
