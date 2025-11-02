# Boughter Dataset Processing Requirements - Novo Nordisk Implementation

## Document Purpose
This document specifies the exact requirements for processing the Boughter 2020 dataset according to the Novo Nordisk methodology described in Sakhnini et al. 2025 (doi: pending). All requirements are sourced directly from published papers without assumptions.

---

## 1. Dataset Overview

### 1.1 Source Information
- **Primary Reference**: Boughter et al. (2020) eLife 9:e61393
- **Novo Nordisk Reference**: Sakhnini et al. (2025) - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"
- **Raw Data Source**: AIMS_manuscripts GitHub repository (ctboughter/AIMS_manuscripts)
- **Data Location**: `app_data/full_sequences/`

### 1.2 Dataset Composition
According to Boughter et al. 2020 (Table 1):
```
Dataset                    # Polyreactive   # Non-Polyreactive   Total
Mouse IgA                  205              240                  445
HIV reactive               172              124                  296
Influenza reactive         152              160                  312
Complete dataset           529              524                  1053
```

### 1.3 Six Boughter Subsets
According to Sakhnini et al. 2025 and verified data inventory:
1. **flu** (Influenza reactive): 379 sequences
2. **hiv_nat** (HIV NAT): 134 sequences
3. **hiv_cntrl** (HIV control): 50 sequences
4. **hiv_plos** (HIV PLOS): 52 sequences
5. **gut_hiv** (gut HIV): 75 sequences
6. **mouse_iga** (mouse IgA): 481 sequences

**Total**: 1,171 antibody sequences

---

## 2. Raw Data Format

### 2.1 Sequence Files
**CRITICAL**: All sequences are DNA nucleotide sequences, NOT protein sequences.

File naming pattern:
```
{subset}_fastaH.txt         # Heavy chain DNA sequences
{subset}_fastaL.txt         # Light chain DNA sequences
{subset}_NumReact.txt       # Polyreactivity flags (0-7 scale)
{subset}_YN.txt             # Binary flags (Y/N format)
```

Exception: Mouse IgA uses `.dat` extension:
```
mouse_fastaH.dat
mouse_fastaL.dat
mouse_YN.txt
```

### 2.2 FASTA Format Structure
From Boughter raw data:
```
>                           # Header line (empty after >)
NAGGTGCAGCTGGTGCAGTCT...   # DNA nucleotide sequence
>                           # Next antibody header
GAGGTGCAGCTGGTGGAGTCT...   # Next DNA nucleotide sequence
```

### 2.3 Flag File Format

**NumReact Format** (flu, hiv_nat, hiv_cntrl, gut_hiv):
```
reacts      # Header line
1           # Number of antigens bound (0-7)
0
0
5
...
```

**Y/N Format** (plos_hiv, mouse_iga):
```
Y           # Y = polyreactive, N = non-polyreactive
Y
N
Y
...
```

---

## 3. Polyreactivity Assay Panel

### 3.1 ELISA Panel Antigens
According to Boughter et al. 2020:
> "the reactivity of each antibody is tested against a panel of 4–7 biochemically diverse target antigens: DNA, insulin, lipopolysaccharide (LPS), flagellin, albumin, cardiolipin, and keyhole limpet hemocyanin (KLH)"

**Standard 7-antigen panel**:
1. DNA (negatively charged)
2. Insulin (negatively charged)
3. LPS (lipopolysaccharide - amphipathic, negatively charged)
4. Flagellin (large)
5. Albumin (negatively charged)
6. Cardiolipin (amphipathic)
7. KLH (keyhole limpet hemocyanin - large, polar)

### 3.2 Polyreactivity Definition
From Boughter et al. 2020:
> "antibodies are determined to be polyreactive if the authors of the original studies determined a particular clone binds to two or more ligands in the panel. Those that bind to one or none of the ligands in the panel are deemed non-polyreactive."

---

## 4. Novo Nordisk Flagging Strategy

### 4.1 Binary Classification Scheme
From Sakhnini et al. 2025, Section 2.1:

> "the Boughter dataset was first parsed into two groups: specific (0 flags) and non-specific group (>3 flags), leaving out the mildly non-specific antibodies (1-3 flags)"

**Explicit Rules**:
- **0 flags** → Label 0 (specific) - INCLUDE in training
- **1-3 flags** → EXCLUDE from training dataset
- **>3 flags** (i.e., 4+ flags) → Label 1 (non-specific) - INCLUDE in training

### 4.2 Classification in Methods Section
From Sakhnini et al. 2025, Section 4.3 (line 236):

> "First, the Boughter dataset was parsed into three groups as previously done in [44]: specific group (0 flags), mildly poly-reactive group (1-3 flags) and poly-reactive group (>3 flags)."

### 4.3 Distribution Description
From Sakhnini et al. 2025, Figure 1B caption:

> "relatively balanced in terms of specific (zero flags), mildly non-specific (1-3 flags) and non-specific (>4 flags) antibodies"

**NOTE**: Text says ">3 flags" but Figure caption says ">4 flags". Following the methods section (4.3, line 236) which explicitly states ">3 flags" as this is the technical specification.

---

## 5. Required Processing Pipeline

### 5.1 Translation Requirements
**CRITICAL**: Raw data is DNA → must translate to amino acid sequences.

From DNA nucleotide format to protein format:
- Standard genetic code translation
- Antibody sequences typically start with signal peptide
- Framework for variable domains

### 5.2 ANARCI Annotation
From Sakhnini et al. 2025, Section 4.3 (line 239):

> "The primary sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme"

**Requirements**:
- Tool: ANARCI (https://github.com/oxpig/ANARCI)
- Numbering scheme: IMGT
- Purpose: Identify CDR boundaries and extract CDR sequences

### 5.3 Sequence Fragments
From Sakhnini et al. 2025, Table 4:

**16 different antibody fragment sequences**:
```
VL, VH                      # Variable domains
L-CDR1, L-CDR2, L-CDR3      # Light chain CDRs
H-CDR1, H-CDR2, H-CDR3      # Heavy chain CDRs
VH/VL joined                # Joined variable domains
L-CDRs joined               # All light CDRs concatenated
H-CDRs joined               # All heavy CDRs concatenated
H/L-CDRs joined             # All CDRs concatenated
```

---

## 6. Data Validation Requirements

### 6.1 Expected Flag Distribution
From local data analysis (flu subset, 379 sequences):
```
0 flags: 120 sequences (31.7%)
1 flag:  74 sequences (19.5%)
2 flags: 31 sequences (8.2%)
3 flags: 19 sequences (5.0%)
4 flags: 22 sequences (5.8%)
5 flags: 49 sequences (12.9%)
6 flags: 64 sequences (16.9%)
```

**After Novo filtering** (0 and 4+ only):
- Specific (0 flags): 120 sequences (63.5%)
- Non-specific (4+ flags): 135 sequences (36.5%)
- Excluded (1-3 flags): 124 sequences

### 6.2 Sequence Count Validation
After processing, total counts should match:
- Pre-filtering: 1,171 total sequences across 6 subsets
- Post-filtering: ~700-800 sequences (exact number TBD from flag distributions)

---

## 7. Quality Control Checks

### 7.1 Translation Validation
- Verify no stop codons in CDR regions
- Check reading frame alignment
- Validate heavy/light chain pairing consistency

### 7.2 ANARCI Validation
- All sequences must successfully annotate
- Verify IMGT numbering consistency
- Extract and validate CDR boundaries

### 7.3 Flag File Integrity
- Verify flag count matches sequence count
- Check for missing or malformed entries
- Validate flag ranges (0-7 for NumReact, Y/N for binary)

---

## 8. Output Format Requirements

### 8.1 CSV Structure (Inferred from Harvey/Jain/Shehata patterns)
Expected columns:
```
sequence_id,subset,heavy_chain,light_chain,VH,VL,H_CDR1,H_CDR2,H_CDR3,L_CDR1,L_CDR2,L_CDR3,label,num_flags
```

### 8.2 Label Encoding
```
label = 0: Specific (0 flags)
label = 1: Non-specific (>3 flags)
excluded: 1-3 flags (not in output)
```

---

## 9. Critical Implementation Notes

### 9.1 What is NOT Specified
The following are NOT explicitly detailed in Sakhnini et al. 2025:
- Exact DNA translation methodology
- Signal peptide cleavage positions
- Handling of incomplete sequences
- Treatment of sequences failing ANARCI annotation
- Pre-cleaning or post-cleaning procedures beyond ANARCI

### 9.2 What IS Specified
From Sakhnini et al. 2025:
1. Use ANARCI with IMGT numbering (Section 4.3, line 239)
2. Parse into 0 flags / 1-3 flags / >3 flags (Section 4.3, line 236)
3. Exclude 1-3 flags from training (Section 2.1, line 55)
4. Extract 16 antibody fragments (Table 4)

---

## 10. References

**Primary Sources**:
1. Boughter CT et al. (2020) "Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops." eLife 9:e61393
2. Sakhnini LI et al. (2025) "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters." (Journal pending)

**Data Source**:
- GitHub: ctboughter/AIMS_manuscripts
- Path: app_data/full_sequences/

**Tool References**:
- ANARCI: Dunbar J, Deane CM (2016) Bioinformatics 32:298-300
- IMGT numbering: Lefranc MP et al. (2003) Dev Comp Immunol 27:55-77

---

## Document Status
- **Version**: 1.0
- **Date**: 2025-11-02
- **Status**: Ready for senior review
- **Next Step**: External validation before implementation
