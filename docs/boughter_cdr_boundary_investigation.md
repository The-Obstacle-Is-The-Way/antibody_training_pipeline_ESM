# Boughter CDR Boundary Investigation

## Executive Summary

**Finding**: Boughter et al. (eLife 2020) dataset uses IgBLAST alignment-based CDR extraction, NOT strict IMGT/ANARCI numbering, despite Sakhnini et al. (2025) stating "IMGT numbering scheme."

**Key Discrepancies**:
- **CDR-H3**: Extends to position 118 (includes FR4 J-anchor W), not strict IMGT 105–117
- **CDR-H2**: Variable lengths suggest IgBLAST alignment boundaries, not consistent IMGT 56–65

**Impact**: Developers using ANARCI with strict IMGT will get mismatches when validating against Boughter's published CDR sequences.

---

## Background

### Discord Report (2025-11-01)

User Hybri reported CDR boundary inconsistencies while preprocessing Boughter dataset:

```
Novo said they used ImGt numbering scheme
CDR3 boundaries are 105 to 117
but I their CDRs data (that I am using to validate my processing).
their CDR3 seems to extend to position 118
including a W at the end of each sequences. the thing is:
1- that's not ImGt boundary,
2- the W is consensus part of Fr4 not CDR3 ...
I don't know why nor how they did that! anarci wouldn't include the W.
```

### Context

- **Sakhnini et al. 2025** (Novo Nordisk/Cambridge): "sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme" ([source](line 240 of Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md))
- **Standard IMGT boundaries** (heavy chain):
  - CDR-H1: 27–38
  - CDR-H2: 56–65
  - CDR-H3: 105–117
  - FR4: 118–128 (starts with J-PHE or J-TRP, often W)

---

## Investigation

### Evidence 1: Boughter's mouse_IgA.dat CDR3s

**File**: `reference_repos/AIMS_manuscripts/app_data/mouse_IgA.dat`

**Sample CDR-H3 sequences** (column 6):
```
ARRGYYYGSFDYW
AIDYDQAMDYW
ARFYGNYEDYYAMDYW
ARLDSSGYDYAMDYW
ARQEGNSGEYYFDYW
ARVYYYGSSNAMDYW
ARHATSYAMDYW
ARGGDYDDDAMDYW
```

**Observation**: ALL 450 CDR3s end with W (tryptophan), with 335/450 (74%) showing YW pattern and 104/450 (23%) showing VW pattern — W is the J-region anchor residue at IMGT position 118.

**Reference**: Lines 1-20 of `reference_repos/AIMS_manuscripts/app_data/mouse_IgA.dat`

---

### Evidence 2: Boughter's CDR Extraction Code

**File**: `reference_repos/AIMS_manuscripts/app_data/adimab_data/GetCDRs_AA.ipynb`

**Method**: Custom IgBLAST alignment parsing, NOT ANARCI

#### CDR3 Heavy Chain Extraction (cell 6):

```python
# Find conserved Cysteine in germline CDR3
where=aa[0][0].find('C')
cdr3H_single=aa[0][1][where+1:]  # Start AFTER the C

# Align to J-gene reference
for j in np.arange(len(heavyJ)):
    test1=cdr3H_final[i][0]
    test2=heavyJ[j][0]
    aa=pairwise2.align.localxs(test1,test2,-1,-1)
    # Find best J-gene match

align_final=pairwise2.align.localxs(test1,heavyJ[Jnum],-1,-1)
where=align_final[0][3]  # J-gene match position
short_3H=aa[0][0][0:where]  # STOP at J-match (INCLUDES FR4 anchor)
```

**Key Logic**:
1. Starts CDR3 after conserved C (IMGT 104)
2. Aligns to J-gene germline sequences
3. Stops at J-gene match boundary — which INCLUDES the conserved W at IMGT 118

**Reference**: Lines 60-82 of `GetCDRs_AA.ipynb`, cells 5-6

---

### Evidence 3: AIMS Tool Does NOT Extract CDRs

**File**: `reference_repos/AIMS/aims_immune/aims_loader.py`

**Function**: `Ig_loader()` (lines 132-274)

```python
def Ig_loader(fastapath,label,loops=6,drop_degens = False,return_index = False,dataLoc = ''):
    # ...
    total_Abs=pandas.read_csv(fastapath,sep=',',header=0,
        names=['cdrL1_aa','cdrL2_aa','cdrL3_aa','cdrH1_aa','cdrH2_aa','cdrH3_aa'])
    # ... (filtering and cleanup only)
```

**Finding**: AIMS loads pre-extracted CDRs from CSV files. It does NOT perform extraction.

**Implication**: CDR boundaries are determined upstream by IgBLAST + GetCDRs_AA.ipynb, not by AIMS itself.

**Reference**: Lines 132-274 of `reference_repos/AIMS/aims_immune/aims_loader.py`

---

### Evidence 4: seq_loader.py Confirms Pre-Processed Data

**File**: `reference_repos/AIMS_manuscripts/seq_loader.py`

**Function**: `getBunker()` (loads mouse IgA data)

```python
def getBunker():
    total_Abs=pandas.read_csv('app_data/mouse_IgA.dat',sep='\s+',header=None,
        names=['cdrL1_aa','cdrL2_aa','cdrL3_aa','cdrH1_aa','cdrH2_aa','cdrH3_aa','react'])
    # ... (filtering only, no extraction)
```

**Finding**: Loads pre-processed CDRs from `mouse_IgA.dat` — the file with W-terminated CDR3s.

**Reference**: Lines 7-8 of `reference_repos/AIMS_manuscripts/seq_loader.py`

---

## Root Cause Analysis

### Why the Discrepancy Exists

1. **IgBLAST alignment boundaries** are flexible and based on germline V/J alignments, not fixed IMGT coordinates
2. **J-gene matching** includes the conserved J-anchor (W/F at position 118) as part of the "functional CDR3"
3. **Boughter's custom script** preserves this boundary choice when extracting CDRs
4. **ANARCI with strict IMGT** uses fixed coordinates: CDR3 = 105–117, FR4 starts at 118

### What "IMGT Numbering" Means in Sakhnini Paper

- **Full-sequence numbering**: Likely used IMGT/ANARCI to number all positions in the sequence
- **CDR extraction**: Used Boughter's pre-processed CDRs (IgBLAST-based boundaries)
- **Terminology confusion**: "IMGT numbering scheme" ≠ "IMGT CDR boundary definitions"

---

## Implications

### For Our Pipeline

**Problem**: If we use ANARCI with strict IMGT boundaries:
```python
# ANARCI output (strict IMGT 105-117)
cdr3 = "ARRGYYYGSFD"  # No W

# Boughter reference (105-118)
cdr3 = "ARRGYYYGSFDYW"  # Has W
```

**Mismatch**: Off-by-one at C-terminus, breaks validation.

### For Hybri's Work

**CDR3 Issue**: Extend to position 118 to match Boughter's data
**CDR2 Issue**: IgBLAST alignment boundaries vary per sequence — cannot be fixed with a single rule

---

## Solutions

### Option 1: Compatibility Mode (Recommended)

Add extraction flags to match Boughter's boundaries:

```python
# In our CDR extractor
def extract_cdrs(sequence, cdr3_include_118=False, cdr2_scheme='imgt'):
    """
    Args:
        cdr3_include_118: If True, extend CDR-H3 to position 118 (Boughter compat)
        cdr2_scheme: 'imgt' (56-65) or 'chothia' (52-56) or 'igblast' (variable)
    """
```

**Advantages**:
- Supports both strict IMGT and Boughter compatibility
- Explicit about which boundaries are used
- Logs scheme choice for reproducibility

### Option 2: Request Clarification (Parallel Track)

**Email Novo Nordisk authors**:
> "Sakhnini et al. states 'ANARCI following IMGT numbering scheme,' but Boughter's mouse_IgA.dat CDR3s extend to position 118 (include J-anchor W). Could you clarify:
> 1. Did you trim position 118 from Boughter's CDR3s before training?
> 2. Or did you use the extended 105-118 boundaries as-is?
> 3. For CDR2, are boundaries IMGT 56-65 or IgBLAST-reported?"

**Timeline**: Wait for Novo's code release (stated "2 weeks" in Discord)

### Option 3: Reverse-Engineer from Published Data

**If Novo releases**:
- Training data with CDR sequences
- Model weights

**Action**: Compare CDR lengths in their training set vs Boughter's published data to infer exact boundaries used.

---

## Recommendations

### Immediate Actions

1. **Document this investigation** ✅ (this file)
2. **Verify findings with independent agent** (next step)
3. **Post concise Discord reply** (after verification)
4. **Add compat flags to our CDR extractor** (separate task)

### Discord Message (Draft for Review)

```
quick update on boundaries: traced boughter's extraction code — it's NOT anarci/strict imgt.

they used igblast alignments + custom parsing (GetCDRs_AA.ipynb in AIMS_manuscripts repo).
CDR3 starts after conserved C, ends at J-gene match — includes position 118 (the W).

novo saying "imgt numbering" likely means full-sequence numbering with imgt,
but boundaries deviate:
• strict imgt CDR-H3: 105–117 (no W)
• boughter/novo: 105–118 (includes J-anchor W)

for CDR2, igblast alignment boundaries vary per sequence
(not fixed imgt 56–65 or chothia 52–56).

solutions:
1. trim pos 118 from boughter CDR3s for strict imgt
2. add --cdr3-include-118 flag to match their boundaries
3. email novo for clarification (or wait for code release)

documented full investigation: [link to this file when pushed]
```

---

## References

### Primary Sources

1. **Boughter et al. 2020** (eLife): "An Automated Immune Molecule Separator"
   - DOI: https://elifesciences.org/articles/61393
   - Repository: https://github.com/ctboughter/AIMS_manuscripts

2. **Sakhnini et al. 2025** (Novo Nordisk/Cambridge): Antibody Non-Specificity PLM Biophysical study
   - Preprint: https://www.biorxiv.org/content/10.1101/2025.04.28.650927
   - HuggingFace: ZYMScott/polyreaction

3. **IMGT Numbering**: Lefranc et al. 2003
   - Standard: http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
   - CDR definitions: IMGT Scientific Chart

### Code References

4. **GetCDRs_AA.ipynb**: `reference_repos/AIMS_manuscripts/app_data/adimab_data/GetCDRs_AA.ipynb`
5. **mouse_IgA.dat**: `reference_repos/AIMS_manuscripts/app_data/mouse_IgA.dat`
6. **seq_loader.py**: `reference_repos/AIMS_manuscripts/seq_loader.py` (line 8)
7. **aims_loader.py**: `reference_repos/AIMS/aims_immune/aims_loader.py` (lines 132-274)

### Investigation Date

- **Conducted**: 2025-11-01
- **Investigator**: John H. Jung, MD
- **Context**: Responding to Hybri's Discord report on CDR boundary mismatches

---

## Appendix: IMGT Position 118 Context

**J-Region Anchor Residues** (IMGT 118):
- **IGHJ1, IGHJ4, IGHJ5**: Trp (W) — WGQG motif
- **IGHJ2**: Phe (F) — FDY motif
- **IGHJ3**: Trp (W) — WGQG motif
- **IGHJ6**: Trp (W) — YYYGMDV motif (less common)

**Prevalence**: ~80% of human antibodies use J4/J6 (both have W at 118)

**Why it matters**: Including position 118 in CDR3 adds a nearly-invariant W to most sequences, which:
- Increases apparent CDR3 length by 1
- Reduces sequence diversity (W is conserved)
- May improve J-gene usage predictions (if that's a modeling goal)

**Boughter's rationale** (inferred): Functional CDR3 includes the J-anchor for structural/binding context.

---

## Resolution (2025-11-02)

### Final Decision: Strict IMGT (CDR-H3 = 105-117, Excluding Position 118)

After first-principles analysis, we determined the correct approach:

**Why exclude position 118:**
1. **Biological**: Position 118 is FR4 (Framework Region 4), NOT CDR
   - It's the conserved J-anchor (W or F in all antibodies)
   - It's germline-encoded, not hypervariable
   - It does NOT contact antigen (β-strand, not loop)

2. **Machine Learning**: Position 118 provides ZERO predictive information
   - It's conserved (no variance to learn from)
   - Including it pollutes the variable region signal

3. **Standardization**: IMGT is the international standard
   - All other datasets (Harvey, Jain, Shehata) use strict IMGT
   - Cross-dataset comparability requires consistent boundaries

**Implementation:**
- **Use ANARCI with strict IMGT** boundaries (CDR3 = 105-117)
- **Document discrepancy**: Boughter's published files include position 118; we use strict IMGT
- **Optional flag**: `--cdr3-include-118` for compatibility testing only (not production)

**Rationale independence:**
- Our decision is based on first principles, NOT on replicating Novo
- If Novo used different boundaries, that would be a methodological limitation in their work

See also: `cdr_boundary_first_principles_audit.md` for full analysis

---

*Document version: 2.0*
*Last updated: 2025-11-02*
*Status: ✅ **RESOLVED - Use Strict IMGT***
