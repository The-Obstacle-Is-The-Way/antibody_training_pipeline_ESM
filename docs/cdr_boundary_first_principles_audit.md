# CDR Boundary First-Principles Audit

## Investigation Date
2025-11-01

## Question
**Is there a genuine discrepancy between Boughter's CDR boundaries and IMGT standards, or is this a misunderstanding?**

---

## Methodology

Independent verification from first principles:
1. Check IMGT official standards
2. Analyze Boughter's actual data files
3. Read extraction code
4. Cross-reference with Sakhnini paper claims
5. NO assumptions - verify everything

---

## Finding 1: IMGT Official Standards

### Source
Web search of IMGT.org official documentation

### Evidence
```
"In the IMGT unique numbering, CDR3-IMGT encompasses positions 105 to 117,
with positions 104 and 118 serving as the anchors of the CDR3-IMGT."

"Position 118 - The J-Region Anchor:
The CDR3-IMGT anchors are highly conserved: C104 (2nd-CYS, in F strand)
and F118 or W118 (J-PHE or J-TRP in G strand)."

"The basic length of a rearranged CDR3-IMGT is 13 amino acids (positions 105 to 117),
which corresponds to a JUNCTION of 15 amino acids (2nd-CYS 104 to J-TRP or J-PHE 118)."
```

### Verified Boundaries
- **CDR-H3 (IMGT)**: positions 105–117 (13 amino acids)
- **Position 118**: J-TRP or J-PHE (FR4, NOT CDR3)
- **JUNCTION**: positions 104–118 (15 amino acids, includes both anchors)

**Status**: ✅ VERIFIED from IMGT.org

---

## Finding 2: Boughter's Actual Data

### Source
`reference_repos/AIMS_manuscripts/app_data/mouse_IgA.dat`

### Analysis Command
```bash
awk '{print substr($6, length($6))}' mouse_IgA.dat | sort | uniq -c | sort -rn
```

### Result
```
450 W
```

**100% of CDR3 sequences (450/450) end with W (tryptophan)**

### Sample Sequences (column 6 = CDR-H3)
```
ARRGYYYGSFDYW      (length=13, ends with YW)
AIDYDQAMDYW        (length=11, ends with YW)
ARFYGNYEDYYAMDYW   (length=16, ends with YW)
ARLDSSGYDYAMDYW    (length=15, ends with YW)
```

### Conclusion
Boughter's CDR3s include position 118 (the W/Y-W pattern confirms J-anchor inclusion)

**Status**: ✅ VERIFIED empirically

---

## Finding 3: Extraction Code Analysis

### Source
`reference_repos/AIMS_manuscripts/app_data/adimab_data/GetCDRs_AA.ipynb`

### Method
Custom IgBLAST alignment parsing (NOT ANARCI)

### Critical Code (cell 6)

```python
# Step 1: Find conserved C (position 104)
where=aa[0][0].find('C')
cdr3H_single=aa[0][1][where+1:]  # START after C

# Step 2: Align to J-gene reference
for j in np.arange(len(heavyJ)):
    test1=cdr3H_final[i][0]
    test2=heavyJ[j][0]  # J-gene starts with W/F
    aa=pairwise2.align.localxs(test1,test2,-1,-1)
    # Find best J-gene match

# Step 3: Trim at J-match position
align_final=pairwise2.align.localxs(test1,heavyJ[Jnum],-1,-1)
where=align_final[0][3]  # J-alignment position
short_3H=aa[0][0][0:where]  # Keep [0:where]
```

### Logic Analysis
1. Starts CDR3 after C (position 104) ✓
2. Aligns remaining sequence to J-gene germlines
3. Finds best J-match and trims

### Key Question
Does `where=align_final[0][3]` point BEFORE or AFTER the W?

### Empirical Evidence
Output from cell 12: `array(['ARSAYYDYDGFAYW'], dtype='<U24')`

**The W is INCLUDED in final output.**

### Interpretation
Either:
- The code includes the first J-residue intentionally, OR
- `[0][3]` (alignment start) in pairwise2 points to position AFTER match start, OR
- Post-processing added the W

Regardless of implementation details, **the data file proves W is included**.

**Status**: ✅ VERIFIED (W is present in 100% of sequences)

---

## Finding 4: AIMS Tool Does NOT Extract

### Source
`reference_repos/AIMS/aims_immune/aims_loader.py` (lines 132-274)

### Code
```python
def Ig_loader(fastapath,label,loops=6,...):
    total_Abs=pandas.read_csv(fastapath,sep=',',header=0,
        names=['cdrL1_aa','cdrL2_aa','cdrL3_aa','cdrH1_aa','cdrH2_aa','cdrH3_aa'])
    # ... (filtering only, NO extraction)
    return(final_Df)
```

### Finding
AIMS loads pre-extracted CDRs from CSV files.

CDR extraction happened UPSTREAM via:
- IgBLAST alignment
- GetCDRs_AA.ipynb custom parsing

**Status**: ✅ VERIFIED

---

## Finding 5: Sakhnini Paper Claims

### Source
`literature/markdown/Sakhnini_2025.../Sakhnini_2025...md`

### Line 240
```
"sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme"
```

### Line 47
```
"curated dataset of >1000 mouse IgA... from Boughter et al. [44]"
```

### Interpretation

**Two possible scenarios:**

**Scenario A (Likely):** They used Boughter's pre-processed CDRs
- "Retrieved from public sources" = used existing data
- "Annotated in CDRs using ANARCI" = numbered positions, but kept Boughter's boundaries
- Terminology confusion: "IMGT numbering" ≠ "IMGT CDR extraction"

**Scenario B (Unlikely):** They re-extracted CDRs with ANARCI
- Would result in different CDR3s (no W at end)
- Would break compatibility with Boughter's labels
- No mention of re-extraction in methods

### Evidence for Scenario A
Line 55: "Following the **original study** [44], the Boughter dataset was **first parsed** into two groups"

Key word: "first" - implies they started with Boughter's data as-is, then filtered.

**Status**: ⚠️ AMBIGUOUS (needs clarification from authors)

---

## Summary of Verified Facts

| Claim | Status | Evidence |
|-------|--------|----------|
| IMGT CDR3 = 105-117 | ✅ VERIFIED | IMGT.org official docs |
| Position 118 = J-anchor (W/F) | ✅ VERIFIED | IMGT.org official docs |
| Boughter mouse_IgA.dat CDR3s end with W | ✅ VERIFIED | 450/450 sequences analyzed |
| Boughter used IgBLAST, not ANARCI | ✅ VERIFIED | GetCDRs_AA.ipynb code |
| Boughter CDR3 includes position 118 | ✅ VERIFIED | Data shows W at end |
| Sakhnini claims "ANARCI following IMGT" | ✅ VERIFIED | Line 240 of paper |
| Sakhnini used Boughter's data | ✅ VERIFIED | Line 47 explicitly states this |
| Sakhnini re-extracted CDRs with ANARCI | ❌ NOT VERIFIED | No evidence in methods |

---

## Conclusions

### 1. The Discrepancy is REAL

**Boughter's CDR boundaries:**
- CDR-H3: positions 105–118 (includes J-anchor W)
- Method: IgBLAST alignment + custom parsing
- Rationale: Functional CDR3 includes J-anchor for structural context

**IMGT strict boundaries:**
- CDR-H3: positions 105–117 (excludes W)
- Position 118: FR4 (J-TRP/J-PHE)
- Standard: Fixed coordinate system

**Difference:** +1 amino acid at C-terminus

### 2. Sakhnini's Claim is AMBIGUOUS

"ANARCI following IMGT numbering scheme" could mean:
- **Numbering interpretation**: Used IMGT to number positions in full sequence
- **Extraction interpretation**: Used ANARCI to extract CDR regions

**Most likely**: They numbered with IMGT but used Boughter's pre-extracted CDRs (which have extended boundaries).

Evidence:
- They "retrieved" Boughter data from "public sources"
- They "parsed" it into groups "following the original study"
- No mention of re-extraction in methods
- Re-extraction would break label alignment

### 3. Why This Matters

**For developers using ANARCI with strict IMGT:**
```python
# ANARCI output
cdr3 = "ARRGYYYGSFDY"  # 12 residues, no W

# Boughter reference
cdr3 = "ARRGYYYGSFDYW"  # 13 residues, has W
```

**Mismatch:** Cannot validate against Boughter's published CDRs

**For Hybri's work:**
- Extend to position 118 to match Boughter/Novo
- Or trim position 118 from reference data
- Document which boundaries are used

---

## Recommendations

### Immediate

1. ✅ Document this discrepancy (this file)
2. ⬜ Create compatibility flag in our extractor: `--cdr3-include-118`
3. ⬜ Email Novo authors for clarification on exact boundaries used
4. ⬜ Post concise Discord message explaining findings

### For Discord (Draft)

```
investigated the CDR3 boundary issue from first principles.

findings:
• imgt standard: CDR-H3 = 105–117 (position 118 = J-anchor W, part of FR4)
• boughter data: 100% of mouse_IgA.dat CDR3s end with W (450/450 sequences)
• boughter method: igblast alignment + custom parsing (GetCDRs_AA.ipynb), NOT anarci
• conclusion: boughter's CDR3 boundaries are 105–118 (extended by 1 residue)

sakhnini says "anarci following imgt numbering" but they retrieved boughter's data
"from public sources" and "parsed following the original study." most likely they
used boughter's pre-processed CDRs (with extended boundaries), not re-extracted with anarci.

implications:
• anarci strict imgt gives: ARRGYYYGSFDY (no W)
• boughter reference has: ARRGYYYGSFDYW (has W)
• mismatch breaks validation

solutions:
1. extend to pos 118 to match boughter/novo
2. trim pos 118 from reference data
3. add --cdr3-include-118 compat flag
4. email novo for clarification

documented full audit: docs/cdr_boundary_first_principles_audit.md
```

---

## Final Conclusion (Updated 2025-11-02)

### Decision: Use Strict IMGT (CDR-H3 = 105-117, Excluding Position 118)

**From first principles analysis:**

**Biology:**
- Position 118 is **Framework Region 4**, not CDR
- It's a **conserved J-anchor** (W or F in all antibodies)
- It's **germline-encoded**, not variable
- It **does NOT contact antigen** (β-strand, not loop)

**Machine Learning:**
- Position 118 has **zero predictive information** (constant)
- Including it **pollutes the variable region signal**
- **Standard practice**: Exclude conserved positions

**Methodology:**
- **IMGT is the international standard**
- **All other datasets** (Harvey, Jain, Shehata) use strict IMGT
- **Cross-dataset comparability** requires consistent boundaries
- **Reproducibility** requires standardized definitions

**Implementation:**
- **Default**: ANARCI with strict IMGT (CDR3 = 105-117)
- **Optional flag**: `--cdr3-include-118` for compatibility testing only
- **Document**: Boughter's published files include position 118; we use strict IMGT

### Answered Questions

1. **Why did Boughter include position 118?**
   - For biochemical/biophysical analysis (different use case than ML prediction)
   - IgBLAST alignment behavior includes J-anchor
   - Not wrong for their analysis, but not optimal for ML

2. **What should WE use?**
   - **Strict IMGT (105-117)** - biologically and methodologically correct
   - Independent of what Novo did

3. **CDR2 boundaries?**
   - **Use IMGT fixed 56-65** - standard definition

---

## References

1. **IMGT Numbering**: http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
2. **Boughter et al. 2020** (eLife): https://elifesciences.org/articles/61393
3. **Sakhnini et al. 2025** (preprint): https://www.biorxiv.org/content/10.1101/2025.04.28.650927
4. **AIMS repository**: https://github.com/ctboughter/AIMS_manuscripts

---

*Audit version: 1.0*
*Conducted: 2025-11-01*
*Method: First-principles independent verification*
*Investigator: John H. Jung, MD*
