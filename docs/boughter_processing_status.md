# Boughter Dataset Processing Status - Complete Analysis

**Date**: 2025-11-02
**Status**: ⚠️ **Partial Information - Unanswered Questions Remain**

---

## Executive Summary

### What We HAVE ✅

1. **Raw Data Files**:
   - All 6 subsets in `test_datasets/boughter/` (DNA FASTA + flags)
   - flu (379), hiv_nat (134), hiv_cntrl (50), hiv_plos (52), gut_hiv (75), mouse_iga (481)

2. **Complete Documentation**:
   - `boughter_data_sources.md` - Dataset specifications from papers
   - `boughter_processing_implementation.md` - Full implementation guide
   - `boughter_cdr_boundary_investigation.md` - CDR boundary analysis

3. **Understanding of Boughter's Method**:
   - Located GetCDRs_AA.ipynb (their extraction code)
   - Uses **igblast + pairwise2 alignment**, NOT ANARCI
   - CDR3 extends to position 118 (includes J-anchor W)
   - CDR2 boundaries vary per sequence (alignment-based)

4. **Implementation Knowledge**:
   - DNA→protein translation (BioPython)
   - ANARCI annotation (IMGT numbering)
   - Novo flagging strategy (0 / 1-3 excluded / >3)
   - 16 fragment extraction patterns

### What We DON'T HAVE ❌

1. **Pre-Extracted CDR Data**:
   - No CSV files with Boughter's pre-extracted CDRs for polyreactivity dataset
   - Only have raw FASTA DNA files (must extract ourselves)

2. **Novo Nordisk's Exact Code**:
   - Unknown: Did they use Boughter's boundaries or ANARCI strict IMGT?
   - Unknown: Was position 118 included or trimmed?
   - Unknown: How were CDR2 variable boundaries handled?

3. **Answers to Critical Questions**:
   - See "Unanswered Questions" section below

---

## Critical Unanswered Questions

### Question 1: CDR Boundary Methodology

**What Novo Says**:
> "The primary sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme" (Sakhnini et al. 2025, Section 4.3, line 239)

**What We Found**:
- Boughter's CDR3 extends to position 118 (not strict IMGT 105-117)
- Boughter used igblast alignment-based extraction
- ANARCI with strict IMGT would give different boundaries

**The Question**:
Did Novo:
- (A) Use Boughter's pre-extracted CDRs with extended boundaries?
- (B) Re-extract using ANARCI and trim to strict IMGT boundaries?
- (C) Use ANARCI but modify to match Boughter's boundaries?

**Status**: ⚠️ **UNANSWERED** - Need Novo's code or clarification

**Evidence** (from `boughter_cdr_boundary_investigation.md`):
```
# Boughter's mouse_IgA.dat CDR3s
ALL sequences end with W (tryptophan) at position 118
Example: "ARRGYYYGSFDYW"  ← W is J-anchor at IMGT 118

# Strict IMGT (ANARCI default)
CDR3 would be: "ARRGYYYGSFD"  ← No W, ends at 117
```

### Question 2: CDR2 Variable Boundaries

**What Hybri Reported** (Discord, Nov 1):
> "then there's a problem with CDR2 ... some CDR2s are shorter than what they're supposed to be ! ... it's not [consistent], I should understand why some CDR2s aren't matching the anarci numbering"

**What We Found**:
- Boughter's GetCDRs_AA.ipynb uses igblast alignment markers
- CDR2 boundaries vary per sequence based on V-gene alignment
- Not fixed IMGT 56-65 coordinates

**The Question**:
How did Novo handle CDR2 boundaries when:
- Some sequences have shorter/longer CDR2 than IMGT standard?
- Igblast gives variable boundaries, ANARCI gives fixed boundaries?

**Status**: ⚠️ **UNANSWERED** - Need Novo's code or clarification

### Question 3: Position 118 Treatment

**The Question**:
Did Novo include or exclude IMGT position 118 in CDR3?

**Impact**:
- Include 118: CDR3 sequences match Boughter's published data
- Exclude 118: CDR3 sequences match strict IMGT definition

**Evidence**:
- Boughter's data: Position 118 INCLUDED
- Novo's statement: "IMGT numbering" (ambiguous)
- Standard IMGT CDR3: Position 118 EXCLUDED (it's FR4)

**Status**: ⚠️ **UNANSWERED** - Need Novo's code or clarification

---

## What Can We Do WITHOUT Novo's Code?

### Option 1: ANARCI with Strict IMGT (Conservative)

**Approach**:
```python
# Use ANARCI with standard IMGT boundaries
# CDR3: 105-117 (no position 118)
# CDR2: 56-65 (fixed)
```

**Pros**:
- Standard, reproducible methodology
- Matches IMGT specification
- Matches what Novo stated in paper

**Cons**:
- CDR sequences won't match Boughter's published data
- May not match Novo's actual implementation
- Validation against Boughter data will show mismatches

### Option 2: Replicate Boughter's IgBLAST Method (Exact Match)

**Approach**:
```python
# Translate DNA → protein
# Run igblast alignment
# Parse CDR boundaries from igblast output
# Use pairwise2 for J-gene alignment (CDR3)
```

**Pros**:
- CDR sequences exactly match Boughter's published data
- No ambiguity about boundaries
- Can validate against mouse_IgA.dat

**Cons**:
- Complex implementation (requires igblast + custom parsing)
- May not match what Novo actually did
- Requires J-gene reference files

### Option 3: ANARCI + Position 118 Extension (Hybrid)

**Approach**:
```python
# Use ANARCI for CDR1, CDR2, FWRs
# Extend CDR3 to include position 118 (custom modification)
# Document this choice explicitly
```

**Pros**:
- Simpler than full igblast replication
- CDR3 matches Boughter's data
- ANARCI handles most complexity

**Cons**:
- Still doesn't solve CDR2 variable boundary issue
- Requires ANARCI modification
- May not match Novo's exact method

### Option 4: Wait for Novo's Code Release

**Status** (from Discord):
- Hybri: "wait for novo's code? [...] we can't know the logic behind it if they don't disclose it"
- Mentioned "2 weeks" for code release

**Pros**:
- Get exact methodology from source
- No guesswork
- Perfect replication

**Cons**:
- Unknown timeline (may be delayed)
- Blocks progress

---

## Our Current Implementation Readiness

### What We CAN Implement NOW ✅

**Stage 1: DNA Translation & CSV Conversion**
```
✅ FASTA DNA parsing
✅ Flag file parsing (NumReact 0-7 and Y/N)
✅ BioPython DNA→protein translation
✅ Sequence pairing (heavy + light + flags)
✅ Novo flagging strategy (0, 1-3 excluded, >3)
✅ CSV output generation
```

**Stage 2: ANARCI Annotation (with caveats)**
```
✅ ANARCI setup (riot_na)
✅ IMGT numbering
⚠️ CDR boundary extraction (strict IMGT - may not match Novo)
✅ 16 fragment types
✅ CSV output per fragment
```

### What Requires Clarification ⚠️

1. **CDR3 boundary**: Position 117 or 118?
2. **CDR2 boundary**: Fixed IMGT or variable igblast?
3. **Validation strategy**: How to verify correctness without Novo's data?

---

## Recommendation

### Immediate Action: Implement Option 1 + Documentation

**Rationale**:
1. Option 1 (ANARCI strict IMGT) is **defensible and reproducible**
2. We can **document the discrepancy explicitly**
3. We can **add Option 3 as a flag** for compatibility mode
4. We can **update when Novo releases code**

**Implementation Plan**:
```python
# convert_boughter_to_csv.py
# - Translate DNA → protein ✅
# - Apply Novo flagging ✅
# - Output: boughter.csv ✅

# process_boughter.py
# - Use ANARCI with IMGT numbering ✅
# - Default: CDR3 = 105-117 (strict IMGT)
# - Flag: --cdr3-include-118 (Boughter compat mode)
# - Document choice in CSV metadata
# - Output: 16 fragment CSVs

# validate_boughter.py
# - Check ANARCI success rate
# - Compare CDR lengths to expected ranges
# - Log discrepancies vs Boughter's published data
# - Generate validation report
```

### Documentation Requirements

**In Every Output File**:
```
# Metadata
processing_date: 2025-11-02
cdr_extraction_method: ANARCI (IMGT numbering)
cdr3_boundary: 105-117 (strict IMGT) OR 105-118 (Boughter compat)
cdr2_boundary: 56-65 (fixed IMGT)
known_discrepancy: See docs/boughter_cdr_boundary_investigation.md
novo_replication_status: Partial (awaiting Novo code release)
```

### Parallel Track: Request Clarification

**Email to Novo Authors** (draft):
```
Subject: Clarification on Boughter Dataset CDR Extraction Boundaries

Dear Dr. Sakhnini, Dr. Lorenzen, Dr. Vendruscolo, and Dr. Granata,

We are replicating your excellent work on antibody non-specificity
prediction (Sakhnini et al. 2025). We have a technical question
regarding the Boughter dataset preprocessing:

Your paper states: "sequences were annotated in the CDRs using ANARCI
following the IMGT numbering scheme" (Section 4.3).

However, Boughter's published mouse_IgA.dat file shows CDR-H3
sequences ending with W (e.g., "ARRGYYYGSFDYW"), suggesting position
118 is included. Standard IMGT CDR-H3 is positions 105-117 (excluding
position 118, which is FR4 J-anchor).

Could you clarify:
1. Did you use CDR3 boundaries 105-117 (strict IMGT) or 105-118
   (Boughter's igblast-based)?
2. For CDR2, did you use fixed IMGT 56-65 or variable igblast
   alignment boundaries?
3. Did you re-extract CDRs or use Boughter's pre-processed data?

This would help us ensure exact replication of your methodology.

Thank you for your time and for making your work publicly available.

Best regards,
[Name]
```

---

## Files Created

1. ✅ `docs/boughter_data_sources.md` - Dataset specifications
2. ✅ `docs/boughter_processing_implementation.md` - Full implementation guide
3. ✅ `docs/boughter_cdr_boundary_investigation.md` - Boundary analysis
4. ✅ `docs/boughter_processing_status.md` - This file

---

## Next Steps

### For Implementation:

1. **Write `convert_boughter_to_csv.py`**:
   - Implement Stage 1 (DNA translation)
   - Use BioPython for translation
   - Apply Novo flagging
   - Generate boughter.csv

2. **Write `process_boughter.py`**:
   - Implement Stage 2 (ANARCI annotation)
   - Default: strict IMGT boundaries
   - Add `--cdr3-include-118` flag for compatibility
   - Generate 16 fragment CSVs

3. **Write `validate_boughter.py`**:
   - Verify ANARCI success rate >95%
   - Check CDR length distributions
   - Generate validation report
   - Document discrepancies vs Boughter data

4. **Test on Small Subset**:
   - Run flu subset only (379 sequences)
   - Verify output format
   - Check for errors

5. **Request AI Senior Consensus Review**:
   - Review all 4 documentation files
   - Validate implementation approach
   - Approve proceeding with Option 1

### For Clarification:

1. **Email Novo authors** (after senior review)
2. **Monitor Novo code release** (check GitHub/HuggingFace)
3. **Update implementation when clarified**

---

## Summary: Do We Have All Answers?

**NO** - We have critical unanswered questions about:
- ❌ Exact CDR boundary methodology (especially CDR2)
- ❌ Position 118 inclusion/exclusion
- ❌ Whether Novo used Boughter's pre-extracted CDRs

**BUT** - We CAN proceed with:
- ✅ Well-documented implementation using ANARCI (Option 1)
- ✅ Compatibility mode for Boughter boundaries (Option 3 hybrid)
- ✅ Explicit documentation of discrepancies
- ✅ Update plan when Novo releases code

**RECOMMENDATION**:
- Implement Option 1 (ANARCI strict IMGT) + documentation
- Add Option 3 flag (--cdr3-include-118) for testing
- Request clarification from Novo authors in parallel
- Update implementation when answers arrive

---

**Document Version**: 1.0
**Author**: Claude Code
**Reviewed By**: [Pending AI Senior Consensus]
**Status**: Ready for Implementation Decision
