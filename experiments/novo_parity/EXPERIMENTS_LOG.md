# Novo Parity Experiments - Execution Log

**Branch**: `ray/novo-parity-experiments`
**Start Date**: 2025-11-03
**Status**: 🧪 Active

---

## 📋 Quick Reference

**Master Plan**: See `../../NOVO_PARITY_EXPERIMENTS.md`
**Current Experiment**: Exp 01 (Baseline)
**Next Up**: Web search validation of reclassification candidates

---

## 🔄 Execution Timeline

### 2025-11-03 21:45 - Initialization

**Action**: Created experiment infrastructure
- ✅ Branch: `ray/novo-parity-experiments`
- ✅ Directories: `scripts/`, `datasets/`, `results/`
- ✅ Planning doc: `NOVO_PARITY_EXPERIMENTS.md`
- ✅ This log file

**Status**: Ready to begin experiments

**Next**: Write `exp_01_baseline.py` and begin web searches

---

## 📊 Experiment Results

### Exp 01: Baseline
**Status**: 📋 Planned
**Started**: Not yet
**Completed**: Not yet
**Dataset**: TBD
**Confusion Matrix**: N/A
**Notes**: Sanity check - verify starting 116 distribution

---

### Exp 02: Biology-First QC
**Status**: 📋 Planned
**Started**: Not yet
**Completed**: Not yet
**Dataset**: TBD
**Confusion Matrix**: TBD
**Notes**: Assistant's deterministic QC method

---

### Exp 03: Parity Shim
**Status**: 📋 Planned
**Started**: Not yet
**Completed**: Not yet
**Dataset**: TBD
**Confusion Matrix**: TBD
**Notes**: Flip 7 to hit 59/27 from Exp 02

---

## 🔍 Web Search Results

### Search 1: Reclassification Candidates (ELISA=0, total_flags≥3)
**Date**: 2025-11-03
**Target**: 7 reclassification candidates
**Query**: Polyreactivity, aggregation, immunogenicity, clinical issues

**Results**:

**✅ STRONG EVIDENCE (2/7):**

1. **atezolizumab** (total_flags=3)
   - ✅ Aggregation-prone due to aglycosylation (N297A mutation)
   - ✅ High anti-drug antibody (ADA) rates in cancer patients
   - ✅ Tm1=63.55°C, Tagg=60.7°C (thermal instability)
   - **Verdict**: Strong candidate for reclassification to non-specific

2. **infliximab** (total_flags=3)
   - ✅ 61% patients develop anti-drug antibodies!
   - ✅ Aggregates recruit MORE CD4 T-cells than native form
   - ✅ Chimeric antibody → higher immunogenicity
   - ✅ Drug-TNF complexes form multimers (aggregate-like)
   - **Verdict**: Strong candidate for reclassification to non-specific

**❌ NO EVIDENCE (5/7):**

3. **bimagrumab** (total_flags=4)
   - Failed Phase IIb/III for efficacy (not biophysics)
   - Good safety profile, well tolerated
   - **Verdict**: No polyreactivity evidence

4. **eldelumab** (total_flags=3)
   - Failed Phase II for efficacy
   - Well tolerated, no immunogenicity observed
   - **Verdict**: No polyreactivity evidence

5. **glembatumumab vedotin** (total_flags=3)
   - Failed METRIC trial for efficacy
   - Short half-life noted, but not aggregation
   - **Verdict**: No polyreactivity evidence

6. **rilotumumab** (total_flags=3)
   - Failed Phase III (increased deaths, mechanism issue)
   - Not biophysical problems
   - **Verdict**: No polyreactivity evidence

7. **seribantumab** (total_flags=3)
   - Failed Phase II for efficacy
   - Paused for business reasons
   - **Verdict**: No polyreactivity evidence

**Summary**: Only 2/7 candidates have published evidence of polyreactivity/aggregation. This suggests our z-score flag method captured OTHER red flags (BVP, self-interaction, chromatography, stability) but not necessarily polyreactivity.

---

## 💡 Insights & Learnings

### Key Finding 1: Z-Score Flags ≠ Polyreactivity
**Date**: 2025-11-03
**Source**: Web search validation
**Insight**: Only 2/7 ELISA=0 candidates with total_flags≥3 have published polyreactivity/aggregation evidence (atezolizumab, infliximab). The other 5 failed for efficacy/mechanism reasons, not biophysical quality. This means our total_flags metric (BVP + self-interaction + chromatography + stability) captures different risks than ELISA polyreactivity.

### Key Finding 2: Atezolizumab & Infliximab Are Smoking Guns
**Date**: 2025-11-03
**Source**: Web search validation
**Insight**: Both have documented aggregation and immunogenicity issues:
- **atezolizumab**: Aglycosylation → aggregates → high ADA rates
- **infliximab**: 61% ADA rate, aggregates recruit more T-cells, chimeric

If Novo used similar criteria to identify problematic antibodies, these 2 are the strongest candidates for reclassification.

---

## ⚠️ Issues & Blockers

None currently.

---

## 📝 Notes & Ideas

- Consider adding experiment for VL annotation filtering (Novo may have removed antibodies with missing VL data)
- Test different z-score thresholds (|z|≥2.0 vs ≥2.5 vs ≥3.0)
- Check if any antibodies have missing biophysical data that would exclude them

---

---

## 🧪 Action Items

### Immediate Next Steps
1. ✅ **COMPLETED**: Web search validation of 7 reclassification candidates
   - **Result**: 2/7 have strong biophysical evidence (atezolizumab, infliximab)
   - **Implication**: Need to revise Exp 05 strategy

2. 🔄 **IN PROGRESS**: Determine reclassification strategy
   - Option A: Only use 2 defensible candidates (can't reach 59/27 math)
   - Option B: Search for other ELISA=0 antibodies with high total_flags
   - Option C: Web search assistant's 7 parity flips for comparison
   - Option D: Accept that exact Novo parity is not achievable with biophysical justification

3. 📋 **TODO**: Write baseline script (Exp 01)

4. 📋 **TODO**: Web search Priority 2 (assistant's 7 parity flips)

---

**Last Updated**: 2025-11-03 22:15
