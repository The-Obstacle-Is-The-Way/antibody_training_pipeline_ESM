# Novo Parity Experimentation Plan

**Branch**: `ray/novo-parity-experiments`
**Date**: 2025-11-03
**Status**: 🧪 Active experimentation | Full traceability required
**Goal**: Explore permutations to understand Novo's 86-antibody (59 spec / 27 nonspec) construction

---

## 🎯 PRIMARY OBJECTIVE

**Test multiple hypotheses** for how Novo got from 116 → 86 antibodies with 59/27 distribution.

**Ground truth**: We start with 116 ELISA-only antibodies (94 specific / 22 non-specific)
**Target**: 86 antibodies (59 specific / 27 non-specific)
**Constraint**: Cannot reach 27 non-specific without reclassification or importing antibodies

**Critical acceptance**: We likely won't match Novo exactly - this is exploratory!
**Primary SSOT**: The full 116-antibody ELISA-only set remains our main test set

---

## 📁 FILE ORGANIZATION

### Directory Structure
```
experiments/novo_parity/
├── scripts/              # Numbered experiment scripts
│   ├── exp_01_baseline.py
│   ├── exp_02_biology_qc.py
│   └── ...
├── datasets/            # Generated test sets
│   ├── jain_exp_01_baseline_86.csv
│   ├── jain_exp_02_biology_qc_86.csv
│   └── ...
├── results/             # Inference results & audit logs
│   ├── exp_01_results.json
│   ├── exp_01_confusion_matrix.txt
│   └── ...
└── EXPERIMENTS_LOG.md   # Running log of all experiments
```

### Naming Conventions

**Scripts**: `exp_{N:02d}_{short_description}.py`
- Example: `exp_01_baseline.py`, `exp_02_biology_qc.py`

**Datasets**: `jain_exp_{N:02d}_{description}_{size}.csv`
- Example: `jain_exp_01_baseline_86.csv`

**Results**: `exp_{N:02d}_{metric}.{json|txt|csv}`
- Example: `exp_01_results.json`, `exp_01_confusion_matrix.txt`

**Audit logs**: `exp_{N:02d}_audit.json`
- Contains: script path, input files, removed IDs, reclassified IDs, timestamps

---

## 🧪 EXPERIMENT REGISTRY

### Experiment 01: Baseline (No QC, Just Count)
**Status**: 📋 Planned
**Hypothesis**: Verify starting numbers
**Method**: Load 116 ELISA-only, count distribution
**Expected**: 94 specific / 22 non-specific = 116 total
**Script**: `experiments/novo_parity/scripts/exp_01_baseline.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_01_baseline_116.csv`
**Web validation**: N/A (sanity check)

---

### Experiment 02: Biology-First QC (Assistant's Method)
**Status**: 📋 Planned
**Hypothesis**: Sequence/biophysics QC yields 66/20 split
**Method**:
  1. Remove VH/VL length outliers (|z|≥2.5)
  2. Remove CDR-H3 pathologies (<5 or >26 aa, |z|≥2.5)
  3. Remove charge extremes (|z|≥2.5)
  4. Tiebreaker removal to reach 86 total

**Expected**: 66 specific / 20 non-specific = 86 total
**Removed IDs**:
  - cixutumumab (VH outlier)
  - muromonab (CDR-H3)
  - lenzilumab (charge)
  - 27 via tiebreaker

**Script**: `experiments/novo_parity/scripts/exp_02_biology_qc.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_02_biology_qc_86.csv`
**Web validation**: TODO - search for clinical/biology issues with removed antibodies

---

### Experiment 03: Parity Shim (Reclassify 7)
**Status**: 📋 Planned
**Hypothesis**: Starting from Exp02 (66/20), flip 7 ELISA=0 → nonspec to hit 59/27
**Method**:
  1. Start with exp_02 dataset (86 antibodies, 66/20)
  2. Reclassify 7 specific → non-specific (deterministic ranking)
  3. Keep same 86 antibodies

**Reclassified IDs** (from assistant's analysis):
  - ganitumab
  - gemtuzumab
  - panitumumab
  - tigatuzumab
  - girentuximab
  - efalizumab
  - zanolimumab

**Expected**: 59 specific / 27 non-specific = 86 total
**Script**: `experiments/novo_parity/scripts/exp_03_parity_shim.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_03_parity_shim_86.csv`
**Web validation**: TODO - search each of the 7 for polyreactivity/clinical issues

---

### Experiment 04: Our Z-Score QC (24 Candidates)
**Status**: 📋 Planned
**Hypothesis**: Remove our 24 QC candidates (z-score + known issues)
**Method**:
  1. Start with 116 ELISA-only
  2. Remove 24 QC candidates identified via z-scoring
  3. Results in 92 antibodies (73 spec / 19 nonspec)
  4. Need to handle the remaining gap

**Removed IDs** (24 total):
  - cixutumumab (VH outlier + nonspec)
  - dalotuzumab, robatumumab (nonspec with issues)
  - 21 specific with chimeric/discontinued/withdrawn/outliers

**Expected**: 92 antibodies (73 specific / 19 non-specific)
**Gap**: Still 6 antibodies away from 86, and wrong distribution
**Script**: `experiments/novo_parity/scripts/exp_04_zscore_qc.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_04_zscore_qc_92.csv`
**Web validation**: TODO - verify chimeric/discontinued status

---

### Experiment 05: Reclass Then Remove (Our Hypothesis)
**Status**: 📋 Planned
**Hypothesis**: Reclassify 5 ELISA=0 (with other flags) → nonspec, THEN remove 30 specific
**Method**:
  1. Start with 116 ELISA-only (94 spec / 22 nonspec)
  2. Reclassify 5 of our 7 candidates (ELISA=0, total≥3) → non-specific
  3. Result: 89 spec / 27 nonspec = 116 total
  4. Remove 30 specific antibodies (21 QC candidates + 9 more)
  5. Final: 59 spec / 27 nonspec = 86 total

**Reclassification candidates** (pick 5 of 7):
  - bimagrumab (ELISA=0, total=4) ← Top pick
  - atezolizumab (ELISA=0, total=3)
  - eldelumab (ELISA=0, total=3)
  - glembatumumab (ELISA=0, total=3)
  - infliximab (ELISA=0, total=3)
  - rilotumumab (ELISA=0, total=3)
  - seribantumab (ELISA=0, total=3)

**Removed IDs** (30 total):
  - 21 from our QC list (z-score analysis)
  - 9 more specific (TBD - need to identify)

**Expected**: 59 specific / 27 non-specific = 86 total
**Script**: `experiments/novo_parity/scripts/exp_05_reclass_then_remove.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_05_reclass_remove_86.csv`
**Web validation**: TODO - search 7 reclass candidates for polyreactivity/aggregation

---

### Experiment 06: ELISA Threshold ≥3 (Alternative Hypothesis)
**Status**: 📋 Planned
**Hypothesis**: Novo used ELISA≥3 instead of ≥4 to define non-specific
**Method**:
  1. Start with full 137 antibodies
  2. Exclude mild (ELISA 1-2) instead of (ELISA 1-3)
  3. Count resulting distribution
  4. Apply QC to reach 86

**Expected**: More non-specific antibodies (possibly 27+)
**Script**: `experiments/novo_parity/scripts/exp_06_threshold_3.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_06_threshold3_86.csv`
**Web validation**: Check Novo paper for exact ">3" interpretation

---

### Experiment 07: Chimeric-Only Removal
**Status**: 📋 Planned
**Hypothesis**: Novo removed ALL chimeric antibodies regardless of other QC
**Method**:
  1. Start with 116 ELISA-only
  2. Remove all chimeric antibodies (11 total from our analysis)
  3. See resulting distribution
  4. Adjust to reach 86

**Chimeric antibodies** (11 from our analysis):
  - muromonab, cetuximab, basiliximab, infliximab
  - rituximab, alemtuzumab, bevacizumab, panitumumab
  - girentuximab, trastuzumab, gemtuzumab

**Expected**: 105 antibodies → need to remove 19 more
**Script**: `experiments/novo_parity/scripts/exp_07_chimeric_removal.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_07_chimeric_86.csv`
**Web validation**: TODO - verify all 11 are truly chimeric (mouse/human)

---

### Experiment 08: Discontinued-Only Removal
**Status**: 📋 Planned
**Hypothesis**: Novo removed failed/discontinued clinical trials
**Method**:
  1. Start with 116 ELISA-only
  2. Remove discontinued antibodies (11 from our list)
  3. See resulting distribution

**Discontinued antibodies** (11 from our analysis):
  - tabalumab, girentuximab, abituzumab, dalotuzumab
  - ganitumab, robatumumab, tigatuzumab, farletuzumab
  - zanolimumab, (others TBD)

**Expected**: 105 antibodies → need to remove 19 more
**Script**: `experiments/novo_parity/scripts/exp_08_discontinued_removal.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_08_discontinued_86.csv`
**Web validation**: TODO - verify discontinued status for each

---

### Experiment 09: Combined QC (Chimeric + Discontinued + Outliers)
**Status**: 📋 Planned
**Hypothesis**: Novo applied multiple QC criteria simultaneously
**Method**:
  1. Start with 116 ELISA-only
  2. Remove chimeric (11)
  3. Remove discontinued (11, some overlap)
  4. Remove VH outliers (3)
  5. Remove charge outliers (2)
  6. See if this naturally hits 86

**Expected**: Likely removes more than 30, need to be selective
**Script**: `experiments/novo_parity/scripts/exp_09_combined_qc.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_09_combined_86.csv`
**Web validation**: N/A (combination of previous validations)

---

### Experiment 10: Top-30 Specific by Risk Score
**Status**: 📋 Planned
**Hypothesis**: Remove the 30 "riskiest" specific antibodies by composite score
**Method**:
  1. Start with 116 ELISA-only (94 spec / 22 nonspec)
  2. Score each specific antibody by: chimeric + discontinued + |VH_z| + |charge_z| + outlier_hits
  3. Remove top 30 specific by score
  4. Reclassify 5 ELISA=0 → nonspec to hit 27
  5. Final: 59 spec / 27 nonspec = 86

**Expected**: 59 specific / 27 non-specific = 86 total
**Script**: `experiments/novo_parity/scripts/exp_10_risk_score.py`
**Output**: `experiments/novo_parity/datasets/jain_exp_10_risk_score_86.csv`
**Web validation**: N/A (composite scoring)

---

### Experiment 11: [RESERVED FOR WEB SEARCH INSIGHTS]
**Status**: 💡 TBD based on web search findings
**Hypothesis**: TBD after searching the 7 reclass candidates
**Method**: TBD
**Script**: TBD
**Output**: TBD
**Web validation**: Primary driver for this experiment

---

### Experiment 12: [RESERVED]
**Status**: 💡 TBD
**Hypothesis**: TBD

---

## 🔍 WEB SEARCH VALIDATION QUEUE

### Priority 1: The 7 Reclassification Candidates
Need to search for polyreactivity, aggregation, clinical issues:
- [ ] bimagrumab
- [ ] atezolizumab
- [ ] eldelumab
- [ ] glembatumumab
- [ ] infliximab
- [ ] rilotumumab
- [ ] seribantumab

### Priority 2: Assistant's 7 Parity Flips
Need to validate why these were chosen:
- [ ] ganitumab
- [ ] gemtuzumab
- [ ] panitumumab
- [ ] tigatuzumab
- [ ] girentuximab
- [ ] efalizumab
- [ ] zanolimumab

### Priority 3: Chimeric Antibodies
Verify all 11 are truly chimeric:
- [ ] muromonab, cetuximab, basiliximab, infliximab
- [ ] rituximab, alemtuzumab, bevacizumab, panitumumab
- [ ] girentuximab, trastuzumab, gemtuzumab

### Priority 4: Discontinued Antibodies
Verify clinical trial failures:
- [ ] tabalumab, girentuximab, abituzumab, dalotuzumab
- [ ] ganitumab, robatumumab, tigatuzumab, farletuzumab
- [ ] zanolimumab

---

## 🔬 WEB SEARCH FINDINGS

**Date**: 2025-11-03
**Search Query**: Polyreactivity, aggregation, off-target binding, biophysical issues, immunogenicity
**Results**: **2/7 candidates have STRONG evidence** for reclassification

### ✅ STRONG EVIDENCE FOR RECLASSIFICATION (2/7)

#### 1. **atezolizumab** - Aggregation-Prone, Non-Trivial Immunogenicity
**Evidence**:
- **Aglycosylated design (N297A mutation)** → eliminates Fc function BUT creates stability problems
- **Unstable thermal profile**: Tm1 ≈ 63.6°C, Tagg ≈ 60.7°C (low for therapeutic antibody)
- **Aggregation-prone**: Engineering work on "Maxatezo" explicitly reports aglycosylated atezolizumab is unstable and prone to aggregation
- **ADA rates**: ~30% in OAK NSCLC study; 13-36% across studies (assay-dependent per FDA labeling)
- **Non-trivial but variable**: ADA incidence varies by trial, not uniformly "high"

**Sources**:
- Nature Scientific Reports (2021): Glycosylated variant improves stability vs aglycosylated form
- FDA labeling: 30% ADA in OAK trial, 13-36% range across studies
- Structural studies show large buried surface area (>2000 Å²)

**Verdict**: ✅ **CONFIRMED candidate for reclassification** - aggregation tendency + non-trivial ADA despite ELISA=0

---

#### 2. **infliximab** - High Immunogenicity, Aggregation Issues
**Evidence**:
- **61% ADA rate**: Classic NEJM Crohn's cohort showed 61% developed anti-infliximab antibodies, with shorter response duration and more infusion reactions
- **Chimeric antibody** (mouse/human) → inherently more immunogenic
- **Aggregates drive CD4+ T-cell activation**: Heat-stressed infliximab aggregates drive extended CD4+ T-cell responses in vitro
  - Aggregates expose larger array of T-cell epitopes across entire variable regions
  - Even mild-condition aggregates induce higher frequency CD4 T-cell response vs native
- **Immune complex formation**: IFX–TNF immune complexes show Fc-dependent uptake; TNF trimer stabilization documented
- **Anti-drug antibodies neutralize binding** → enables TNF-α pro-inflammatory effects
- **Associated with allergic reactions and loss of response**

**Sources**:
- NEJM (2003): 61% ADA in Crohn's disease cohort
- Frontiers Immunology (2020): Aggregate-induced T-cell activation
- Multiple studies on immune complex formation and multimeric target binding

**Verdict**: ✅ **CONFIRMED candidate for reclassification** - high ADA rate + aggregate-driven immunogenicity

---

### ❌ NO PUBLIC PSR/BVP EVIDENCE (5/7)

**Framing**: These antibodies failed for efficacy or clinical safety reasons, NOT developability flags. No public polyreactivity (PSR) or biophysical property (BVP) red flags documented.

#### 3. **bimagrumab** - Clinical Efficacy Failure, Well-Tolerated
**Evidence**:
- sIBM trials: well-tolerated, no functional benefit
- Obesity studies: show fat-mass loss with preserved/increased lean mass
- **Safety profile**: Good - well tolerated up to 2 years
- **No public PSR/BVP red flags**
- Later got FDA Breakthrough Therapy designation (still in development)

**Verdict**: ❌ **NO polyreactivity evidence** - failed for efficacy, not biophysics

---

#### 4. **eldelumab** - Phase 2 UC/CD, Limited/Mixed Efficacy
**Evidence**:
- Phase 2 UC/CD: primary endpoint not achieved
- **Safety profile**: Generally acceptable safety
- **No immunogenicity observed** in trials
- **No public polyreactivity evidence**

**Verdict**: ❌ **NO polyreactivity evidence** - failed for efficacy, not biophysics

---

#### 5. **glembatumumab vedotin** - METRIC Phase 2b PFS Failure
**Evidence**:
- METRIC Phase 2b failed PFS primary endpoint
- No developability/aggregation signal publicly reported
- ADC safety profile generally acceptable
- **No public PSR/BVP red flags**

**Verdict**: ❌ **NO polyreactivity evidence** - failed for efficacy, not biophysics

---

#### 6. **rilotumumab** - Phase 3 Safety Signal (Excess Deaths)
**Evidence**:
- Phase 3 terminated for excess deaths vs control
- **Safety signal**: Mechanism-related, not polyreactivity
- HGF/MET pathway toxicity
- **No public PSR/BVP red flags**

**Verdict**: ❌ **NO polyreactivity evidence** - safety signal is mechanism-based, not biophysics

---

#### 7. **seribantumab** - Mixed Trials, NRG1+ Programs Show Response
**Evidence**:
- Older trials mixed efficacy
- NRG1-fusion programs now show responses with manageable safety
- **No public PSR/BVP red flags**
- Development continues in select indications

**Verdict**: ❌ **NO polyreactivity evidence** - no developability issues documented

---

### 📊 SUMMARY: Web Search Results

| Antibody | ELISA | total_flags | Biophysical Issues? | Reclassification? |
|----------|-------|-------------|---------------------|-------------------|
| **atezolizumab** | 0 | 3 | ✅ Aggregation + High ADA | ✅ **YES** |
| **infliximab** | 0 | 3 | ✅ 61% ADA + Aggregation | ✅ **YES** |
| bimagrumab | 0 | 4 | ❌ Efficacy failure only | ❌ NO |
| eldelumab | 0 | 3 | ❌ Efficacy failure only | ❌ NO |
| glembatumumab | 0 | 3 | ❌ Efficacy failure only | ❌ NO |
| rilotumumab | 0 | 3 | ❌ Mechanism toxicity | ❌ NO |
| seribantumab | 0 | 3 | ❌ Efficacy failure only | ❌ NO |

**Key Finding**: Only **2/7 candidates** have published PSR/BVP evidence justifying reclassification from specific → non-specific. The other 5 failed for efficacy or mechanism-related safety, NOT developability flags.

---

### 🧠 IMPLICATIONS FOR EXPERIMENT DESIGN

**Original Hypothesis** (Exp 05):
- Reclassify 5 of these 7 from ELISA=0 → non-specific based on total_flags≥3
- **Problem**: Only 2/7 have public PSR/BVP justification!

**Honest Framing Options**:
1. **Evidence-based (2 only)**: atezolizumab + infliximab
   - Results in 62/24 split (not 59/27)
   - Most scientifically rigorous

2. **Parity levers (use 5 including 3 without evidence)**:
   - Explicitly document that 3 have "no public PSR/BVP evidence—used only as parity levers"
   - Acknowledge this is a methodological choice to match Novo's distribution
   - Less defensible scientifically

3. **Search for OTHER ELISA=0 antibodies** with total_flags≥3:
   - May find better-justified candidates in the 116 set
   - Need to scan full dataset

**Next Action**: Search the 116 set for OTHER ELISA=0 antibodies with total_flags≥3 to find 3 more candidates with potential justification

---

## 🔬 BREAKTHROUGH: JAIN 2017 PNAS SUPPLEMENTAL DATA

**Date**: 2025-11-03
**Discovery**: Converted raw Jain PNAS .xlsx files (SD01, SD02, SD03) to CSV and found **actual biophysical measurements**!

### 📊 SD03 Contains Gold Standard Biophysical Assays:

**Key Columns**:
- **PSR (Poly-Specificity Reagent)**: 0-1 score, >0.4 = high polyreactivity
- **AC-SINS (Affinity-Capture Self-Interaction Nanoparticle Spectroscopy)**: ∆λmax (nm), >20 = strong self-interaction/aggregation
- **HIC Retention Time**: Hydrophobic interaction chromatography, >10 min = hydrophobic
- **Fab Tm by DSF**: Thermal stability, <65°C = unstable
- **SGAC-SINS**: Salt-gradient aggregation propensity
- **ELISA**: Raw ELISA OD values (not just 0-6 flags!)
- **BVP ELISA**: Baculovirus particle ELISA (orthogonal polyreactivity assay)

### 🧪 Research Logic: ELISA First, Then Biophysics QC

**What Novo LIKELY Did (Industry Standard Workflow)**:

```
Step 1: ELISA Primary Filter
  137 total → Remove mild (ELISA 1-3) → 116 (94 spec / 22 nonspec)

Step 2: Biophysical QC (Secondary/Confirmatory)
  116 → Calculate composite risk from PSR + AC-SINS + HIC + Tm
       → Remove/reclassify high-risk antibodies → 86 (59 spec / 27 nonspec)
```

**Why this makes sense**:
- ELISA is PRIMARY polyreactivity assay (most sensitive)
- PSR, AC-SINS, HIC are SECONDARY/orthogonal confirmatory assays
- Standard pharma workflow: categorize first, then apply developability filters
- Composite risk scoring is industry best practice (multiple orthogonal measurements)

---

### 📊 Biophysical Data Analysis: The 116 ELISA-Only Set

**High PSR (>0.4) in our 116 set:**

| Antibody | Label | ELISA | PSR Score | Notes |
|----------|-------|-------|-----------|-------|
| ixekizumab | 1 (nonspec) | 6 | 0.810 | Already caught by ELISA ✅ |
| bococizumab | 1 (nonspec) | 6 | 0.760 | Known failure case, ELISA caught it ✅ |
| bimagrumab | 0 (spec) | 0 | 0.697 | ⚠️ ELISA missed it! |
| cixutumumab | 1 (nonspec) | 6 | 0.657 | ELISA caught it ✅ |
| lenzilumab | 1 (nonspec) | 6 | 0.655 | ELISA caught it ✅ |
| emibetuzumab | 1 (nonspec) | 6 | 0.643 | ELISA caught it ✅ |
| bavituximab | 0 (spec) | 0 | 0.557 | ⚠️ ELISA missed it! |
| briakinumab | 1 (nonspec) | 4 | 0.556 | ELISA caught it ✅ |
| ganitumab | 0 (spec) | 0 | 0.553 | ⚠️ ELISA missed it! |
| gantenerumab | 1 (nonspec) | 6 | 0.553 | ELISA caught it ✅ |
| patritumab | 1 (nonspec) | 4 | 0.520 | ELISA caught it ✅ |
| olaratumab | 0 (spec) | 0 | 0.483 | ⚠️ ELISA missed it! |

**Key Finding**: 8/12 high-PSR antibodies were correctly caught by ELISA. But **4 specific antibodies have PSR >0.4 despite ELISA=0**!

**The 4 ELISA-discordant cases:**
1. bimagrumab (PSR=0.697, total_flags=4)
2. bavituximab (PSR=0.557, total_flags=2)
3. ganitumab (PSR=0.553, total_flags=2)
4. olaratumab (PSR=0.483, total_flags=2)

---

### 🧮 Composite Biophysical Risk Scoring

**Formula**: Risk = PSR_norm + AC-SINS_norm + HIC_norm + (1 - Tm_norm)

Each metric normalized 0-1 within the 94 specific antibodies, then summed.

**Top 10 Riskiest Specific Antibodies** (from 116 set):

| Rank | Antibody | Risk Score | PSR | AC-SINS | HIC | Tm | Notes |
|------|----------|------------|-----|---------|-----|----|-------|
| 1 | bavituximab | 2.74 | 0.557 | Extreme | High | 59.5°C | All metrics bad |
| 2 | lirilumab | 2.57 | 0.183 | High | Extreme | 69.0°C | HIC retention problem |
| 3 | bimagrumab | 2.40 | 0.697 | Extreme | Med | 72.0°C | PSR + AC-SINS |
| 4 | basiliximab | 2.40 | 0.397 | Extreme | Low | 60.5°C | AC-SINS + low Tm |
| 5 | glembatumumab | 2.11 | 0.166 | Extreme | High | 72.0°C | AC-SINS + HIC |
| 6 | urelumab | 1.95 | 0.0 | Extreme | Med | 74.5°C | AC-SINS only |
| 7 | infliximab | 1.95 | 0.0 | Extreme | Med | 81.5°C | AC-SINS (+ 61% ADA!) |
| 8 | drozitumab | 1.93 | 0.0 | Extreme | Low | 68.0°C | AC-SINS only |
| 9 | tremelimumab | 1.84 | 0.145 | Extreme | High | 71.5°C | AC-SINS + HIC |
| 10 | nimotuzumab | 1.83 | 0.0 | Low | Extreme | 68.5°C | HIC retention problem |

**Notable**: Infliximab (#7) has high composite risk + literature evidence (61% ADA, aggregation). Atezolizumab is #13 with known aggregation issues.

---

### 🎯 Hypothesis: Composite Risk Threshold for 116→86

**Math Requirements**:
- Start: 94 specific + 22 non-specific = 116
- Target: 59 specific + 27 non-specific = 86
- Need: Remove 35 specific, add 5 non-specific (net: -30 antibodies)

**Approach**:
1. **Reclassify 4-5 high-PSR specific → non-specific** (ELISA discordant cases)
   - bimagrumab, bavituximab, ganitumab, olaratumab
   - Plus 1 more: infliximab or atezolizumab?

2. **Remove top 30-35 by composite risk**
   - Use composite risk threshold to identify developability failures
   - Orthogonal biophysical evidence trumps ELISA=0

---

### 🧪 Experiments to Test (Using Real Biophysics)

**Experiment 11**: PSR Threshold Reclassification
- Reclassify all ELISA=0 with PSR >0.4 → non-specific
- See if this explains the +5 non-specific shift

**Experiment 12**: Composite Risk Top-35 Removal
- Remove top 35 specific antibodies by composite risk score
- Check if remaining distribution matches 59/27

**Experiment 13**: Combined PSR + Risk Removal
- Reclassify 4 PSR-discordant → non-specific
- Remove next 30-31 by composite risk
- Should yield 59/27 exactly

**Experiment 14**: Sensitivity Analysis
- Vary PSR threshold (0.3, 0.4, 0.5, 0.6)
- Vary risk threshold
- Find optimal combination that hits 59/27

---

## 📊 RESULTS TRACKING

### Confusion Matrix Comparison Template

**Target (Novo's reported)**:
```
                Predicted
              Spec | Nonspec
Actual   ──────────┼─────────
Spec        40    |   19      (59 total)
Nonspec     10    |   17      (27 total)

Accuracy: 66.28% (57/86)
```

**Experiment N**:
```
                Predicted
              Spec | Nonspec
Actual   ──────────┼─────────
Spec        ??    |   ??      (?? total)
Nonspec     ??    |   ??      (?? total)

Accuracy: ??.??% (??/??)
Delta: ±?.??%
```

### Results Summary Table

| Exp | Description | Size | Spec/Nonspec | Accuracy | Δ Novo | Status |
|-----|-------------|------|--------------|----------|--------|--------|
| 01  | Baseline    | 116  | 94/22        | N/A      | N/A    | 📋 Planned |
| 02  | Biology QC  | 86   | 66/20        | TBD      | TBD    | 📋 Planned |
| 03  | Parity Shim | 86   | 59/27        | TBD      | TBD    | 📋 Planned |
| 04  | Z-score QC  | 92   | 73/19        | TBD      | TBD    | 📋 Planned |
| 05  | Reclass+Rem | 86   | 59/27        | TBD      | TBD    | 📋 Planned |
| 06  | Threshold 3 | 86?  | ?/?          | TBD      | TBD    | 📋 Planned |
| 07  | Chimeric    | 86   | ?/?          | TBD      | TBD    | 📋 Planned |
| 08  | Discontinue | 86   | ?/?          | TBD      | TBD    | 📋 Planned |
| 09  | Combined QC | 86   | ?/?          | TBD      | TBD    | 📋 Planned |
| 10  | Risk Score  | 86   | 59/27        | TBD      | TBD    | 📋 Planned |

---

## 🔗 PROVENANCE CHAIN

### Input Files
- `test_datasets/jain_ELISA_ONLY_116.csv` (SSOT: 116 antibodies, 94 spec / 22 nonspec)
- `test_datasets/jain/jain_116_qc_candidates.csv` (24 QC candidates from z-score analysis)
- `test_datasets/jain/jain_ELISA_ONLY_116_with_zscores.csv` (with all features)

### Generated Files
Each experiment generates:
1. **Dataset CSV**: `experiments/novo_parity/datasets/jain_exp_{N}_{desc}_{size}.csv`
2. **Audit log**: `experiments/novo_parity/results/exp_{N}_audit.json`
3. **Confusion matrix**: `experiments/novo_parity/results/exp_{N}_confusion_matrix.txt`
4. **Inference results**: `experiments/novo_parity/results/exp_{N}_results.json`

### Audit Log Schema
```json
{
  "experiment_id": "exp_01",
  "timestamp": "2025-11-03T21:45:00Z",
  "script": "experiments/novo_parity/scripts/exp_01_baseline.py",
  "input_files": ["test_datasets/jain_ELISA_ONLY_116.csv"],
  "output_file": "experiments/novo_parity/datasets/jain_exp_01_baseline_116.csv",
  "method": "Baseline count, no modifications",
  "removed_ids": [],
  "reclassified_ids": [],
  "final_counts": {
    "total": 116,
    "specific": 94,
    "non_specific": 22
  },
  "parameters": {},
  "notes": "Sanity check - verify starting distribution"
}
```

---

## 🚀 EXECUTION PLAN

### Phase 1: Setup & Baseline (Today)
1. ✅ Create branch `ray/novo-parity-experiments`
2. ✅ Create directory structure
3. ✅ Create this planning document
4. ⏸️ Write baseline script (Exp 01)
5. ⏸️ Begin web search validation

### Phase 2: Core Experiments (Next)
1. ⏸️ Implement Exp 02-05 (the main hypotheses)
2. ⏸️ Run inference on each
3. ⏸️ Compare confusion matrices
4. ⏸️ Document findings

### Phase 3: Alternative Hypotheses (If needed)
1. ⏸️ Implement Exp 06-10 based on Phase 2 insights
2. ⏸️ Run inference
3. ⏸️ Identify best match (if any)

### Phase 4: Documentation & Decision (Final)
1. ⏸️ Update `EXPERIMENTS_LOG.md` with all results
2. ⏸️ Choose primary test set for paper
3. ⏸️ Merge or document experimentation branch
4. ⏸️ Email Max with findings

---

## 📝 RUNNING LOG

### 2025-11-03 21:45 - Branch Created
- Created `ray/novo-parity-experiments` branch
- Set up directory structure
- Created master planning document
- Ready for experimentation

**Next**: Begin web search validation of the 7 reclassification candidates

---

## ✅ ACCEPTANCE CRITERIA

**Success is NOT matching Novo exactly!**

Success is:
1. ✅ Full traceability (every CSV has a script, every script has provenance)
2. ✅ Understanding the space of possibilities
3. ✅ Documenting what DOESN'T work (as valuable as what does)
4. ✅ Having defensible test sets for inference
5. ✅ Being transparent about methodology

**Primary test set remains**: 116 ELISA-only antibodies (method-faithful)
**Secondary exploration**: Which permutations get close to Novo's 59/27?

---

## 🎯 KEY PRINCIPLES

1. **No slop** - Every file is traceable to a script
2. **Reproducible** - Anyone can re-run the experiments
3. **Transparent** - Document what fails, not just what succeeds
4. **Iterative** - Update this plan as we learn from web search
5. **Principled** - No arbitrary choices without documented rationale

---

**This is a living document - update as experiments run!**
