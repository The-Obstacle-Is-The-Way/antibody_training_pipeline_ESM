# Systematic Investigation Plan - Post-Hyperparameter Sweep

**Date**: 2025-11-02
**Status**: 📋 **READY TO EXECUTE**
**Context**: Hyperparameter tuning exhausted - no improvement beyond 67.5% on Boughter 10-CV

---

## Current Situation

### What We Know

**Boughter 10-CV Performance (stuck at ceiling)**:
```
Tested 16+ hyperparameter configurations:
  • C: 0.001 to 100
  • Penalties: L1, L2
  • Solvers: lbfgs, liblinear, saga
  • class_weight: 'balanced' and None

Best result: 67.39% ± 4.89% (C=0.01, L2, None)
Baseline: 67.50% ± 8.90% (C=1.0, L2, 'balanced')
Target: 71.0% (Novo benchmark)

GAP: -3.5% to -3.6% (cannot close via hyperparameters!)
```

**Jain Test Performance (even worse)**:
```
Current: 55.3% accuracy
Target: 68.6% (Novo benchmark)
GAP: -13.3% total

Breakdown:
  • -3.5% = Boughter training gap (can't fix with hyperparams)
  • -9.8% = Generalization gap (MAJOR issue)
```

---

## Investigation Areas (Priority Order)

### **PRIORITY 1: Jain Test Evaluation Issues** 🚨

**Hypothesis**: The 9.8% generalization gap may be partly due to **testing methodology issues**, not just model quality.

#### Issues to Check:

1. **Threshold Mismatch**:
   ```python
   Problem:
     • Boughter: 48.5% / 51.5% (balanced) → default threshold 0.5 OK
     • Jain: 71.3% / 28.7% (IMBALANCED) → threshold 0.5 may be wrong!

   Test:
     • Check if Novo tuned threshold on Jain
     • Try threshold optimization (0.3 - 0.7 range)
     • Use balanced accuracy instead of raw accuracy
   ```

2. **Metric Computation for Imbalanced Data**:
   ```python
   Current test.py uses:
     • accuracy_score (can be misleading on imbalanced data)
     • Default threshold (0.5)

   Should check:
     • Per-class precision/recall
     • Balanced accuracy
     • F1 per class
     • Confusion matrix ratios
   ```

3. **Preprocessing Differences (Boughter vs Jain)**:
   ```python
   Question: Do we preprocess Jain the SAME way as Boughter?

   Check:
     • Sequence filtering
     • CDR annotation
     • V-domain extraction
     • Gap handling
   ```

#### Action Items:

- [ ] **Check test.py evaluation**: Are we using default threshold 0.5?
- [ ] **Threshold tuning**: Test range 0.3-0.7 on Jain
- [ ] **Compare confusion matrices**: Ours vs Novo's (accounting for size difference)
- [ ] **Verify Jain preprocessing**: Compare to Boughter pipeline
- [ ] **Check embedding extraction**: Same settings for both datasets?

**Estimated Impact**: Could recover 3-5% of the 9.8% gap

---

### **PRIORITY 2: Boughter Training Data Verification**

**Hypothesis**: Our Boughter dataset may differ from Novo's.

#### Issues to Check:

1. **Dataset Size**:
   ```
   Ours: 914 antibodies (443 specific / 471 non-specific)
   Novo's: ??? (not specified in paper)

   Question: Are we using the same training set?
   ```

2. **Filtering Criteria**:
   ```
   Did Novo exclude:
     • Mild non-specific (1-3 flags)?  ✓ Yes (we do this)
     • Any QC filtering?  ??? (unknown)
     • Specific antibody families?  ??? (unknown)
   ```

3. **Preprocessing Pipeline**:
   ```
   Check:
     • V-domain extraction method
     • CDR annotation (IMGT/ANARCI)
     • Gap character handling
     • Sequence sanitization
   ```

#### Action Items:

- [ ] **Count Boughter antibodies**: Verify 914 is correct
- [ ] **Check filtering**: Compare to original Boughter paper
- [ ] **Verify preprocessing**: Document our pipeline
- [ ] **Compare to literature**: Check if others report ~914 antibodies

**Estimated Impact**: Could recover 1-2% of the 3.5% training gap

---

### **PRIORITY 3: Embedding Extraction Verification**

**Hypothesis**: ESM embedding extraction may differ from Novo's.

#### Issues to Check:

1. **ESM Model Version**:
   ```python
   Ours: facebook/esm1v_t33_650M_UR90S_1 (HuggingFace)
   Novo's: "ESM-1v" (exact checkpoint unclear)

   Question: Same model weights?
   ```

2. **Pooling Strategy**:
   ```python
   Our implementation (model.py:58-62):
     # Masked mean pooling
     masked_embeddings = embeddings * attention_mask
     sum_embeddings = masked_embeddings.sum(dim=1)
     sum_mask = attention_mask.sum(dim=1)
     mean_embeddings = sum_embeddings / sum_mask

   Novo's: "mean-mode" (not fully specified)

   Questions:
     • Do they exclude CLS/EOS tokens? (we do)
     • Do they use attention mask? (we do)
     • Same numerical precision?
   ```

3. **Batch Size Effects**:
   ```python
   Ours: batch_size=8
   Novo's: ??? (not specified)

   Question: Does batch size affect embeddings?
   (Shouldn't, but worth checking)
   ```

#### Action Items:

- [ ] **Extract embeddings twice**: Check reproducibility
- [ ] **Compare embedding distributions**: Boughter vs Jain
- [ ] **Check for NaN/Inf**: Verify no numerical issues
- [ ] **Test different batch sizes**: Verify consistency

**Estimated Impact**: Could recover 1-2% if there's a numerical issue

---

### **PRIORITY 4: Alternative Model Architectures**

**Hypothesis**: LogisticRegression may not be optimal (though Novo used it successfully).

#### Options to Test:

1. **Linear models with better regularization**:
   ```python
   • ElasticNet (L1 + L2 combo)
   • SGDClassifier with log loss
   • PassiveAggressiveClassifier
   ```

2. **Shallow non-linear models**:
   ```python
   • Random Forest (Novo tested this, got ~69%)
   • XGBoost
   • LightGBM
   ```

3. **Simple neural networks**:
   ```python
   • MLP (1-2 hidden layers)
   • SVM with RBF kernel
   ```

#### Action Items:

- [ ] **Test Random Forest**: Novo reported similar performance
- [ ] **Test XGBoost**: Often works well on embeddings
- [ ] **Compare**: If non-linear helps, suggests feature interactions matter

**Estimated Impact**: Low priority - only if other fixes don't work

---

## Execution Plan

### Phase 1: Quick Wins (1-2 hours)

1. **Check test.py threshold**: Is it hardcoded to 0.5?
2. **Tune threshold on Jain**: Test 0.3, 0.4, 0.5, 0.6, 0.7
3. **Compare confusion matrices**: Ours vs Novo's ratios

**If successful**: Could immediately improve Jain test accuracy

---

### Phase 2: Data Verification (2-3 hours)

1. **Verify Boughter dataset**: Count antibodies, check filtering
2. **Compare preprocessing**: Document Boughter vs Jain pipelines
3. **Check embedding extraction**: Verify reproducibility

**If successful**: Could identify data issues causing gaps

---

### Phase 3: Deep Investigation (1-2 days)

1. **Contact Novo authors**: Request exact implementation details
2. **Test alternative models**: Random Forest, XGBoost
3. **Cross-dataset analysis**: Compare embedding distributions

**If successful**: Could close remaining gaps or confirm ceiling

---

## Success Criteria

### Minimum Success:
- [ ] Understand why we're stuck at 67.5% on Boughter
- [ ] Identify cause of 9.8% generalization gap
- [ ] Document discrepancies with Novo's methodology

### Full Success:
- [ ] Achieve ≥70% on Boughter 10-CV
- [ ] Achieve ≥65% on Jain test (close 5% of the 13.3% gap)
- [ ] Match Novo's methodology exactly

### Stretch Success:
- [ ] Match Novo's 71% on Boughter
- [ ] Match Novo's 68.6% on Jain
- [ ] Publish replication study documenting findings

---

## Key Questions to Answer

1. **Is our 67.5% on Boughter a real ceiling, or are we missing something?**
   - If ceiling: Need to investigate dataset/preprocessing
   - If missing something: Find the bug/difference

2. **Why do we drop 12.2% from CV to Jain test?**
   - Novo only drops 2.4% (71% → 68.6%)
   - This is a MAJOR red flag!

3. **Are we testing Jain correctly?**
   - Threshold tuning?
   - Metric computation?
   - Preprocessing alignment?

4. **Is there a systematic bug in our pipeline?**
   - Data loading?
   - Embedding extraction?
   - Model evaluation?

---

## Documentation to Create

After each phase:

1. **Testing Methodology Analysis** (`JAIN_TESTING_ANALYSIS.md`)
   - Threshold tuning results
   - Confusion matrix comparison
   - Metric verification

2. **Data Verification Report** (`BOUGHTER_DATA_VERIFICATION.md`)
   - Dataset size/composition
   - Filtering criteria
   - Preprocessing pipeline

3. **Embedding Analysis** (`EMBEDDING_VERIFICATION.md`)
   - Reproducibility tests
   - Distribution comparisons
   - Numerical stability

4. **Final Replication Report** (`NOVO_REPLICATION_FINAL.md`)
   - What we matched
   - What we couldn't match
   - Lessons learned

---

## Next Immediate Action

**START HERE**:

1. Check test.py: How are we computing metrics on Jain?
2. Verify threshold: Are we using default 0.5?
3. Test threshold tuning: Can we improve 55.3% → 60%+ on Jain?

**If threshold tuning helps**: Strong evidence of evaluation methodology issue!
**If threshold tuning doesn't help**: Move to data/preprocessing verification.

---

**Status**: Ready to execute Phase 1
**Estimated total time**: 3-5 hours for Phase 1+2, then reassess
**Priority**: P0 - Blocking accurate Novo replication
