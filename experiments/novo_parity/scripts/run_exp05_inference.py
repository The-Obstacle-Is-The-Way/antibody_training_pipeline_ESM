#!/usr/bin/env python3
"""
Run Inference on Experiment 05 Dataset (Jain 86 PSR-Hybrid Parity)

Compare our model's predictions on the exp05 dataset against Novo's reported results:
- Target: 66.28% accuracy (57/86 correct)
- Target confusion matrix: [[40, 19], [10, 17]]

Usage:
    python experiments/novo_parity/scripts/run_exp05_inference.py
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add parent directory to path for imports
# Go up 3 levels: scripts/ -> novo_parity/ -> experiments/ -> repo_root/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

def main():
    print("="*80)
    print("EXPERIMENT 05: INFERENCE ON PSR-HYBRID PARITY DATASET")
    print("="*80)
    print()

    # Paths
    model_path = REPO_ROOT / 'models' / 'boughter_vh_esm1v_logreg.pkl'
    dataset_path = REPO_ROOT / 'experiments' / 'novo_parity' / 'datasets' / 'jain_86_exp05.csv'

    # Load the trained model
    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    # Verify model configuration
    has_scaler = hasattr(classifier, 'scaler') and classifier.scaler is not None
    print(f"✅ Model loaded successfully")
    print(f"   - Has StandardScaler: {has_scaler} (should be False)")
    print(f"   - Classifier: {classifier.classifier.__class__.__name__}")
    print()

    # Load Experiment 05 dataset (86 antibodies)
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {len(df)} antibodies")
    print(f"   - Specific (label=0): {(df['label']==0).sum()}")
    print(f"   - Non-specific (label=1): {(df['label']==1).sum()}")
    print()

    # Extract sequences and labels
    sequences = df['vh_sequence'].tolist()  # VH heavy chain sequences
    y_true = df['label'].values
    antibody_ids = df['id'].tolist()

    # Generate embeddings
    print("Generating ESM-1v embeddings...")
    X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)
    print(f"✅ Embeddings generated: shape {X_test.shape}")
    print()

    # Make predictions
    print("Making predictions...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    print(f"✅ Predictions complete")
    print()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Display results
    print("="*80)
    print("RESULTS: EXPERIMENT 05 INFERENCE")
    print("="*80)
    print()

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions: {(y_true == y_pred).sum()}/{len(y_true)}")
    print()

    print("OUR Confusion Matrix:")
    print("              Predicted")
    print("              Specific(0) Non-spec(1)   Total")
    print(f"Actual Specific(0):     {cm[0,0]:2d}         {cm[0,1]:2d}        {cm[0,0]+cm[0,1]:2d}")
    print(f"Actual Non-spec(1):     {cm[1,0]:2d}         {cm[1,1]:2d}        {cm[1,0]+cm[1,1]:2d}")
    print(f"                       ---        ---       ---")
    print(f"Total:                  {cm[:,0].sum():2d}         {cm[:,1].sum():2d}        {len(y_true):2d}")
    print()

    print("NOVO Confusion Matrix (Target):")
    print("              Predicted")
    print("              Specific(0) Non-spec(1)   Total")
    print("Actual Specific(0):     40         19        59")
    print("Actual Non-spec(1):     10         17        27")
    print("                       ---        ---       ---")
    print("Total:                  50         36        86")
    print()

    # Check for exact match
    novo_cm = np.array([[40, 19], [10, 17]])
    novo_accuracy = 57/86

    match_cm = np.array_equal(cm, novo_cm)
    match_acc = abs(accuracy - novo_accuracy) < 0.0001

    if match_cm:
        print("✅✅✅ PERFECT MATCH! Confusion matrix is IDENTICAL to Novo!")
    else:
        print("⚠️ Confusion matrix differs from Novo:")
        diff = cm - novo_cm
        print(f"   Difference matrix:")
        print(f"   {diff}")
        print()
        print(f"   Cell-by-cell:")
        print(f"   - TN (spec→spec):  Ours={cm[0,0]:2d}, Novo=40, Diff={cm[0,0]-40:+d}")
        print(f"   - FP (spec→nonspec): Ours={cm[0,1]:2d}, Novo=19, Diff={cm[0,1]-19:+d}")
        print(f"   - FN (nonspec→spec): Ours={cm[1,0]:2d}, Novo=10, Diff={cm[1,0]-10:+d}")
        print(f"   - TP (nonspec→nonspec): Ours={cm[1,1]:2d}, Novo=17, Diff={cm[1,1]-17:+d}")

    print()
    if match_acc:
        print(f"✅✅✅ PERFECT MATCH! Accuracy is IDENTICAL to Novo!")
    else:
        print(f"⚠️ Accuracy differs: Ours={accuracy:.4f} ({accuracy*100:.2f}%), Novo={novo_accuracy:.4f} ({novo_accuracy*100:.2f}%)")
        print(f"   Difference: {(accuracy-novo_accuracy)*100:+.2f} percentage points")

    print()
    print("="*80)
    print("DETAILED METRICS")
    print("="*80)
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Specific', 'Non-specific']))

    # Compare with Novo
    print()
    print("="*80)
    print("COMPARISON WITH NOVO NORDISK BENCHMARK")
    print("="*80)
    print()
    print(f"| Metric              | Ours         | Novo         | Match      |")
    print(f"|---------------------|--------------|--------------|------------|")
    print(f"| Accuracy            | {accuracy:.4f}       | 0.6628       | {'✅ YES' if match_acc else '❌ NO'} |")
    print(f"| Confusion Matrix    | [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]] | [[40,19],[10,17]] | {'✅ YES' if match_cm else '❌ NO'} |")
    print(f"| Spec TN             | {cm[0,0]:2d}           | 40           | {'✅ YES' if cm[0,0]==40 else '❌ NO'} |")
    print(f"| Spec FP             | {cm[0,1]:2d}           | 19           | {'✅ YES' if cm[0,1]==19 else '❌ NO'} |")
    print(f"| Non-spec FN         | {cm[1,0]:2d}           | 10           | {'✅ YES' if cm[1,0]==10 else '❌ NO'} |")
    print(f"| Non-spec TP         | {cm[1,1]:2d}           | 17           | {'✅ YES' if cm[1,1]==17 else '❌ NO'} |")
    print()

    # Analyze misclassifications
    print("="*80)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*80)
    print()

    # False Positives (specific predicted as non-specific)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_ids = [antibody_ids[i] for i in range(len(y_true)) if fp_mask[i]]
    print(f"False Positives (Specific → Non-specific): {len(fp_ids)}")
    if fp_ids:
        print("  ", ", ".join(fp_ids[:10]))
        if len(fp_ids) > 10:
            print(f"   ... and {len(fp_ids)-10} more")
    print()

    # False Negatives (non-specific predicted as specific)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_ids = [antibody_ids[i] for i in range(len(y_true)) if fn_mask[i]]
    print(f"False Negatives (Non-specific → Specific): {len(fn_ids)}")
    if fn_ids:
        print("  ", ", ".join(fn_ids))
    print()

    print("="*80)
    print("EXPERIMENT 05 DATASET PROVENANCE")
    print("="*80)
    print()
    print("Start: 116 antibodies (94 specific / 22 non-specific)")
    print("  ↓")
    print("Step 1: Reclassify 5 specific → non-specific")
    print("  - Tier A (PSR >0.4): bimagrumab, bavituximab, ganitumab, olaratumab")
    print("  - Tier B (Clinical): infliximab (61% ADA)")
    print("  ↓")
    print("Result: 89 specific / 27 non-specific")
    print("  ↓")
    print("Step 2: Remove top 30 specific by composite risk")
    print("  - Risk = PSR_norm + AC-SINS_norm + HIC_norm + (1 - Tm_norm)")
    print("  ↓")
    print("Final: 59 specific / 27 non-specific = 86 total ✅")
    print()

    print("="*80)

    if match_cm and match_acc:
        print("🎉 SUCCESS! EXACT NOVO PARITY ACHIEVED WITH EXPERIMENT 05! 🎉")
    else:
        print("⚠️ Parity not achieved - but we have transparent provenance")
        print()
        print("Next steps:")
        print("1. Analyze misclassified antibodies (PSR, AC-SINS, clinical data)")
        print("2. Try sensitivity analyses (PSR threshold, risk weights)")
        print("3. Document findings in NOVO_PARITY_EXPERIMENTS.md")

    print("="*80)

if __name__ == "__main__":
    main()
