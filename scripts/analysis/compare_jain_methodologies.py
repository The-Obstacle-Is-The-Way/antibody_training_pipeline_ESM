#!/usr/bin/env python3
"""
Compare Jain Dataset Methodologies: OLD (94→91→86) vs P5e-S2 (137→116→86)

This script performs a detailed comparison of the two competing Jain datasets:
1. VH_only_jain_test_PARITY_86.csv (OLD reverse-engineered method)
2. VH_only_jain_86_p5e_s2.csv (P5e-S2 canonical method)

Output:
- Antibody composition differences
- Label differences
- Model prediction differences
- Confusion matrix comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Paths
OLD_FILE = Path("test_datasets/jain/VH_only_jain_test_PARITY_86.csv")
P5E_S2_FILE = Path("test_datasets/jain/VH_only_jain_86_p5e_s2.csv")
MODEL_FILE = Path("models/boughter_vh_esm1v_logreg.pkl")
OLD_PRED_FILE = Path("test_results/jain_old_verification/predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251105_083012.csv")
P5E_S2_PRED_FILE = Path("test_results/jain_p5e_s2_verification/predictions_boughter_vh_esm1v_logreg_VH_only_jain_86_p5e_s2_20251105_082929.csv")

def load_datasets():
    """Load both datasets"""
    old_df = pd.read_csv(OLD_FILE)
    p5e_s2_df = pd.read_csv(P5E_S2_FILE)

    return old_df, p5e_s2_df

def compare_antibody_composition(old_df, p5e_s2_df):
    """Compare which antibodies are in each dataset"""
    old_ids = set(old_df['id'].tolist())
    p5e_s2_ids = set(p5e_s2_df['id'].tolist())

    only_old = old_ids - p5e_s2_ids
    only_p5e_s2 = p5e_s2_ids - old_ids
    shared = old_ids & p5e_s2_ids

    print("="*80)
    print("ANTIBODY COMPOSITION COMPARISON")
    print("="*80)
    print(f"\nOLD dataset:    {len(old_ids)} antibodies")
    print(f"P5e-S2 dataset: {len(p5e_s2_ids)} antibodies")
    print(f"Shared:         {len(shared)} antibodies ({len(shared)/86*100:.1f}%)")
    print(f"Different:      {len(only_old) + len(only_p5e_s2)} antibodies ({(len(only_old) + len(only_p5e_s2))/86*100:.1f}%)")

    print(f"\n{'='*80}")
    print(f"ANTIBODIES IN OLD BUT NOT IN P5e-S2 ({len(only_old)} total)")
    print(f"{'='*80}")
    for ab in sorted(only_old):
        old_label = old_df[old_df['id'] == ab]['label'].values[0]
        print(f"  {ab:25s} label={old_label}")

    print(f"\n{'='*80}")
    print(f"ANTIBODIES IN P5e-S2 BUT NOT IN OLD ({len(only_p5e_s2)} total)")
    print(f"{'='*80}")
    for ab in sorted(only_p5e_s2):
        p5e_label = p5e_s2_df[p5e_s2_df['id'] == ab]['label'].values[0]
        print(f"  {ab:25s} label={p5e_label}")

    return only_old, only_p5e_s2, shared

def compare_labels(old_df, p5e_s2_df, shared):
    """Compare labels for shared antibodies"""
    print(f"\n{'='*80}")
    print("LABEL COMPARISON FOR SHARED ANTIBODIES")
    print(f"{'='*80}")

    label_diffs = []
    for ab in shared:
        old_label = old_df[old_df['id'] == ab]['label'].values[0]
        p5e_label = p5e_s2_df[p5e_s2_df['id'] == ab]['label'].values[0]

        if old_label != p5e_label:
            label_diffs.append((ab, old_label, p5e_label))

    if label_diffs:
        print(f"\n⚠️  FOUND {len(label_diffs)} LABEL DIFFERENCES:")
        for ab, old_label, p5e_label in label_diffs:
            print(f"  {ab:25s} OLD={old_label} → P5e-S2={p5e_label}")
    else:
        print(f"\n✅ All shared antibodies have SAME labels")

    # Label distribution
    print(f"\n{'='*80}")
    print("LABEL DISTRIBUTION")
    print(f"{'='*80}")

    old_dist = old_df['label'].value_counts().sort_index()
    p5e_s2_dist = p5e_s2_df['label'].value_counts().sort_index()

    print(f"\nOLD dataset:")
    print(f"  Specific (0):     {old_dist.get(0.0, 0)} antibodies")
    print(f"  Non-specific (1): {old_dist.get(1.0, 0)} antibodies")

    print(f"\nP5e-S2 dataset:")
    print(f"  Specific (0):     {p5e_s2_dist.get(0, 0)} antibodies")
    print(f"  Non-specific (1): {p5e_s2_dist.get(1, 0)} antibodies")

    return label_diffs

def load_predictions():
    """Load model predictions"""
    old_pred = pd.read_csv(OLD_PRED_FILE)
    p5e_s2_pred = pd.read_csv(P5E_S2_PRED_FILE)

    return old_pred, p5e_s2_pred

def compare_confusion_matrices(old_pred, p5e_s2_pred):
    """Compare confusion matrices"""
    from sklearn.metrics import confusion_matrix

    print(f"\n{'='*80}")
    print("CONFUSION MATRIX COMPARISON")
    print(f"{'='*80}")

    # OLD dataset
    old_cm = confusion_matrix(old_pred['y_true'], old_pred['y_pred'])
    print(f"\nOLD dataset (VH_only_jain_test_PARITY_86.csv):")
    print(f"  Confusion Matrix: {old_cm.tolist()}")
    print(f"  TN={old_cm[0,0]}, FP={old_cm[0,1]}, FN={old_cm[1,0]}, TP={old_cm[1,1]}")
    print(f"  Accuracy: {(old_cm[0,0] + old_cm[1,1]) / old_cm.sum():.2%} ({old_cm[0,0] + old_cm[1,1]}/{old_cm.sum()})")

    # P5e-S2 dataset
    p5e_s2_cm = confusion_matrix(p5e_s2_pred['y_true'], p5e_s2_pred['y_pred'])
    print(f"\nP5e-S2 dataset (VH_only_jain_86_p5e_s2.csv):")
    print(f"  Confusion Matrix: {p5e_s2_cm.tolist()}")
    print(f"  TN={p5e_s2_cm[0,0]}, FP={p5e_s2_cm[0,1]}, FN={p5e_s2_cm[1,0]}, TP={p5e_s2_cm[1,1]}")
    print(f"  Accuracy: {(p5e_s2_cm[0,0] + p5e_s2_cm[1,1]) / p5e_s2_cm.sum():.2%} ({p5e_s2_cm[0,0] + p5e_s2_cm[1,1]}/{p5e_s2_cm.sum()})")

    # Novo parity
    novo_cm = [[40, 19], [10, 17]]
    print(f"\nNovo published (Sakhnini et al. 2025):")
    print(f"  Confusion Matrix: {novo_cm}")
    print(f"  TN={novo_cm[0][0]}, FP={novo_cm[0][1]}, FN={novo_cm[1][0]}, TP={novo_cm[1][1]}")
    print(f"  Accuracy: {(novo_cm[0][0] + novo_cm[1][1]) / 86:.2%} ({novo_cm[0][0] + novo_cm[1][1]}/86)")

    # Check which matches Novo
    print(f"\n{'='*80}")
    print("NOVO PARITY CHECK")
    print(f"{'='*80}")

    old_matches = (old_cm == novo_cm).all()
    p5e_matches = (p5e_s2_cm == novo_cm).all()

    if old_matches:
        print(f"✅ OLD dataset: EXACT MATCH to Novo!")
    else:
        print(f"❌ OLD dataset: Does NOT match Novo")

    if p5e_matches:
        print(f"✅ P5e-S2 dataset: EXACT MATCH to Novo!")
    else:
        print(f"❌ P5e-S2 dataset: Does NOT match Novo")

    return old_cm, p5e_s2_cm

def compare_predictions_detail(old_df, p5e_s2_df, old_pred, p5e_s2_pred, shared):
    """Detailed comparison of predictions on shared antibodies"""
    print(f"\n{'='*80}")
    print("PREDICTION DIFFERENCES ON SHARED ANTIBODIES")
    print(f"{'='*80}")

    # Merge predictions with IDs
    old_df_merged = old_df.copy()
    old_df_merged['y_pred_old'] = old_pred['y_pred'].values
    old_df_merged['y_proba_old'] = old_pred['y_proba'].values

    p5e_s2_df_merged = p5e_s2_df.copy()
    p5e_s2_df_merged['y_pred_p5e'] = p5e_s2_pred['y_pred'].values
    p5e_s2_df_merged['y_proba_p5e'] = p5e_s2_pred['y_proba'].values

    # Find shared antibodies with different predictions
    pred_diffs = []
    for ab in shared:
        old_row = old_df_merged[old_df_merged['id'] == ab].iloc[0]
        p5e_row = p5e_s2_df_merged[p5e_s2_df_merged['id'] == ab].iloc[0]

        if old_row['y_pred_old'] != p5e_row['y_pred_p5e']:
            pred_diffs.append({
                'antibody': ab,
                'true_label': old_row['label'],  # Should be same in both
                'old_pred': old_row['y_pred_old'],
                'p5e_pred': p5e_row['y_pred_p5e'],
                'old_proba': old_row['y_proba_old'],
                'p5e_proba': p5e_row['y_proba_p5e'],
            })

    if pred_diffs:
        print(f"\n⚠️  FOUND {len(pred_diffs)} SHARED ANTIBODIES WITH DIFFERENT PREDICTIONS:")
        print(f"\n{'Antibody':<25s} {'True':>5s} {'OLD':>5s} {'P5e':>5s} {'OLD_prob':>8s} {'P5e_prob':>8s}")
        print("-"*80)
        for diff in pred_diffs:
            print(f"{diff['antibody']:<25s} {int(diff['true_label']):>5d} "
                  f"{int(diff['old_pred']):>5d} {int(diff['p5e_pred']):>5d} "
                  f"{diff['old_proba']:>8.4f} {diff['p5e_proba']:>8.4f}")
    else:
        print(f"\n✅ All shared antibodies have SAME predictions")

    return pred_diffs

def main():
    """Main comparison function"""
    print("="*80)
    print("JAIN DATASET METHODOLOGY COMPARISON")
    print("OLD (94→91→86) vs P5e-S2 (137→116→86)")
    print("="*80)

    # Load datasets
    old_df, p5e_s2_df = load_datasets()

    # Compare antibody composition
    only_old, only_p5e_s2, shared = compare_antibody_composition(old_df, p5e_s2_df)

    # Compare labels
    label_diffs = compare_labels(old_df, p5e_s2_df, shared)

    # Load predictions
    old_pred, p5e_s2_pred = load_predictions()

    # Compare confusion matrices
    old_cm, p5e_s2_cm = compare_confusion_matrices(old_pred, p5e_s2_pred)

    # Compare predictions on shared antibodies
    pred_diffs = compare_predictions_detail(old_df, p5e_s2_df, old_pred, p5e_s2_pred, shared)

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n1. Antibody Composition:")
    print(f"   - Shared: {len(shared)}/86 ({len(shared)/86*100:.1f}%)")
    print(f"   - Only in OLD: {len(only_old)}")
    print(f"   - Only in P5e-S2: {len(only_p5e_s2)}")

    print(f"\n2. Labels:")
    print(f"   - Differences: {len(label_diffs) if label_diffs else 0}")

    print(f"\n3. Model Predictions:")
    print(f"   - OLD accuracy: {(old_cm[0,0] + old_cm[1,1]) / old_cm.sum():.2%}")
    print(f"   - P5e-S2 accuracy: {(p5e_s2_cm[0,0] + p5e_s2_cm[1,1]) / p5e_s2_cm.sum():.2%}")
    print(f"   - Prediction differences on shared: {len(pred_diffs) if pred_diffs else 0}")

    print(f"\n4. Novo Parity:")
    novo_cm = [[40, 19], [10, 17]]
    old_matches = (old_cm == novo_cm).all()
    p5e_matches = (p5e_s2_cm == novo_cm).all()
    print(f"   - OLD: {'✅ EXACT MATCH' if old_matches else '❌ NO MATCH'}")
    print(f"   - P5e-S2: {'✅ EXACT MATCH' if p5e_matches else '❌ NO MATCH'}")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    if old_matches and not p5e_matches:
        print("\n🚨 CRITICAL: OLD dataset achieves Novo parity, P5e-S2 does NOT!")
        print("   The 28% antibody difference affects model performance.")
        print("   Need to investigate which methodology is actually correct.")
    elif p5e_matches and not old_matches:
        print("\n✅ P5e-S2 dataset achieves Novo parity, OLD does not.")
        print("   P5e-S2 is the correct methodology.")
    elif old_matches and p5e_matches:
        print("\n🤔 BOTH datasets achieve Novo parity despite different antibodies.")
        print("   This is a remarkable coincidence!")
    else:
        print("\n❌ NEITHER dataset achieves Novo parity.")
        print("   Further investigation needed.")

    print(f"\nGenerated: {pd.Timestamp.now()}")

if __name__ == "__main__":
    main()
