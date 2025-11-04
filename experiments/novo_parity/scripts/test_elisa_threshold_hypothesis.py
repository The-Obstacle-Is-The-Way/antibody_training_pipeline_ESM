#!/usr/bin/env python3
"""
Test Hybri's ELISA Threshold Hypothesis vs P5e-S2 Method

Hybri's hypothesis: Novo used simple ELISA threshold (1.3 or 1.5) instead of PSR
Our method: P5e-S2 (PSR reclassification + PSR/AC-SINS removal)

This experiment tests both approaches and compares results against Novo's benchmark.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
MODEL_PATH = BASE_DIR / "models/boughter_vh_esm1v_logreg.pkl"
JAIN_116 = BASE_DIR / "test_datasets/jain_ELISA_ONLY_116.csv"
JAIN_SD03 = BASE_DIR / "test_datasets/jain_sd03.csv"
P5E_S2_DATASET = BASE_DIR / "test_datasets/jain/jain_86_novo_parity.csv"

# Novo benchmark
NOVO_CM = np.array([[40, 19], [10, 17]])
NOVO_ACC = 0.6628


def load_model():
    """Load trained model"""
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded (device: {model.device})")
    return model


def test_elisa_threshold(model, threshold=1.3):
    """Test simple ELISA threshold approach"""
    print(f"\n{'='*60}")
    print(f"TEST 1: Simple ELISA Threshold (<= {threshold})")
    print(f"{'='*60}")

    # Load 116 SSOT
    df116 = pd.read_csv(JAIN_116)
    sd03 = pd.read_csv(JAIN_SD03)

    # Merge to get continuous ELISA values
    merged = df116.merge(sd03[['Name', 'ELISA']], left_on='id', right_on='Name', how='left')

    # Apply threshold
    df_86 = merged[merged['ELISA'] <= threshold].copy()

    print(f"\n116 → ELISA <= {threshold} → {len(df_86)} antibodies")
    print(f"  Specific: {(df_86['label']==0).sum()}")
    print(f"  Non-specific: {(df_86['label']==1).sum()}")

    # Generate embeddings
    print("\nGenerating embeddings...")
    sequences = df_86['vh_sequence'].tolist()
    X = model.embedding_extractor.extract_batch_embeddings(sequences)

    # Predict
    y_true = df_86['label'].values
    y_pred = model.predict(X)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"\nResults:")
    print(f"  Confusion Matrix: {cm.tolist()}")
    print(f"  Accuracy: {acc*100:.2f}%")

    print(f"\nComparison to Novo:")
    print(f"  Novo CM:  {NOVO_CM.tolist()}")
    print(f"  Ours:     {cm.tolist()}")
    print(f"  Match: {'✅ YES' if np.array_equal(cm, NOVO_CM) else '❌ NO'}")

    return {"method": f"ELISA<={threshold}", "cm": cm, "acc": acc, "n": len(df_86)}


def test_p5e_s2(model):
    """Test P5e-S2 approach"""
    print(f"\n{'='*60}")
    print(f"TEST 2: P5e-S2 Method (PSR Reclassification + Removal)")
    print(f"{'='*60}")

    # Load P5e-S2 dataset
    df = pd.read_csv(P5E_S2_DATASET)

    print(f"\n116 → P5e-S2 pipeline → {len(df)} antibodies")
    print(f"  Specific: {(df['label']==0).sum()}")
    print(f"  Non-specific: {(df['label']==1).sum()}")

    # Generate embeddings
    print("\nGenerating embeddings...")
    sequences = df['vh_sequence'].tolist()
    X = model.embedding_extractor.extract_batch_embeddings(sequences)

    # Predict
    y_true = df['label'].values
    y_pred = model.predict(X)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"\nResults:")
    print(f"  Confusion Matrix: {cm.tolist()}")
    print(f"  Accuracy: {acc*100:.2f}%")

    print(f"\nComparison to Novo:")
    print(f"  Novo CM:  {NOVO_CM.tolist()}")
    print(f"  Ours:     {cm.tolist()}")
    print(f"  Match: {'✅ YES' if np.array_equal(cm, NOVO_CM) else '❌ NO'}")

    return {"method": "P5e-S2", "cm": cm, "acc": acc, "n": len(df)}


def main():
    print("="*60)
    print("ELISA THRESHOLD HYPOTHESIS TEST")
    print("="*60)
    print("\nHypothesis (Hybri): Novo used simple ELISA threshold (1.3-1.5)")
    print("Counter (Ours): Novo used P5e-S2 (PSR reclassification + removal)\n")

    # Load model
    model = load_model()

    # Test both approaches
    result_elisa_13 = test_elisa_threshold(model, threshold=1.3)
    result_elisa_15 = test_elisa_threshold(model, threshold=1.5)
    result_p5e = test_p5e_s2(model)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print("| Method | N | Accuracy | CM Match | Confusion Matrix |")
    print("|--------|---|----------|----------|------------------|")

    for result in [result_elisa_13, result_elisa_15, result_p5e]:
        match = "✅" if np.array_equal(result['cm'], NOVO_CM) else "❌"
        print(f"| {result['method']:12s} | {result['n']:2d} | {result['acc']*100:5.2f}% | {match:8s} | {result['cm'].tolist()} |")

    print(f"\nNovo benchmark: {NOVO_CM.tolist()}, {NOVO_ACC*100:.2f}%")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}\n")

    if np.array_equal(result_p5e['cm'], NOVO_CM):
        print("✅ P5e-S2 achieves EXACT Novo parity")
        print("❌ Simple ELISA thresholds do NOT match")
        print("\nHybri's hypothesis: REJECTED")
        print("Reason: Simple threshold gives wrong label distribution")
        print("        (all specific antibodies, no reclassification)")
    else:
        print("⚠️ Neither method achieves exact parity")
        print("Need further investigation")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
