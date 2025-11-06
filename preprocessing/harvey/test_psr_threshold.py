"""
Test Harvey Dataset with PSR Threshold

This tests our trained model on the Harvey dataset (141,021 nanobodies)
using the PSR-specific threshold (0.5495) discovered from Shehata analysis.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def print_metrics(y_true, y_pred, dataset_name):
    """Calculate and print performance metrics"""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{dataset_name} Results")
    print(f"{'='*60}")
    print(f"Confusion Matrix: {cm.tolist()}")
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print(f"\nAccuracy: {acc*100:.1f}%")
    print(f"Sensitivity: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
    print(f"Specificity: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%")
    print(f"{'='*60}\n")

    return {"cm": cm, "accuracy": acc}


def main():
    print("=" * 60)
    print("HARVEY DATASET TEST - PSR Threshold 0.5495")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load model
    model_path = "models/boughter_vh_esm1v_logreg.pkl"
    print(f"Loading model from {model_path}...")
    sys.stdout.flush()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Override batch_size for large-scale inference on MPS
    model.embedding_extractor.batch_size = 2

    print(f"✓ Model loaded (device: {model.device})")
    print(f"  Batch size: {model.embedding_extractor.batch_size}")
    print()

    # Novo benchmark
    novo_cm = np.array([[19778, 49962], [4186, 67633]])
    novo_acc = (19778 + 67633) / 141559

    print("Novo Benchmark (from Figure S14):")
    print(f"  Confusion Matrix: {novo_cm.tolist()}")
    print(f"  Accuracy: {novo_acc*100:.1f}%")
    print()

    # Load Harvey data
    harvey_file = "test_datasets/harvey/fragments/VHH_only_harvey.csv"
    print(f"Loading Harvey dataset from {harvey_file}...")
    sys.stdout.flush()

    df = pd.read_csv(harvey_file, comment="#")
    sequences = df["sequence"].tolist()
    y_true = df["label"].values

    print(f"  Dataset size: {len(sequences)} sequences")
    print(f"  Class distribution: {pd.Series(y_true).value_counts().to_dict()}")
    print()

    # Extract embeddings (this will take ~20-30 minutes)
    print("Extracting ESM-1v embeddings...")
    print("  (This will take ~20-30 minutes for 141k sequences)")
    sys.stdout.flush()

    start_embed = time.time()
    X_embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)
    embed_time = time.time() - start_embed

    print(f"✓ Embeddings extracted in {embed_time/60:.1f} minutes")
    print(f"  Embedding shape: {X_embeddings.shape}")
    print()

    # Run inference with PSR threshold
    print("Running inference with PSR threshold (0.5495)...")
    sys.stdout.flush()

    y_pred = model.predict(X_embeddings, assay_type="PSR")

    # Calculate metrics
    results = print_metrics(y_true, y_pred, "Harvey (PSR threshold=0.5495)")

    # Compare to Novo
    cm = results["cm"]
    diff = np.abs(cm - novo_cm).sum()

    print("\nComparison to Novo:")
    print(f"  Novo CM:  {novo_cm.tolist()}")
    print(f"  Our CM:   {cm.tolist()}")
    print(f"  Difference: {diff} (sum of absolute differences)")
    print(f"  Accuracy gap: {(results['accuracy'] - novo_acc)*100:.1f}pp")
    print()

    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("✓ Harvey test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
