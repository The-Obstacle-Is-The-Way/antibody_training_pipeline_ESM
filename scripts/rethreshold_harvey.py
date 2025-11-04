#!/usr/bin/env python3
"""
Re-threshold Harvey predictions with custom threshold (0.5495 vs 0.5).

This script loads the trained model, re-runs predictions on Harvey dataset,
and compares results with different probability thresholds.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Add parent directory to path for classifier import
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path: str):
    """Load trained model."""
    print(f"Loading model: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_test_data(data_path: str):
    """Load test dataset."""
    print(f"Loading test data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Total samples: {len(df)}")
    return df


def evaluate_with_threshold(y_true, y_proba, threshold: float):
    """Evaluate predictions with custom threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, cm, y_pred


def main():
    # Paths
    model_path = "models/boughter_vh_esm1v_logreg.pkl"
    data_path = "test_datasets/harvey/VHH_only_harvey.csv"

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    if not Path(data_path).exists():
        print(f"Error: Test data not found at {data_path}")
        sys.exit(1)

    # Load model and data
    model = load_model(model_path)
    df = load_test_data(data_path)

    # Extract features (embeddings) and labels
    print("\nExtracting embeddings...")
    X = np.array(df["embedding"].apply(eval).tolist())
    y_true = df["label"].values

    print(f"  Embedding shape: {X.shape}")
    print(f"  True labels: {np.bincount(y_true)}")

    # Get probability predictions
    print("\nGenerating probability predictions...")
    y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (non-specific)

    print(f"  Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}")
    print(f"  Probability mean: {y_proba.mean():.4f}")
    print(f"  Probability median: {np.median(y_proba):.4f}")

    # Test different thresholds
    thresholds = [0.5, 0.5495]

    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    results = {}
    for threshold in thresholds:
        print(f"\n--- Threshold = {threshold} ---")
        accuracy, cm, y_pred = evaluate_with_threshold(y_true, y_proba, threshold)

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )

        # Store results
        results[threshold] = {
            "accuracy": accuracy,
            "cm": cm,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

        # Print results
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(f"[[{tn:5d}, {fp:5d}],")
        print(f" [{fn:5d}, {tp:5d}]]")
        print("\n                Predicted")
        print("                Spec    Non-spec")
        print(f"True    Spec    {tn:5d}     {fp:5d}      ({tn+fp} specific)")
        print(f"        Non-spec {fn:4d}     {tp:5d}      ({fn+tp} non-specific)")
        print(f"\nSensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")

    # Compare thresholds
    print("\n" + "=" * 80)
    print("COMPARISON: 0.5 vs 0.5495")
    print("=" * 80)

    r1 = results[0.5]
    r2 = results[0.5495]

    acc_diff = (r2["accuracy"] - r1["accuracy"]) * 100
    sens_diff = (r2["sensitivity"] - r1["sensitivity"]) * 100
    spec_diff = (r2["specificity"] - r1["specificity"]) * 100

    print(f"\nAccuracy change: {acc_diff:+.2f}pp")
    print(f"Sensitivity change: {sens_diff:+.2f}pp")
    print(f"Specificity change: {spec_diff:+.2f}pp")

    print("\nConfusion matrix changes:")
    cm_diff = r2["cm"] - r1["cm"]
    print(f"[[{cm_diff[0,0]:+4d}, {cm_diff[0,1]:+4d}],")
    print(f" [{cm_diff[1,0]:+4d}, {cm_diff[1,1]:+4d}]]")

    # Compare to Novo benchmark
    print("\n" + "=" * 80)
    print("COMPARISON TO NOVO BENCHMARK")
    print("=" * 80)

    novo_acc = 0.617  # 61.7%
    novo_cm = np.array([[19778, 49962], [4186, 67633]])

    for threshold in thresholds:
        r = results[threshold]
        acc_gap = (r["accuracy"] - novo_acc) * 100

        print(f"\n--- Threshold {threshold} vs Novo ---")
        print(f"Our accuracy: {r['accuracy']*100:.2f}%")
        print(f"Novo accuracy: {novo_acc*100:.2f}%")
        print(f"Gap: {acc_gap:+.2f}pp")

        cm_diff = r["cm"] - novo_cm
        print("\nConfusion matrix difference from Novo:")
        print(f"[[{cm_diff[0,0]:+5d}, {cm_diff[0,1]:+5d}],")
        print(f" [{cm_diff[1,0]:+5d}, {cm_diff[1,1]:+5d}]]")

        total_diff = np.abs(cm_diff).sum()
        print(f"Total cell differences: {total_diff} predictions")

    # Save results
    output_path = "test_results/harvey_threshold_comparison.csv"
    Path("test_results").mkdir(exist_ok=True)

    comparison_df = pd.DataFrame(
        {
            "threshold": list(results.keys()),
            "accuracy": [r["accuracy"] for r in results.values()],
            "sensitivity": [r["sensitivity"] for r in results.values()],
            "specificity": [r["specificity"] for r in results.values()],
            "precision": [r["precision"] for r in results.values()],
            "f1_score": [r["f1"] for r in results.values()],
            "tn": [r["tn"] for r in results.values()],
            "fp": [r["fp"] for r in results.values()],
            "fn": [r["fn"] for r in results.values()],
            "tp": [r["tp"] for r in results.values()],
        }
    )

    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
