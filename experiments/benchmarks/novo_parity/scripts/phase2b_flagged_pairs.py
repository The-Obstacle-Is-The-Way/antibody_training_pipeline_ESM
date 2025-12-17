#!/usr/bin/env python3
"""
Phase 2B: Test All 28 Flagged Specific Pairs

The 8 flagged specifics (those with non-ELISA developability flags):
1. nimotuzumab (HIC=25.0, 7.8σ outlier)
2. lebrikizumab (HIC=12.38, chromatography)
3. gemtuzumab (HIC=12.26, chromatography)
4. galiximab (HIC=12.20, chromatography)
5. bevacizumab (2 flags: chromatography + stability)
6. lampalizumab (stability)
7. otelixizumab (stability)
8. bapineuzumab (self-interaction)

C(8,2) = 28 pairs to test.

Usage:
    python -m experiments.benchmarks.novo_parity.scripts.phase2b_flagged_pairs
"""

from __future__ import annotations

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
)
DATASET_PATH = PROJECT_ROOT / "data/test/jain/canonical/jain_86_novo_parity.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments/benchmarks/novo_parity/results"

# The 8 flagged specifics
FLAGGED_SPECIFICS = [
    "nimotuzumab",
    "lebrikizumab",
    "gemtuzumab",
    "galiximab",
    "bevacizumab",
    "lampalizumab",
    "otelixizumab",
    "bapineuzumab",
]

# Novo target
NOVO_CM = np.array([[40, 17], [10, 19]])
NOVO_ACCURACY = 59 / 86  # 68.6%


def main() -> None:
    print("=" * 80)
    print("PHASE 2B: TEST ALL 28 FLAGGED PAIRS")
    print("=" * 80)
    print()

    # Load model
    print(f"Loading model: {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH, "rb") as f:
        classifier = pickle.load(f)  # nosec B301
    print("Model loaded successfully")
    print()

    # Load dataset
    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} antibodies")
    print()

    # Verify all flagged specifics exist and are currently specific
    print("Verifying flagged specifics...")
    for ab in FLAGGED_SPECIFICS:
        row = df[df["id"] == ab]
        if len(row) == 0:
            raise ValueError(f"'{ab}' not found in dataset!")
        if row.iloc[0]["label"] != 0:
            raise ValueError(f"'{ab}' is not labeled as specific (label=0)!")
    print(f"All {len(FLAGGED_SPECIFICS)} flagged specifics verified")
    print()

    # Pre-compute embeddings (only need to do this once)
    print("Generating ESM-1v embeddings for all 86 antibodies...")
    sequences = df["vh_sequence"].tolist()
    X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)
    print(f"Embeddings shape: {X_test.shape}")
    print()

    # Get predictions and probabilities (same for all tests)
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Show prediction info for flagged specifics
    print("=" * 80)
    print("FLAGGED SPECIFICS - MODEL PREDICTIONS")
    print("=" * 80)
    print()
    print(f"{'Antibody':<15} {'Pred':>6} {'P(non-spec)':>12} {'Will become':>15}")
    print("-" * 50)
    for ab in FLAGGED_SPECIFICS:
        idx = df[df["id"] == ab].index[0]
        pred = y_pred[idx]
        prob = y_proba[idx][1]
        if pred == 0:
            outcome = "FN if reclassified"
        else:
            outcome = "TP if reclassified"
        print(f"{ab:<15} {pred:>6} {prob:>12.4f} {outcome:>15}")
    print()

    # Test all pairs
    print("=" * 80)
    print("TESTING ALL C(8,2) = 28 PAIRS")
    print("=" * 80)
    print()

    all_pairs = list(combinations(FLAGGED_SPECIFICS, 2))
    matches = []
    results_detail = []

    for i, (ab1, ab2) in enumerate(all_pairs):
        # Create modified labels
        y_true_modified = df["label"].values.astype(int).copy()
        idx1 = df[df["id"] == ab1].index[0]
        idx2 = df[df["id"] == ab2].index[0]
        y_true_modified[idx1] = 1
        y_true_modified[idx2] = 1

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_modified, y_pred)
        accuracy = accuracy_score(y_true_modified, y_pred)

        is_match = np.array_equal(cm, NOVO_CM)

        result = {
            "pair": [ab1, ab2],
            "confusion_matrix": cm.tolist(),
            "accuracy": accuracy,
            "match": is_match,
        }
        results_detail.append(result)

        if is_match:
            matches.append((ab1, ab2, cm, accuracy))
            print(f"[{i + 1:2d}/28] {ab1} + {ab2}")
            print(f"         CM: {cm.tolist()} ✅ MATCH!")
        else:
            # Show if close
            diff = cm - NOVO_CM
            if np.abs(diff).sum() <= 4:  # Close to target
                print(f"[{i + 1:2d}/28] {ab1} + {ab2}")
                print(f"         CM: {cm.tolist()} (diff: {diff.tolist()})")

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    if matches:
        print(f"🎉 FOUND {len(matches)} MATCHING PAIR(S)! 🎉")
        print()
        for ab1, ab2, cm, accuracy in matches:
            print(f"  • {ab1} + {ab2}")
            print(f"    CM: {cm.tolist()}")
            print(f"    Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print()

            # Get biological justification
            for ab in [ab1, ab2]:
                row = df[df["id"] == ab].iloc[0]
                print(f"    {ab}:")
                print(f"      - total_flags: {row['total_flags']}")
                print(f"      - self_interaction: {row['flag_self_interaction']}")
                print(f"      - chromatography: {row['flag_chromatography']}")
                print(f"      - stability: {row['flag_stability']}")
                print(f"      - HIC: {row['hic']}")
            print()
    else:
        print("❌ NO EXACT MATCHES FOUND in flagged pairs")
        print()
        print("Closest results:")
        sorted_results = sorted(
            results_detail,
            key=lambda x: np.abs(np.array(x["confusion_matrix"]) - NOVO_CM).sum(),
        )
        for r in sorted_results[:5]:
            diff = np.array(r["confusion_matrix"]) - NOVO_CM
            print(f"  {r['pair'][0]} + {r['pair'][1]}: diff={diff.tolist()}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "novo_target": NOVO_CM.tolist(),
        "flagged_specifics": FLAGGED_SPECIFICS,
        "total_pairs_tested": len(all_pairs),
        "matches_found": len(matches),
        "matching_pairs": [[ab1, ab2] for ab1, ab2, _, _ in matches] if matches else [],
        "all_results": results_detail,
    }

    results_file = RESULTS_DIR / "phase2b_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {results_file}")

    return len(matches) > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
