#!/usr/bin/env python3
"""
Batch Permutation Testing: Find Novo's Exact 86-Antibody Set

Systematically tests reclassification and removal strategies to match Novo's
confusion matrix [[40, 19], [10, 17]] exactly.

Usage:
    python batch_permutation_test.py [--permutations P1,P2,P3] [--quick]
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Input files
ELISA_116_PATH = REPO_ROOT / "test_datasets" / "jain_ELISA_ONLY_116.csv"
SD03_PATH = REPO_ROOT / "test_datasets" / "jain_sd03.csv"
MODEL_PATH = REPO_ROOT / "models" / "boughter_vh_esm1v_logreg.pkl"

# Output directory
RESULTS_DIR = REPO_ROOT / "experiments" / "novo_parity" / "results" / "permutations"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Novo target
NOVO_CM = np.array([[40, 19], [10, 17]])
NOVO_ACCURACY = 57 / 86

print("Loading data and model...")
df_116 = pd.read_csv(ELISA_116_PATH)
df_sd03 = pd.read_csv(SD03_PATH)
df = df_116.merge(
    df_sd03[
        [
            "Name",
            "Poly-Specificity Reagent (PSR) SMP Score (0-1)",
            "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average",
            "HIC Retention Time (Min)a",
            "Fab Tm by DSF (°C)",
        ]
    ],
    left_on="id",
    right_on="Name",
    how="left",
)
df = df.rename(
    columns={
        "Poly-Specificity Reagent (PSR) SMP Score (0-1)": "psr",
        "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": "ac_sins",
        "HIC Retention Time (Min)a": "hic",
        "Fab Tm by DSF (°C)": "fab_tm",
    }
)

with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)

print(f"✅ Loaded {len(df)} antibodies from 116 set")
print(f"✅ Loaded model: {classifier.classifier.__class__.__name__}")
print()

# ==============================================================================
# RECLASSIFICATION STRATEGIES
# ==============================================================================


def get_reclassification_R1():
    """R1: PSR-Hybrid (4 PSR >0.4 + infliximab)"""
    tier_a = ["bimagrumab", "bavituximab", "ganitumab", "olaratumab"]
    tier_b = ["infliximab"]
    return tier_a + tier_b


def get_reclassification_R2():
    """R2: PSR + Atezolizumab (4 PSR >0.4 + atezolizumab)"""
    tier_a = ["bimagrumab", "bavituximab", "ganitumab", "olaratumab"]
    tier_b = ["atezolizumab"]
    return tier_a + tier_b


def get_reclassification_R3(df):
    """R3: Pure PSR - top 5 by PSR among ELISA=0"""
    elisa_0 = df[df["elisa_flags"] == 0].copy()
    top5 = elisa_0.nlargest(5, "psr")
    return top5["id"].tolist()


def get_reclassification_R4(df):
    """R4: Total Flags - top 5 by total_flags among ELISA=0"""
    elisa_0 = df[df["elisa_flags"] == 0].copy()
    top5 = elisa_0.nlargest(5, "total_flags")
    return top5["id"].tolist()


def get_reclassification_R5(df):
    """R5: AC-SINS Based - top 5 by AC-SINS among ELISA=0"""
    elisa_0 = df[df["elisa_flags"] == 0].copy()
    top5 = elisa_0.nlargest(5, "ac_sins")
    return top5["id"].tolist()


RECLASSIFICATION_STRATEGIES = {
    "R1": ("PSR-Hybrid (4 PSR + infliximab)", get_reclassification_R1),
    "R2": ("PSR + Atezolizumab", get_reclassification_R2),
    "R3": ("Pure PSR Top 5", get_reclassification_R3),
    "R4": ("Total Flags Top 5", get_reclassification_R4),
    "R5": ("AC-SINS Top 5", get_reclassification_R5),
}

# ==============================================================================
# REMOVAL STRATEGIES
# ==============================================================================


def calculate_composite_risk(df_specific, weights=None):
    """Calculate composite risk score with optional weights"""
    if weights is None:
        weights = {"psr": 1.0, "ac_sins": 1.0, "hic": 1.0, "tm_inv": 1.0}

    df_s = df_specific.copy()

    # Normalize each metric 0-1
    df_s["psr_norm"] = df_s["psr"]  # Already 0-1

    # Avoid division by zero
    ac_range = df_s["ac_sins"].max() - df_s["ac_sins"].min()
    if ac_range > 0:
        df_s["ac_sins_norm"] = (df_s["ac_sins"] - df_s["ac_sins"].min()) / ac_range
    else:
        df_s["ac_sins_norm"] = 0.0

    hic_range = df_s["hic"].max() - df_s["hic"].min()
    if hic_range > 0:
        df_s["hic_norm"] = (df_s["hic"] - df_s["hic"].min()) / hic_range
    else:
        df_s["hic_norm"] = 0.0

    tm_range = df_s["fab_tm"].max() - df_s["fab_tm"].min()
    if tm_range > 0:
        df_s["tm_inv_norm"] = 1 - ((df_s["fab_tm"] - df_s["fab_tm"].min()) / tm_range)
    else:
        df_s["tm_inv_norm"] = 0.0

    # Composite risk
    df_s["risk_score"] = (
        weights["psr"] * df_s["psr_norm"]
        + weights["ac_sins"] * df_s["ac_sins_norm"]
        + weights["hic"] * df_s["hic_norm"]
        + weights["tm_inv"] * df_s["tm_inv_norm"]
    )

    return df_s


def get_removal_S1(df_after_reclass):
    """S1: Composite Risk (equal weights)"""
    specific = df_after_reclass[df_after_reclass["label"] == 0].copy()
    specific = calculate_composite_risk(specific)
    specific_sorted = specific.sort_values("risk_score", ascending=False)
    return specific_sorted.head(30)["id"].tolist()


def get_removal_S2(df_after_reclass):
    """S2: PSR-Weighted Composite (PSR 2x)"""
    specific = df_after_reclass[df_after_reclass["label"] == 0].copy()
    weights = {"psr": 2.0, "ac_sins": 1.0, "hic": 1.0, "tm_inv": 1.0}
    specific = calculate_composite_risk(specific, weights)
    specific_sorted = specific.sort_values("risk_score", ascending=False)
    return specific_sorted.head(30)["id"].tolist()


def get_removal_S3(df_after_reclass):
    """S3: Pure PSR"""
    specific = df_after_reclass[df_after_reclass["label"] == 0].copy()
    specific_sorted = specific.sort_values("psr", ascending=False)
    return specific_sorted.head(30)["id"].tolist()


def get_removal_S4(df_after_reclass):
    """S4: Total Flags"""
    specific = df_after_reclass[df_after_reclass["label"] == 0].copy()
    specific_sorted = specific.sort_values("total_flags", ascending=False)
    return specific_sorted.head(30)["id"].tolist()


def get_removal_S5(df_after_reclass):
    """S5: AC-SINS Focus"""
    specific = df_after_reclass[df_after_reclass["label"] == 0].copy()
    specific_sorted = specific.sort_values("ac_sins", ascending=False)
    return specific_sorted.head(30)["id"].tolist()


REMOVAL_STRATEGIES = {
    "S1": ("Composite Risk (equal)", get_removal_S1),
    "S2": ("PSR-Weighted Composite", get_removal_S2),
    "S3": ("Pure PSR", get_removal_S3),
    "S4": ("Total Flags", get_removal_S4),
    "S5": ("AC-SINS Focus", get_removal_S5),
}

# ==============================================================================
# PERMUTATION DEFINITIONS
# ==============================================================================

PERMUTATIONS = {
    "P1": ("R1", "S1", "Exp-05 Baseline"),
    "P2": ("R2", "S1", "Swap infliximab → atezolizumab"),
    "P3": ("R1", "S2", "PSR-weighted removal"),
    "P4": ("R1", "S4", "Remove by total_flags"),
    "P5": ("R1", "S3", "Remove by pure PSR"),
    "P6": ("R3", "S1", "Pure PSR reclassification"),
    "P7": ("R4", "S1", "Total flags reclassification"),
    "P8": ("R5", "S1", "AC-SINS reclassification"),
    "P9": ("R2", "S2", "Atezolizumab + PSR-weighted"),
    "P10": ("R3", "S3", "Pure PSR throughout"),
    "P11": ("R4", "S4", "Total flags throughout"),
    "P12": ("R5", "S5", "AC-SINS throughout"),
}

# ==============================================================================
# PERMUTATION EXECUTION
# ==============================================================================


def apply_permutation(df, reclass_ids, removal_ids):
    """Apply reclassification and removal to create 86-antibody set"""
    df_work = df.copy()

    # Step 1: Reclassify
    for ab_id in reclass_ids:
        df_work.loc[df_work["id"] == ab_id, "label"] = 1
        df_work.loc[df_work["id"] == ab_id, "reclassified"] = True

    # Verify counts after reclassification
    counts_after_reclass = df_work["label"].value_counts()
    specific_after = counts_after_reclass.get(0, 0)
    nonspec_after = counts_after_reclass.get(1, 0)

    assert (
        specific_after == 89
    ), f"Expected 89 specific after reclass, got {specific_after}"
    assert (
        nonspec_after == 27
    ), f"Expected 27 non-specific after reclass, got {nonspec_after}"

    # Step 2: Remove
    df_work = df_work[~df_work["id"].isin(removal_ids)].copy()

    # Verify final counts
    final_counts = df_work["label"].value_counts()
    final_specific = final_counts.get(0, 0)
    final_nonspec = final_counts.get(1, 0)

    assert final_specific == 59, f"Expected 59 specific in final, got {final_specific}"
    assert (
        final_nonspec == 27
    ), f"Expected 27 non-specific in final, got {final_nonspec}"
    assert len(df_work) == 86, f"Expected 86 total, got {len(df_work)}"

    return df_work


def run_inference(df_test):
    """Run model inference on test set"""
    sequences = df_test["vh_sequence"].tolist()
    y_true = df_test["label"].values

    # Generate embeddings (suppress progress bars)
    X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)

    # Predictions
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return cm, accuracy, y_pred, y_proba


def test_permutation(perm_id, reclass_strategy, removal_strategy, description):
    """Test a single permutation"""
    print(f"\n{'='*80}")
    print(f"Testing {perm_id}: {description}")
    print(f"  Reclassification: {reclass_strategy}")
    print(f"  Removal: {removal_strategy}")
    print(f"{'='*80}")

    # Get reclassification IDs
    reclass_func = RECLASSIFICATION_STRATEGIES[reclass_strategy][1]
    if callable(reclass_func):
        if reclass_strategy in ["R1", "R2"]:
            reclass_ids = reclass_func()
        else:
            reclass_ids = reclass_func(df)

    print(f"\n✅ Reclassification: {len(reclass_ids)} antibodies")
    for i, ab_id in enumerate(reclass_ids, 1):
        print(f"  {i}. {ab_id}")

    # Apply reclassification
    df_after_reclass = df.copy()
    for ab_id in reclass_ids:
        df_after_reclass.loc[df_after_reclass["id"] == ab_id, "label"] = 1

    # Get removal IDs
    removal_func = REMOVAL_STRATEGIES[removal_strategy][1]
    removal_ids = removal_func(df_after_reclass)

    print(f"\n✅ Removal: {len(removal_ids)} antibodies (top 10 shown)")
    for i, ab_id in enumerate(removal_ids[:10], 1):
        print(f"  {i}. {ab_id}")
    if len(removal_ids) > 10:
        print(f"  ... and {len(removal_ids) - 10} more")

    # Create final dataset
    df_final = apply_permutation(df, reclass_ids, removal_ids)

    print(f"\n✅ Final dataset: {len(df_final)} antibodies")
    print(f"  Specific: {(df_final['label']==0).sum()}")
    print(f"  Non-specific: {(df_final['label']==1).sum()}")

    # Run inference
    print("\n🔄 Running inference...")
    cm, accuracy, y_pred, y_proba = run_inference(df_final)

    # Results
    print(f"\n{'='*80}")
    print(f"RESULTS: {perm_id}")
    print(f"{'='*80}")
    print("\nConfusion Matrix:")
    print(f"  [[{cm[0,0]:2d}, {cm[0,1]:2d}],")
    print(f"   [{cm[1,0]:2d}, {cm[1,1]:2d}]]")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Compare to Novo
    match = np.array_equal(cm, NOVO_CM)
    print(f"\n{'='*80}")
    print("COMPARISON TO NOVO")
    print(f"{'='*80}")
    print(
        f"Novo CM:  [[{NOVO_CM[0,0]:2d}, {NOVO_CM[0,1]:2d}], [{NOVO_CM[1,0]:2d}, {NOVO_CM[1,1]:2d}]]"
    )
    print(f"Ours:     [[{cm[0,0]:2d}, {cm[0,1]:2d}], [{cm[1,0]:2d}, {cm[1,1]:2d}]]")
    print(
        f"Diff:     [[{cm[0,0]-NOVO_CM[0,0]:+2d}, {cm[0,1]-NOVO_CM[0,1]:+2d}], [{cm[1,0]-NOVO_CM[1,0]:+2d}, {cm[1,1]-NOVO_CM[1,1]:+2d}]]"
    )

    if match:
        print("\n🎉🎉🎉 EXACT MATCH! We found Novo's permutation! 🎉🎉🎉")
    else:
        diff_count = np.abs(cm - NOVO_CM).sum()
        print(f"\n⚠️ Not a match (difference: {diff_count} cells)")

    # Save results
    result = {
        "permutation_id": perm_id,
        "description": description,
        "reclassification_strategy": reclass_strategy,
        "removal_strategy": removal_strategy,
        "reclassified_ids": reclass_ids,
        "removed_ids": removal_ids,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy),
        "exact_match": bool(match),
        "timestamp": datetime.now().isoformat(),
    }

    result_file = RESULTS_DIR / f"{perm_id}_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Results saved to: {result_file}")

    return result


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Batch permutation testing")
    parser.add_argument(
        "--permutations",
        type=str,
        default="P1,P2,P3,P4,P5",
        help="Comma-separated list of permutation IDs (e.g., P1,P2,P3)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip embedding generation (use cached if available)",
    )
    args = parser.parse_args()

    perm_ids = [p.strip() for p in args.permutations.split(",")]

    print("=" * 80)
    print("BATCH PERMUTATION TESTING: Reverse Engineering Novo's 86-Antibody Set")
    print("=" * 80)
    print()
    print(f"Testing {len(perm_ids)} permutations: {', '.join(perm_ids)}")
    print()

    results = []
    matches = []

    for perm_id in perm_ids:
        if perm_id not in PERMUTATIONS:
            print(f"⚠️ Warning: Unknown permutation {perm_id}, skipping")
            continue

        reclass, removal, desc = PERMUTATIONS[perm_id]
        result = test_permutation(perm_id, reclass, removal, desc)
        results.append(result)

        if result["exact_match"]:
            matches.append(perm_id)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Tested: {len(results)} permutations")
    print(f"Exact matches: {len(matches)}")

    if matches:
        print(f"\n🎉 EXACT MATCH(ES) FOUND: {', '.join(matches)}")
        print("\nMatching permutations:")
        for perm_id in matches:
            result = [r for r in results if r["permutation_id"] == perm_id][0]
            print(f"  {perm_id}: {result['description']}")
    else:
        print("\n⚠️ No exact matches found. Closest results:")
        # Sort by total difference
        for result in sorted(
            results,
            key=lambda r: np.abs(np.array(r["confusion_matrix"]) - NOVO_CM).sum(),
        )[:3]:
            cm = np.array(result["confusion_matrix"])
            diff = np.abs(cm - NOVO_CM).sum()
            print(
                f"  {result['permutation_id']}: Difference = {diff} cells, CM = {cm.tolist()}"
            )

    print()
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
