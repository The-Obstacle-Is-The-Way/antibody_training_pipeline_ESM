#!/usr/bin/env python3
"""
Targeted Permutation Testing: Finding the EXACT Novo Matrix

Tests strategic swaps of reclassification candidates to hit [[40, 19], [10, 17]] exactly.

All permutations maintain:
- 5 reclassifications (specific → non-specific)
- 30 removals by pure PSR (S3)
- 59 specific / 27 non-specific = 86 total
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "novo_parity"
DATASETS_DIR = EXPERIMENTS_DIR / "datasets"
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "permutations" / "targeted"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Novo target
NOVO_CM = np.array([[40, 19], [10, 17]])
NOVO_ACC = 0.6628

print("=" * 80)
print("TARGETED PERMUTATION TESTING: Finding the Exact Novo Matrix")
print("=" * 80)
print()
print("Target: [[40, 19], [10, 17]]")
print()

# Load data
df_116 = pd.read_csv(
    REPO_ROOT / "test_datasets" / "jain" / "jain_ELISA_ONLY_116_with_zscores.csv"
)
bio = pd.read_csv(REPO_ROOT / "test_datasets" / "jain_sd03.csv")

# Merge biophysical data
bio_clean = bio[
    [
        "Name",
        "Poly-Specificity Reagent (PSR) SMP Score (0-1)",
        "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average",
        "HIC Retention Time (Min)a",
        "Fab Tm by DSF (°C)",
    ]
].copy()
bio_clean.columns = ["id", "psr", "ac_sins", "hic", "fab_tm"]
df_116 = df_116.merge(bio_clean, on="id", how="left")

# Load model
model_path = REPO_ROOT / "models" / "boughter_vh_esm1v_logreg.pkl"
with open(model_path, "rb") as f:
    classifier = pickle.load(f)

print("Loaded 116-antibody dataset and model")
print()

# ============================================================================
# PERMUTATION DEFINITIONS
# ============================================================================

PERMUTATIONS = {
    "P5": {
        "desc": "Baseline (4 PSR + infliximab)",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "olaratumab"],
        "tier_b": ["infliximab"],
    },
    "P5b": {
        "desc": "Swap Tier B: infliximab → atezolizumab",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "olaratumab"],
        "tier_b": ["atezolizumab"],
    },
    "P5c": {
        "desc": "Swap Tier B: infliximab → denosumab",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "olaratumab"],
        "tier_b": ["denosumab"],
    },
    "P5d": {
        "desc": "Swap weakest PSR: olaratumab → basiliximab (PSR=0.397, AC-SINS=28.76)",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "basiliximab"],
        "tier_b": ["infliximab"],
    },
    "P5e": {
        "desc": "Swap weakest PSR: olaratumab → eldelumab (lowest Tm=59.50)",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "eldelumab"],
        "tier_b": ["infliximab"],
    },
    "P5f": {
        "desc": "Swap weakest PSR: olaratumab → lirilumab (extreme HIC=25, AC-SINS=21)",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "lirilumab"],
        "tier_b": ["infliximab"],
    },
    "P5g": {
        "desc": "Swap weakest PSR: olaratumab → glembatumumab (AC-SINS=28.88)",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "glembatumumab"],
        "tier_b": ["infliximab"],
    },
    "P5h": {
        "desc": "Swap weakest PSR: olaratumab → seribantumab (AC-SINS=21.21)",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "seribantumab"],
        "tier_b": ["infliximab"],
    },
    "P5i": {
        "desc": "Swap borderline FN: bavituximab → basiliximab",
        "tier_a": ["bimagrumab", "basiliximab", "ganitumab", "olaratumab"],
        "tier_b": ["infliximab"],
    },
    "P5j": {
        "desc": "Double swap: olaratumab→basiliximab + infliximab→atezolizumab",
        "tier_a": ["bimagrumab", "bavituximab", "ganitumab", "basiliximab"],
        "tier_b": ["atezolizumab"],
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def remove_by_pure_psr(df_after_reclass, n=30):
    """S3: Remove top N by pure PSR ranking"""
    specific = df_after_reclass[df_after_reclass["label"] == 0].copy()
    specific_sorted = specific.sort_values("psr", ascending=False)
    return specific_sorted.head(n)["id"].tolist()


def test_permutation(perm_id, config):
    """Test a single permutation"""
    print(f"Testing {perm_id}: {config['desc']}")

    # Step 1: Reclassify
    reclass_ids = config["tier_a"] + config["tier_b"]
    df_work = df_116.copy()

    # Mark reclassified
    df_work["reclassified"] = False
    df_work["reclassification_reason"] = ""
    df_work["label_original"] = df_work["label"].copy()

    for ab_id in reclass_ids:
        if ab_id in df_work["id"].values:
            idx = df_work[df_work["id"] == ab_id].index[0]
            if df_work.loc[idx, "label"] == 0:  # Only flip if currently specific
                df_work.loc[idx, "label"] = 1
                df_work.loc[idx, "reclassified"] = True
                if ab_id in config["tier_a"]:
                    df_work.loc[idx, "reclassification_reason"] = "Tier A: PSR-based"
                else:
                    df_work.loc[idx, "reclassification_reason"] = (
                        "Tier B: Clinical evidence"
                    )

    # Step 2: Remove by pure PSR
    remove_ids = remove_by_pure_psr(df_work, n=30)

    # Create final dataset
    df_final = df_work[~df_work["id"].isin(remove_ids)].copy()

    # Validate distribution
    n_spec = (df_final["label"] == 0).sum()
    n_nonspec = (df_final["label"] == 1).sum()
    n_total = len(df_final)

    print(
        f"  Distribution: {n_spec} specific / {n_nonspec} non-specific = {n_total} total",
        end="",
    )
    if n_spec == 59 and n_nonspec == 27 and n_total == 86:
        print(" ✅")
    else:
        print(" ❌ WRONG! Expected 59/27=86")
        return None

    # Step 3: Run inference
    sequences = df_final["vh_sequence"].tolist()
    y_true = df_final["label"].values

    X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Compare to Novo
    diff = np.abs(cm - NOVO_CM).sum()
    exact_match = np.array_equal(cm, NOVO_CM)

    print(f"  CM: {cm.tolist()}")
    print(f"  Accuracy: {acc:.4f} ({int(acc*len(df_final))}/{len(df_final)})")
    print(f"  Diff from Novo: {diff} cells", end="")

    if exact_match:
        print(" 🎉 EXACT MATCH!")
    elif diff <= 2:
        print(" ✨ Very close!")
    else:
        print()

    # Save results
    result = {
        "permutation_id": perm_id,
        "description": config["desc"],
        "tier_a": config["tier_a"],
        "tier_b": config["tier_b"],
        "reclassified_ids": reclass_ids,
        "removed_ids": remove_ids,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "novo_cm": NOVO_CM.tolist(),
        "novo_accuracy": NOVO_ACC,
        "difference_cells": int(diff),
        "exact_match": bool(exact_match),
        "tn_match": bool(cm[0, 0] == NOVO_CM[0, 0]),
        "fp_match": bool(cm[0, 1] == NOVO_CM[0, 1]),
        "fn_diff": int(cm[1, 0] - NOVO_CM[1, 0]),
        "tp_diff": int(cm[1, 1] - NOVO_CM[1, 1]),
        "timestamp": datetime.now().isoformat(),
    }

    result_path = RESULTS_DIR / f"{perm_id}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save dataset if exact match
    if exact_match or diff <= 2:
        dataset_path = DATASETS_DIR / f"jain_86_{perm_id.lower()}.csv"
        df_final["prediction"] = y_pred
        df_final["prob_specific"] = y_proba[:, 0]
        df_final["prob_nonspecific"] = y_proba[:, 1]
        df_final.to_csv(dataset_path, index=False)
        print(f"  💾 Saved dataset: {dataset_path.name}")

    print()
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = []
    exact_matches = []

    # Test all permutations
    for perm_id, config in PERMUTATIONS.items():
        result = test_permutation(perm_id, config)
        if result:
            results.append(result)
            if result["exact_match"]:
                exact_matches.append(perm_id)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Sort by difference from Novo
    results_sorted = sorted(results, key=lambda x: x["difference_cells"])

    print("| Rank | Perm | CM | Diff | Acc | Description |")
    print("|------|------|----|------|-----|-------------|")
    for i, r in enumerate(results_sorted[:10], 1):
        cm_str = str(r["confusion_matrix"]).replace(" ", "")
        mark = "🎉" if r["exact_match"] else "✨" if r["difference_cells"] <= 2 else ""
        print(
            f"| {i} | {r['permutation_id']} | {cm_str} | {r['difference_cells']} | {r['accuracy']:.4f} | {r['description'][:50]} {mark} |"
        )

    print()

    if exact_matches:
        print(f"🎉 EXACT MATCHES FOUND: {', '.join(exact_matches)}")
        print()
        print("Next step: Validate biophysical plausibility of exact matches")
    else:
        print("No exact matches found.")
        print(
            f'Closest match: {results_sorted[0]["permutation_id"]} ({results_sorted[0]["difference_cells"]} cells off)'
        )

    print()
    print(f"Results saved to: {RESULTS_DIR}")
