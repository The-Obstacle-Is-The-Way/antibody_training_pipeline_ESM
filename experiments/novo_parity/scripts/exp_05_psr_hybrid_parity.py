#!/usr/bin/env python3
"""
Experiment 05: PSR-Hybrid Parity Approach
==========================================

Hypothesis: Novo used ELISA as primary filter, then applied biophysical QC
            to reclassify ELISA-discordant antibodies and remove high-risk candidates.

Method:
  Step 1: Start with 116 ELISA-only antibodies (94 spec / 22 nonspec)
  Step 2: Reclassify 5 specific → non-specific
          - Tier A (PSR evidence): 4 antibodies with ELISA=0 but PSR >0.4
          - Tier B (Clinical evidence): 1 antibody with strong ADA/aggregation data
  Step 3: Calculate composite risk for remaining specific antibodies
  Step 4: Remove top 30 by composite risk
  Final: 59 specific + 27 non-specific = 86 total

Author: Claude Code
Date: 2025-11-03
Branch: ray/novo-parity-experiments
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# File paths (resolve relative to script location)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = (
    SCRIPT_DIR.parent.parent.parent
)  # experiments/novo_parity/scripts -> repo root
INPUT_116 = BASE_DIR / "test_datasets/jain_ELISA_ONLY_116.csv"
INPUT_SD03 = BASE_DIR / "test_datasets/jain_sd03.csv"
OUTPUT_DIR = BASE_DIR / "experiments/novo_parity"
OUTPUT_DATASET = OUTPUT_DIR / "datasets/jain_86_exp05.csv"
OUTPUT_AUDIT = OUTPUT_DIR / "results/audit_exp05.json"
OUTPUT_REMOVED = OUTPUT_DIR / "results/removed_30_exp05.txt"

# Constants
PSR_FLIP_THRESHOLD = 0.4
TIMESTAMP = datetime.now().isoformat()

# Tier A: PSR-based reclassification (ELISA=0, PSR >0.4)
TIER_A_FLIPS = ["bimagrumab", "bavituximab", "ganitumab", "olaratumab"]

# Tier B: Clinical/biophysics evidence
TIER_B_FLIP = "infliximab"  # 61% ADA + aggregation + chimeric

ALL_FLIPS = TIER_A_FLIPS + [TIER_B_FLIP]


def load_data():
    """Load and merge datasets"""
    print("Loading data...")

    # Load 116 ELISA-only set
    df116 = pd.read_csv(INPUT_116)
    print(f"  Loaded 116 antibodies: {len(df116)} rows")
    print(f"    Specific: {(df116['label'] == 0).sum()}")
    print(f"    Non-specific: {(df116['label'] == 1).sum()}")

    # Load SD03 biophysical data
    sd03 = pd.read_csv(INPUT_SD03)
    print(f"  Loaded SD03: {len(sd03)} rows")

    # Merge
    merged = df116.merge(
        sd03[
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

    # Rename columns for easier handling
    merged = merged.rename(
        columns={
            "Poly-Specificity Reagent (PSR) SMP Score (0-1)": "psr",
            "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": "ac_sins",
            "HIC Retention Time (Min)a": "hic",
            "Fab Tm by DSF (°C)": "fab_tm",
        }
    )

    print(f"  Merged: {len(merged)} rows")
    print(f"  Missing PSR: {merged['psr'].isna().sum()}")

    return merged


def calculate_composite_risk(df):
    """Calculate composite risk score for specific antibodies only"""
    print("\nCalculating composite risk scores...")

    # Only calculate for specific antibodies (label=0)
    specific = df[df["label"] == 0].copy()

    # Normalize each metric 0-1
    specific["psr_norm"] = specific["psr"]  # Already 0-1
    specific["ac_sins_norm"] = (specific["ac_sins"] - specific["ac_sins"].min()) / (
        specific["ac_sins"].max() - specific["ac_sins"].min()
    )
    specific["hic_norm"] = (specific["hic"] - specific["hic"].min()) / (
        specific["hic"].max() - specific["hic"].min()
    )
    specific["tm_inv_norm"] = 1 - (
        (specific["fab_tm"] - specific["fab_tm"].min())
        / (specific["fab_tm"].max() - specific["fab_tm"].min())
    )

    # Composite risk = sum of normalized metrics
    specific["risk_score"] = (
        specific["psr_norm"]
        + specific["ac_sins_norm"]
        + specific["hic_norm"]
        + specific["tm_inv_norm"]
    )

    # Merge back to full dataframe
    df = df.merge(specific[["id", "risk_score"]], on="id", how="left")

    print(f"  Calculated risk for {len(specific)} specific antibodies")
    print(
        f"  Risk range: {specific['risk_score'].min():.3f} - {specific['risk_score'].max():.3f}"
    )

    return df


def step1_reclassify(df):
    """Step 1: Reclassify 5 specific → non-specific"""
    print("\n" + "=" * 60)
    print("STEP 1: Reclassify 5 specific → non-specific")
    print("=" * 60)

    df = df.copy()
    df["label_original"] = df["label"]
    df["reclassified"] = False
    df["reclassification_reason"] = ""

    # Reclassify Tier A (PSR-based)
    for ab_id in TIER_A_FLIPS:
        idx = df[df["id"] == ab_id].index
        if len(idx) > 0:
            psr_val = df.loc[idx[0], "psr"]
            df.loc[idx, "label"] = 1
            df.loc[idx, "reclassified"] = True
            df.loc[idx, "reclassification_reason"] = (
                f"Tier A: PSR={psr_val:.3f} >0.4 despite ELISA=0"
            )
            print(f"  ✅ Reclassified {ab_id}: PSR={psr_val:.3f}")

    # Reclassify Tier B (clinical evidence)
    idx = df[df["id"] == TIER_B_FLIP].index
    if len(idx) > 0:
        df.loc[idx, "label"] = 1
        df.loc[idx, "reclassified"] = True
        df.loc[idx, "reclassification_reason"] = (
            "Tier B: 61% ADA (NEJM) + aggregation + chimeric"
        )
        print(f"  ✅ Reclassified {TIER_B_FLIP}: Clinical evidence (61% ADA)")

    # Verify counts
    spec_count = (df["label"] == 0).sum()
    nonspec_count = (df["label"] == 1).sum()

    print("\nAfter reclassification:")
    print(f"  Specific: {spec_count}")
    print(f"  Non-specific: {nonspec_count}")
    print(f"  Total: {len(df)}")
    print("  Expected: 89 spec / 27 nonspec / 116 total")

    assert spec_count == 89, f"Expected 89 specific, got {spec_count}"
    assert nonspec_count == 27, f"Expected 27 non-specific, got {nonspec_count}"

    return df


def step2_remove_top30(df):
    """Step 2: Remove top 30 specific antibodies by composite risk"""
    print("\n" + "=" * 60)
    print("STEP 2: Remove top 30 specific by composite risk")
    print("=" * 60)

    # Get remaining specific antibodies
    specific = df[df["label"] == 0].copy()

    print(f"\nRemaining specific antibodies: {len(specific)}")

    # Sort by risk (descending), then by total_flags (descending), then by id (alphabetical)
    specific_sorted = specific.sort_values(
        by=["risk_score", "total_flags", "id"], ascending=[False, False, True]
    )

    # Top 30 to remove
    to_remove = specific_sorted.head(30)

    print("\nTop 30 by risk score:")
    for i, row in enumerate(to_remove.itertuples(), 1):
        print(
            f"  {i:2d}. {row.id:20s} risk={row.risk_score:.3f} psr={row.psr:.3f} total_flags={row.total_flags}"
        )

    # Mark for removal
    df["removed"] = False
    df.loc[to_remove.index, "removed"] = True
    df.loc[to_remove.index, "removal_reason"] = (
        "Top 30 by composite risk (PSR+AC-SINS+HIC+Tm)"
    )

    # Create final dataset (removed antibodies excluded)
    final = df[~df["removed"]].copy()

    # Verify counts
    spec_count = (final["label"] == 0).sum()
    nonspec_count = (final["label"] == 1).sum()

    print("\nFinal dataset after removal:")
    print(f"  Specific: {spec_count}")
    print(f"  Non-specific: {nonspec_count}")
    print(f"  Total: {len(final)}")
    print("  Expected: 59 spec / 27 nonspec / 86 total")

    assert spec_count == 59, f"Expected 59 specific, got {spec_count}"
    assert nonspec_count == 27, f"Expected 27 non-specific, got {nonspec_count}"
    assert len(final) == 86, f"Expected 86 total, got {len(final)}"

    return df, final, to_remove["id"].tolist()


def save_outputs(df_all, df_final, removed_ids):
    """Save all outputs"""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save final 86 dataset
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_DATASET, index=False)
    print(f"  ✅ Saved dataset: {OUTPUT_DATASET}")

    # Save removed IDs
    OUTPUT_REMOVED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REMOVED, "w") as f:
        f.write("# Experiment 05: Removed 30 Antibodies (Top by Composite Risk)\n")
        f.write("# Sorted by risk score descending\n\n")
        for ab_id in removed_ids:
            f.write(f"{ab_id}\n")
    print(f"  ✅ Saved removed IDs: {OUTPUT_REMOVED}")

    # Generate audit log
    reclassified = df_all[df_all["reclassified"]].copy()

    audit = {
        "experiment_id": "exp_05",
        "experiment_name": "PSR-Hybrid Parity Approach",
        "timestamp": TIMESTAMP,
        "script": str(Path(__file__).relative_to(BASE_DIR)),
        "input_files": [
            str(INPUT_116.relative_to(BASE_DIR)),
            str(INPUT_SD03.relative_to(BASE_DIR)),
        ],
        "output_file": str(OUTPUT_DATASET.relative_to(BASE_DIR)),
        "method": {
            "step_1": "Reclassify 5 specific → non-specific",
            "step_2": "Remove top 30 specific by composite risk",
        },
        "parameters": {
            "psr_flip_threshold": PSR_FLIP_THRESHOLD,
            "composite_risk_formula": "PSR_norm + AC-SINS_norm + HIC_norm + (1 - Tm_norm)",
            "tier_a_rationale": "ELISA=0 but PSR >0.4 (polyreactivity missed by ELISA)",
            "tier_b_rationale": "61% ADA (NEJM) + aggregation + chimeric",
        },
        "reclassified_ids": {
            "tier_a_psr_based": [
                {
                    "id": row["id"],
                    "psr": float(row["psr"]),
                    "reason": row["reclassification_reason"],
                }
                for _, row in reclassified[
                    reclassified["id"].isin(TIER_A_FLIPS)
                ].iterrows()
            ],
            "tier_b_clinical": [
                {"id": row["id"], "reason": row["reclassification_reason"]}
                for _, row in reclassified[reclassified["id"] == TIER_B_FLIP].iterrows()
            ],
        },
        "removed_ids": removed_ids,
        "counts": {
            "step_0_start": {"total": 116, "specific": 94, "non_specific": 22},
            "step_1_after_reclassification": {
                "total": 116,
                "specific": 89,
                "non_specific": 27,
            },
            "step_2_final": {"total": 86, "specific": 59, "non_specific": 27},
        },
        "validation": {
            "target_distribution": "59 specific / 27 non-specific = 86 total",
            "achieved": "✅ EXACT MATCH",
        },
    }

    with open(OUTPUT_AUDIT, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  ✅ Saved audit log: {OUTPUT_AUDIT}")


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 05: PSR-HYBRID PARITY APPROACH")
    print("=" * 60)
    print(f"Start time: {TIMESTAMP}")

    # Load data
    df = load_data()

    # Calculate composite risk
    df = calculate_composite_risk(df)

    # Step 1: Reclassify 5
    df = step1_reclassify(df)

    # Step 2: Remove top 30
    df_all, df_final, removed_ids = step2_remove_top30(df)

    # Save outputs
    save_outputs(df_all, df_final, removed_ids)

    print("\n" + "=" * 60)
    print("✅ EXPERIMENT 05 COMPLETE!")
    print("=" * 60)
    print("\nFinal distribution: 59 specific / 27 non-specific = 86 total")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Audit log: {OUTPUT_AUDIT}")
    print(f"Removed IDs: {OUTPUT_REMOVED}")


if __name__ == "__main__":
    main()
