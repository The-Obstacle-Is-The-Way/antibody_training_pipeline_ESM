#!/usr/bin/env python3
"""
Z-Score Analysis of Jain 116-Antibody ELISA-Only Set

Find biological aberrations via z-scoring:
  - VH/VL sequence length outliers
  - CDR H3 length outliers
  - Unusual charge/pI distributions
  - Known chimeric/discontinued/failed antibodies

Goal: Reverse-engineer which 30 antibodies Novo removed to get from 116 → 86

Date: 2025-11-03
"""

import numpy as np
import pandas as pd


def load_data():
    """Load the 116-antibody ELISA-only test set."""
    print("Loading 116-antibody ELISA-only set...")
    df = pd.read_csv("test_datasets/jain_ELISA_ONLY_116.csv")
    print(f"  Loaded: {len(df)} antibodies")
    print(
        f"  Distribution: {(df['label']==0).sum()} specific / {(df['label']==1).sum()} non-specific\n"
    )
    return df


def calculate_sequence_features(df):
    """Calculate sequence-based features for z-scoring."""
    print("Calculating sequence features...")

    # Sequence lengths
    df["vh_length"] = df["vh_sequence"].str.len()
    df["vl_length"] = df["vl_sequence"].str.len()

    # Charge at pH 7.4 (rough approximation)
    def calc_charge(seq):
        if pd.isna(seq):
            return np.nan
        pos = (
            seq.count("K") + seq.count("R") + seq.count("H")
        )  # His ~50% protonated at pH 7.4
        neg = seq.count("D") + seq.count("E")
        return pos - neg

    df["vh_charge"] = df["vh_sequence"].apply(calc_charge)
    df["vl_charge"] = df["vl_sequence"].apply(calc_charge)
    df["total_charge"] = df["vh_charge"] + df["vl_charge"]

    # CDR H3 length (IMGT positions 105-117, rough approximation)
    # We'll use a simple heuristic: extract ~middle third as proxy for CDR3
    def estimate_cdr_h3_length(vh_seq):
        if pd.isna(vh_seq):
            return np.nan
        # CDR H3 is typically positions ~95-102 in IMGT (variable length)
        # Rough proxy: last 15-25 residues contain CDR H3
        # For now, just flag extremely short/long VH as likely H3 issues
        return len(vh_seq)  # Placeholder - VH length correlates with H3

    df["cdr_h3_proxy"] = df["vh_sequence"].apply(estimate_cdr_h3_length)

    print(
        f"  VH length: mean={df['vh_length'].mean():.1f}, std={df['vh_length'].std():.1f}"
    )
    print(
        f"  VL length: mean={df['vl_length'].mean():.1f}, std={df['vl_length'].std():.1f}"
    )
    print(
        f"  VH charge: mean={df['vh_charge'].mean():.1f}, std={df['vh_charge'].std():.1f}\n"
    )

    return df


def calculate_zscores(df):
    """Calculate z-scores for all numeric features."""
    print("Calculating z-scores...")

    features = ["vh_length", "vl_length", "vh_charge", "vl_charge", "total_charge"]

    for feat in features:
        mean = df[feat].mean()
        std = df[feat].std()
        df[f"{feat}_zscore"] = (df[feat] - mean) / std

    # Flag outliers (|z| > 2.5)
    df["vh_length_outlier"] = df["vh_length_zscore"].abs() > 2.5
    df["vl_length_outlier"] = df["vl_length_zscore"].abs() > 2.5
    df["charge_outlier"] = df["total_charge_zscore"].abs() > 2.5

    df["any_outlier"] = (
        df["vh_length_outlier"] | df["vl_length_outlier"] | df["charge_outlier"]
    )

    n_outliers = df["any_outlier"].sum()
    print(f"  Outliers found (|z| > 2.5): {n_outliers} antibodies")
    print(f"    VH length: {df['vh_length_outlier'].sum()}")
    print(f"    VL length: {df['vl_length_outlier'].sum()}")
    print(f"    Charge: {df['charge_outlier'].sum()}\n")

    return df


def flag_known_issues(df):
    """Flag antibodies with known clinical/biological issues."""
    print("Flagging known biological/clinical issues...")

    # Known chimeric antibodies (mouse/human, higher immunogenicity)
    chimeric = [
        "muromonab",
        "cetuximab",
        "basiliximab",
        "infliximab",
        "rituximab",
        "trastuzumab",
        "abciximab",
        "daclizumab",
        "gemtuzumab",
        "alemtuzumab",
        "ibritumomab",
        "tositumomab",
        "bevacizumab",
        "panitumumab",
        "girentuximab",
    ]

    # Known discontinued/failed antibodies
    discontinued = [
        "muromonab",
        "girentuximab",
        "tabalumab",
        "abituzumab",
        "figitumumab",
        "dalotuzumab",
        "ganitumab",
        "robatumumab",
        "gemtuzumab",
        "zanolimumab",
        "epratuzumab",
        "farletuzumab",
        "lucatumumab",
        "mapatumumab",
        "tigatuzumab",
        "lexatumumab",
    ]

    # Known withdrawn antibodies
    withdrawn = ["muromonab", "efalizumab", "gemtuzumab", "natalizumab"]

    df["chimeric"] = df["id"].str.lower().isin([x.lower() for x in chimeric])
    df["discontinued"] = df["id"].str.lower().isin([x.lower() for x in discontinued])
    df["withdrawn"] = df["id"].str.lower().isin([x.lower() for x in withdrawn])

    df["known_issue"] = df["chimeric"] | df["discontinued"] | df["withdrawn"]

    print(f"  Chimeric: {df['chimeric'].sum()}")
    print(f"  Discontinued: {df['discontinued'].sum()}")
    print(f"  Withdrawn: {df['withdrawn'].sum()}")
    print(f"  Any known issue: {df['known_issue'].sum()}\n")

    return df


def identify_qc_candidates(df):
    """Identify candidates for Novo's 30-antibody removal."""
    print("=" * 80)
    print("CANDIDATES FOR QC REMOVAL (116 → 86)")
    print("=" * 80)

    # Combine outliers + known issues
    df["qc_candidate"] = df["any_outlier"] | df["known_issue"]

    candidates = df[df["qc_candidate"]].copy()
    candidates = candidates.sort_values("vh_length_zscore", key=abs, ascending=False)

    print(f"\nTotal QC candidates: {len(candidates)} antibodies")
    print("  Need to remove: 30 antibodies (116 → 86)")
    print(f"  Gap: {30 - len(candidates)} antibodies (may need additional criteria)\n")

    # Show top candidates
    print("TOP QC CANDIDATES (sorted by |VH z-score|):")
    print("-" * 80)

    cols = [
        "id",
        "label",
        "elisa_flags",
        "vh_length",
        "vh_length_zscore",
        "vl_length",
        "vl_length_zscore",
        "total_charge",
        "total_charge_zscore",
        "chimeric",
        "discontinued",
        "withdrawn",
    ]

    # Get full candidates dataframe for flags lookup
    for idx, row in candidates.iterrows():
        flags = []
        if row["vh_length_outlier"]:
            flags.append(f"VH_z={row['vh_length_zscore']:.2f}")
        if row["vl_length_outlier"]:
            flags.append(f"VL_z={row['vl_length_zscore']:.2f}")
        if row["charge_outlier"]:
            flags.append(f"Charge_z={row['total_charge_zscore']:.2f}")
        if row["chimeric"]:
            flags.append("CHIMERIC")
        if row["discontinued"]:
            flags.append("DISCONTINUED")
        if row["withdrawn"]:
            flags.append("WITHDRAWN")

        label_str = "specific" if row["label"] == 0 else "non-spec"
        print(
            f"{row['id']:20s} | {label_str:8s} | ELISA={row['elisa_flags']} | "
            f"VH={row['vh_length']:3d} (z={row['vh_length_zscore']:+.2f}) | "
            f"VL={row['vl_length']:3d} | {', '.join(flags)}"
        )

    return df, candidates


def analyze_by_label(df, candidates):
    """Analyze QC candidates by label (specific vs non-specific)."""
    print("\n" + "=" * 80)
    print("QC CANDIDATE BREAKDOWN BY LABEL")
    print("=" * 80)

    cand_specific = candidates[candidates["label"] == 0]
    cand_nonspec = candidates[candidates["label"] == 1]

    print(f"\nSpecific (label=0) candidates: {len(cand_specific)}")
    print(f"Non-specific (label=1) candidates: {len(cand_nonspec)}")

    print("\nCurrent distribution (116): 94 specific / 22 non-specific")
    print("Target distribution (86): 59 specific / 27 non-specific")
    print("Need to remove: 35 specific / -5 non-specific (?!)")
    print("\n⚠️  PROBLEM: Novo has MORE non-specific (27 vs 22)")
    print(
        "    This suggests they RECLASSIFIED 5 antibodies from specific → non-specific"
    )
    print("    OR used different ELISA thresholding")

    return cand_specific, cand_nonspec


def export_results(df, candidates):
    """Export results to CSV."""
    output_dir = "test_datasets/jain"

    # Full dataset with z-scores
    zscore_path = f"{output_dir}/jain_ELISA_ONLY_116_with_zscores.csv"
    df.to_csv(zscore_path, index=False)
    print(f"\n✓ Saved: {zscore_path}")

    # QC candidates only
    candidates_path = f"{output_dir}/jain_116_qc_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    print(f"✓ Saved: {candidates_path}")


def main():
    print("=" * 80)
    print("Z-Score Analysis: Jain 116-Antibody ELISA-Only Set")
    print("=" * 80)
    print("Goal: Reverse-engineer Novo's 30-antibody QC removal (116 → 86)")
    print("=" * 80)
    print()

    # Load data
    df = load_data()

    # Calculate features
    df = calculate_sequence_features(df)

    # Z-scores
    df = calculate_zscores(df)

    # Known issues
    df = flag_known_issues(df)

    # Identify QC candidates
    df, candidates = identify_qc_candidates(df)

    # Analyze by label
    cand_specific, cand_nonspec = analyze_by_label(df, candidates)

    # Export
    export_results(df, candidates)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Total antibodies: 116")
    print(f"QC candidates found: {len(candidates)}")
    print(f"  Specific: {len(cand_specific)}")
    print(f"  Non-specific: {len(cand_nonspec)}")
    print(f"\nNeed to identify: {30 - len(candidates)} more candidates")
    print("\n⚠️  Key issue: Novo has 27 non-specific (we have 22)")
    print("   This suggests threshold change OR reclassification")
    print("=" * 80)


if __name__ == "__main__":
    main()
