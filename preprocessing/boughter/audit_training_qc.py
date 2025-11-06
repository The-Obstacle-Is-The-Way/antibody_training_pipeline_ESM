#!/usr/bin/env python3
"""
Comprehensive QC Audit of Boughter Training Set
================================================

Purpose: Search for ANY potential QC issues in our final 914 training sequences
that might explain the 3.5% accuracy gap with Novo.

Checks:
1. Stop codons (*)
2. Unknown amino acids (X)
3. Gap characters (-)
4. Non-standard amino acids
5. Unusual sequence lengths (too short/long)
6. Repeated residues (homopolymers)
7. Suspicious CDR lengths
8. Empty/whitespace sequences

Date: 2025-11-04
"""

from pathlib import Path

import pandas as pd

# Paths
BOUGHTER_DIR = Path("train_datasets/boughter")
TRAINING_FILE = BOUGHTER_DIR / "canonical" / "VH_only_boughter_training.csv"
FULL_FILE = BOUGHTER_DIR / "annotated" / "VH_only_boughter.csv"

# Standard amino acids (20 standard + X for unknown)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYX")  # X is semi-acceptable


def check_stop_codons(df, seq_col="sequence"):
    """Check for stop codons (*) in sequences"""
    has_stop = df[seq_col].str.contains(r"\*", na=False, regex=True)
    return df[has_stop], has_stop.sum()


def check_gap_characters(df, seq_col="sequence"):
    """Check for gap characters (-) in sequences"""
    has_gaps = df[seq_col].str.contains("-", na=False)
    return df[has_gaps], has_gaps.sum()


def check_unknown_aa(df, seq_col="sequence"):
    """Check for unknown amino acids (X) in sequences"""
    has_X = df[seq_col].str.contains("X", na=False)
    return df[has_X], has_X.sum()


def check_non_standard_aa(df, seq_col="sequence"):
    """Check for any non-standard amino acids (not in ACDEFGHIKLMNPQRSTVWYX)"""

    def has_non_standard(seq):
        if pd.isna(seq):
            return False
        return any(aa not in EXTENDED_AA for aa in str(seq))

    non_standard_mask = df[seq_col].apply(has_non_standard)
    return df[non_standard_mask], non_standard_mask.sum()


def check_sequence_lengths(df, seq_col="sequence", min_len=95, max_len=500):
    """Check for sequences that are too short or too long"""
    lengths = df[seq_col].str.len()
    too_short = lengths < min_len
    too_long = lengths > max_len

    return {
        "too_short": (df[too_short], too_short.sum()),
        "too_long": (df[too_long], too_long.sum()),
        "length_stats": lengths.describe(),
    }


def check_homopolymers(df, seq_col="sequence", min_repeat=7):
    """Check for long runs of repeated amino acids (e.g., AAAAAAA)"""

    def has_homopolymer(seq):
        if pd.isna(seq):
            return False
        # Check for any amino acid repeated 7+ times
        return any(aa * min_repeat in str(seq) for aa in STANDARD_AA)

    homopolymer_mask = df[seq_col].apply(has_homopolymer)
    return df[homopolymer_mask], homopolymer_mask.sum()


def check_empty_sequences(df, seq_col="sequence"):
    """Check for empty or whitespace-only sequences"""
    is_empty = df[seq_col].isna() | (df[seq_col].str.strip() == "")
    return df[is_empty], is_empty.sum()


def check_cdr_lengths(df):
    """Check for suspiciously short/long CDRs"""
    issues = []

    # CDR-H1: typically 10-12 residues (IMGT 27-38)
    if "cdr1_aa_H" in df.columns:
        h1_lengths = df["cdr1_aa_H"].str.len()
        h1_too_short = h1_lengths < 8
        h1_too_long = h1_lengths > 15
        if h1_too_short.sum() > 0:
            issues.append(
                ("CDR-H1 too short (<8)", df[h1_too_short], h1_too_short.sum())
            )
        if h1_too_long.sum() > 0:
            issues.append(("CDR-H1 too long (>15)", df[h1_too_long], h1_too_long.sum()))

    # CDR-H2: typically 8-10 residues (IMGT 56-65)
    if "cdr2_aa_H" in df.columns:
        h2_lengths = df["cdr2_aa_H"].str.len()
        h2_too_short = h2_lengths < 6
        h2_too_long = h2_lengths > 12
        if h2_too_short.sum() > 0:
            issues.append(
                ("CDR-H2 too short (<6)", df[h2_too_short], h2_too_short.sum())
            )
        if h2_too_long.sum() > 0:
            issues.append(("CDR-H2 too long (>12)", df[h2_too_long], h2_too_long.sum()))

    # CDR-H3: highly variable (typically 6-20 residues, IMGT 105-117)
    if "cdr3_aa_H" in df.columns:
        h3_lengths = df["cdr3_aa_H"].str.len()
        h3_too_short = h3_lengths < 4  # Very short
        h3_too_long = h3_lengths > 25  # Very long
        if h3_too_short.sum() > 0:
            issues.append(
                ("CDR-H3 too short (<4)", df[h3_too_short], h3_too_short.sum())
            )
        if h3_too_long.sum() > 0:
            issues.append(("CDR-H3 too long (>25)", df[h3_too_long], h3_too_long.sum()))

    return issues


def audit_dataset(file_path, dataset_name):
    """Run comprehensive QC audit on a dataset"""
    print(f"\n{'=' * 80}")
    print(f"QC AUDIT: {dataset_name}")
    print(f"File: {file_path}")
    print(f"{'=' * 80}\n")

    # Load data
    df = pd.read_csv(file_path, comment="#")
    print(f"Total sequences: {len(df)}")

    # Determine sequence column
    seq_col = "sequence" if "sequence" in df.columns else "heavy_seq"
    print(f"Sequence column: {seq_col}\n")

    # Check if 'id' column exists, if not use index
    has_id = "id" in df.columns
    if not has_id:
        print("Note: No 'id' column found, using row index\n")
        df["_index"] = df.index

    # Track all issues
    total_issues = 0
    flagged_sequences = set()

    # Determine ID column
    id_col = "id" if has_id else "_index"

    # 1. Stop codons
    print("1. Checking for stop codons (*)...")
    stop_df, stop_count = check_stop_codons(df, seq_col)
    if stop_count > 0:
        print(f"   ❌ FOUND {stop_count} sequences with stop codons")
        flagged_sequences.update(stop_df[id_col].tolist())
        total_issues += stop_count
        print(f"   Examples (row indices): {stop_df[id_col].head(3).tolist()}")
    else:
        print("   ✅ No stop codons found")

    # 2. Gap characters
    print("\n2. Checking for gap characters (-)...")
    gap_df, gap_count = check_gap_characters(df, seq_col)
    if gap_count > 0:
        print(f"   ❌ FOUND {gap_count} sequences with gap characters")
        flagged_sequences.update(gap_df[id_col].tolist())
        total_issues += gap_count
        print(f"   Examples (row indices): {gap_df[id_col].head(3).tolist()}")
    else:
        print("   ✅ No gap characters found")

    # 3. Unknown amino acids (X)
    print("\n3. Checking for unknown amino acids (X)...")
    x_df, x_count = check_unknown_aa(df, seq_col)
    if x_count > 0:
        print(f"   ❌ FOUND {x_count} sequences with X")
        flagged_sequences.update(x_df[id_col].tolist())
        total_issues += x_count
        print(f"   Examples (row indices): {x_df[id_col].head(3).tolist()}")
        # Show first 10 with X positions
        print("\n   First 10 sequences with X:")
        for _idx, row in x_df.head(10).iterrows():
            seq = str(row[seq_col])
            x_positions = [i for i, c in enumerate(seq) if c == "X"]
            print(
                f"      Row {row[id_col]}: X at positions {x_positions[:5]}... (seq preview: {seq[:50]}...)"
            )
    else:
        print("   ✅ No X amino acids found")

    # 4. Non-standard amino acids
    print("\n4. Checking for non-standard amino acids...")
    non_std_df, non_std_count = check_non_standard_aa(df, seq_col)
    if non_std_count > 0:
        print(f"   ❌ FOUND {non_std_count} sequences with non-standard amino acids")
        flagged_sequences.update(non_std_df[id_col].tolist())
        total_issues += non_std_count
        print(f"   Examples (row indices): {non_std_df[id_col].head(3).tolist()}")
        # Show what characters
        for _idx, row in non_std_df.head(3).iterrows():
            seq = str(row[seq_col])
            bad_chars = [c for c in seq if c not in EXTENDED_AA]
            print(f"      Row {row[id_col]}: {set(bad_chars)}")
    else:
        print("   ✅ All amino acids are standard")

    # 5. Empty sequences
    print("\n5. Checking for empty sequences...")
    empty_df, empty_count = check_empty_sequences(df, seq_col)
    if empty_count > 0:
        print(f"   ❌ FOUND {empty_count} empty sequences")
        flagged_sequences.update(empty_df[id_col].tolist())
        total_issues += empty_count
    else:
        print("   ✅ No empty sequences found")

    # 6. Sequence lengths
    print("\n6. Checking sequence lengths...")
    length_results = check_sequence_lengths(df, seq_col, min_len=95, max_len=500)
    print("   Length statistics:")
    print(f"   {length_results['length_stats']}")

    too_short_df, too_short_count = length_results["too_short"]
    too_long_df, too_long_count = length_results["too_long"]

    if too_short_count > 0:
        print(f"\n   ❌ FOUND {too_short_count} sequences too short (<95 aa)")
        flagged_sequences.update(too_short_df[id_col].tolist())
        total_issues += too_short_count
        print(f"   Examples (row indices): {too_short_df[id_col].head(3).tolist()}")
    else:
        print("   ✅ No sequences too short (<95 aa)")

    if too_long_count > 0:
        print(f"\n   ❌ FOUND {too_long_count} sequences too long (>500 aa)")
        flagged_sequences.update(too_long_df[id_col].tolist())
        total_issues += too_long_count
    else:
        print("   ✅ No sequences too long (>500 aa)")

    # 7. Homopolymers
    print("\n7. Checking for homopolymers (7+ repeated amino acids)...")
    homo_df, homo_count = check_homopolymers(df, seq_col, min_repeat=7)
    if homo_count > 0:
        print(f"   ⚠️  FOUND {homo_count} sequences with homopolymers")
        flagged_sequences.update(homo_df[id_col].tolist())
        print(f"   Examples (row indices): {homo_df[id_col].head(3).tolist()}")
        # Show the homopolymers
        for _idx, row in homo_df.head(3).iterrows():
            seq = str(row[seq_col])
            for aa in STANDARD_AA:
                if aa * 7 in seq:
                    print(f"      Row {row[id_col]}: contains {aa}×7+")
    else:
        print("   ✅ No long homopolymers found")

    # 8. CDR lengths (if CDR columns exist)
    print("\n8. Checking CDR lengths...")
    cdr_issues = check_cdr_lengths(df)
    if cdr_issues:
        for issue_type, issue_df, issue_count in cdr_issues:
            print(f"   ⚠️  FOUND {issue_count} sequences: {issue_type}")
            print(
                f"   Examples (row indices): {issue_df[id_col].head(3).tolist() if id_col in issue_df.columns else 'N/A'}"
            )
    else:
        print("   ✅ All CDR lengths look normal")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total sequences: {len(df)}")
    print(f"Unique sequences with issues: {len(flagged_sequences)}")
    print(f"Total issue count: {total_issues}")

    if len(flagged_sequences) > 0:
        print("\n❌ QC ISSUES FOUND - Novo may have filtered these sequences!")
        print(f"\nFlagged sequence {'IDs' if has_id else 'row indices'}:")
        for seq_id in sorted(flagged_sequences)[:20]:  # Show first 20
            if has_id:
                print(f"   - {seq_id}")
            else:
                print(f"   - Row {seq_id}")
        if len(flagged_sequences) > 20:
            print(f"   ... and {len(flagged_sequences) - 20} more")

        # Calculate potential impact
        impact_pct = (len(flagged_sequences) / len(df)) * 100
        print(
            f"\nPotential impact: {len(flagged_sequences)}/{len(df)} ({impact_pct:.1f}%)"
        )
        print("If Novo filtered these, their training set would be:")
        print(f"   {len(df) - len(flagged_sequences)} sequences")
    else:
        print("\n✅ NO QC ISSUES FOUND - Dataset is clean!")

    return {
        "total_sequences": len(df),
        "flagged_sequences": flagged_sequences,
        "total_issues": total_issues,
        "stop_codons": stop_count,
        "gap_chars": gap_count,
        "unknown_aa": x_count,
        "non_standard": non_std_count,
        "empty": empty_count,
        "too_short": too_short_count,
        "too_long": too_long_count,
        "homopolymers": homo_count,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BOUGHTER TRAINING SET QC AUDIT")
    print("Searching for any QC issues that might explain Novo's 3.5% accuracy gap")
    print("=" * 80)

    # Audit training set (914 sequences)
    training_results = audit_dataset(
        TRAINING_FILE, "Boughter Training Set (914 sequences)"
    )

    # Also audit full set for comparison (1,065 sequences)
    print("\n\n")
    full_results = audit_dataset(FULL_FILE, "Boughter Full Set (1,065 sequences)")

    # Final comparison
    print(f"\n\n{'=' * 80}")
    print("FINAL COMPARISON")
    print(f"{'=' * 80}")
    print("\nTraining Set (914 sequences):")
    print(
        f"   Flagged: {len(training_results['flagged_sequences'])} ({len(training_results['flagged_sequences']) / training_results['total_sequences'] * 100:.1f}%)"
    )
    print("\nFull Set (1,065 sequences):")
    print(
        f"   Flagged: {len(full_results['flagged_sequences'])} ({len(full_results['flagged_sequences']) / full_results['total_sequences'] * 100:.1f}%)"
    )

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")

    if len(training_results["flagged_sequences"]) == 0:
        print("✅ Our training set is CLEAN - no QC issues found")
        print("✅ The 3.5% gap is NOT due to missing QC filtering")
        print(
            "✅ Gap is likely due to hyperparameters, random seed, or ESM embedding differences"
        )
    else:
        print(
            f"❌ Found {len(training_results['flagged_sequences'])} sequences with potential QC issues"
        )
        print("❌ Novo may have filtered these, explaining part of the accuracy gap")
        print("❌ Recommend additional QC filtering to match Novo's methodology")
