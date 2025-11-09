#!/usr/bin/env python3
"""
Jain Dataset Fragment Extraction Script

Processes the Jain dataset to extract all 16 antibody fragment types
using ANARCI (IMGT numbering scheme) following Sakhnini et al. 2025 methodology.

SOURCE OF TRUTH: jain_with_private_elisa_FULL.csv
- Uses ELISA-based labeling (elisa_flags: 0-6 range)
- Distribution: 94 specific / 22 non-specific / 21 mild
- REPLACES old fragments that used flags_total (paper-based, 67/27/43)

Labeling Rule (ELISA-based):
- elisa_flags = 0       → Specific (label 0)
- elisa_flags = 1-3     → Mild (label NaN)
- elisa_flags >= 4      → Non-specific (label 1)

Fragments extracted:
1. VH (full heavy variable domain)
2. VL (full light variable domain)
3. H-CDR1, H-CDR2, H-CDR3
4. L-CDR1, L-CDR2, L-CDR3
5. H-CDRs (concatenated H-CDR1+2+3)
6. L-CDRs (concatenated L-CDR1+2+3)
7. H-FWRs (concatenated H-FWR1+2+3+4)
8. L-FWRs (concatenated L-FWR1+2+3+4)
9. VH+VL (paired variable domains)
10. All-CDRs (H-CDRs + L-CDRs)
11. All-FWRs (H-FWRs + L-FWRs)
12. Full (VH + VL, alias for compatibility)

Date: 2025-11-06
Issue: Jain label discrepancy fix (38.7% error rate in old fragments)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import riot_na
from tqdm.auto import tqdm

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def jain_label_from_elisa(elisa_flags: int | float | None) -> int | None:
    """
    Convert ELISA flags to binary label.

    SSOT Labeling Rule:
    - 0 flags       → Specific (label 0)
    - 1-3 flags     → Mild (label NaN) - excluded from training
    - 4-6 flags     → Non-specific (label 1)

    Args:
        elisa_flags: Integer count of ELISA flags (0-6)

    Returns:
        0, 1, or None (NaN for mild)
    """
    if pd.isna(elisa_flags):
        return None

    elisa_flags = int(elisa_flags)

    if elisa_flags == 0:
        return 0
    elif 1 <= elisa_flags <= 3:
        return None  # Mild - will become NaN in CSV
    else:  # >= 4
        return 1


def annotate_sequence(seq_id: str, sequence: str, chain: str) -> dict[str, str] | None:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    assert chain in ("H", "L"), "chain must be 'H' or 'L'"

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments (gap-free sequences)
        fragments = {
            f"full_seq_{chain}": annotation.sequence_aa,
            f"fwr1_aa_{chain}": annotation.fwr1_aa,
            f"cdr1_aa_{chain}": annotation.cdr1_aa,
            f"fwr2_aa_{chain}": annotation.fwr2_aa,
            f"cdr2_aa_{chain}": annotation.cdr2_aa,
            f"fwr3_aa_{chain}": annotation.fwr3_aa,
            f"cdr3_aa_{chain}": annotation.cdr3_aa,
            f"fwr4_aa_{chain}": annotation.fwr4_aa,
        }

        # Create concatenated fragments
        fragments[f"cdrs_{chain}"] = "".join(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        )

        fragments[f"fwrs_{chain}"] = "".join(
            [
                fragments[f"fwr1_aa_{chain}"],
                fragments[f"fwr2_aa_{chain}"],
                fragments[f"fwr3_aa_{chain}"],
                fragments[f"fwr4_aa_{chain}"],
            ]
        )

        return fragments

    except Exception as e:
        print(f"Warning: Failed to annotate {seq_id} ({chain}): {e}", file=sys.stderr)
        return None


def process_jain_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process Jain ELISA-based CSV to extract all fragments.

    Args:
        csv_path: Path to jain_with_private_elisa_FULL.csv

    Returns:
        DataFrame with all fragments and metadata
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total antibodies: {len(df)}")
    print("  Annotating sequences with ANARCI (IMGT scheme)...")

    # Verify source file is correct
    if "elisa_flags" not in df.columns:
        print("ERROR: Source file missing 'elisa_flags' column!")
        print("This script requires jain_with_private_elisa_FULL.csv as input.")
        sys.exit(1)

    results = []
    failures = []

    for _idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
        # Annotate heavy chain
        heavy_frags = annotate_sequence(f"{row['id']}_VH", row["vh_sequence"], "H")

        # Annotate light chain
        light_frags = annotate_sequence(f"{row['id']}_VL", row["vl_sequence"], "L")

        if heavy_frags is None or light_frags is None:
            print(f"  Skipping {row['id']} - annotation failed")
            failures.append(row["id"])
            continue

        # Apply ELISA-based labeling (SSOT)
        label = jain_label_from_elisa(row["elisa_flags"])

        # Combine all fragments and metadata
        result = {
            "id": row["id"],
            "label": label,
            "elisa_flags": row["elisa_flags"],
            "source": "jain2017_pnas",
        }

        result.update(heavy_frags)
        result.update(light_frags)

        # Create paired/combined fragments
        result["vh_vl"] = result["full_seq_H"] + result["full_seq_L"]
        result["all_cdrs"] = result["cdrs_H"] + result["cdrs_L"]
        result["all_fwrs"] = result["fwrs_H"] + result["fwrs_L"]

        results.append(result)

    df_annotated = pd.DataFrame(results)

    print(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")

    if failures:
        print(f"  Failures: {len(failures)}")
        print(f"  Failed IDs: {failures}")

    # Verify label distribution matches SSOT
    print("\n  Label distribution (ELISA-based):")
    print(f"    Specific (0):     {(df_annotated['label'] == 0).sum()}")
    print(f"    Non-specific (1): {(df_annotated['label'] == 1).sum()}")
    print(f"    Mild (NaN):       {df_annotated['label'].isna().sum()}")

    expected = {"specific": 94, "nonspecific": 22, "mild": 21}
    actual_specific = (df_annotated["label"] == 0).sum()
    actual_nonspecific = (df_annotated["label"] == 1).sum()
    actual_mild = df_annotated["label"].isna().sum()

    if (
        actual_specific == expected["specific"]
        and actual_nonspecific == expected["nonspecific"]
        and actual_mild == expected["mild"]
    ):
        print("    ✓ Distribution matches SSOT expectations!")
    else:
        print(
            f"    ⚠ WARNING: Expected {expected['specific']}/{expected['nonspecific']}/{expected['mild']}"
        )

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create separate CSV files for each fragment type.

    Following the 16-fragment methodology from Sakhnini et al. 2025.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all 16 fragment types (matching shehata/boughter patterns)
    fragments = {
        # 1-2: Full variable domains
        "VH_only": ("full_seq_H", "vh_sequence"),
        "VL_only": ("full_seq_L", "vl_sequence"),
        # 3-5: Heavy CDRs
        "H-CDR1": ("cdr1_aa_H", "h_cdr1"),
        "H-CDR2": ("cdr2_aa_H", "h_cdr2"),
        "H-CDR3": ("cdr3_aa_H", "h_cdr3"),
        # 6-8: Light CDRs
        "L-CDR1": ("cdr1_aa_L", "l_cdr1"),
        "L-CDR2": ("cdr2_aa_L", "l_cdr2"),
        "L-CDR3": ("cdr3_aa_L", "l_cdr3"),
        # 9-10: Concatenated CDRs
        "H-CDRs": ("cdrs_H", "h_cdrs"),
        "L-CDRs": ("cdrs_L", "l_cdrs"),
        # 11-12: Concatenated FWRs
        "H-FWRs": ("fwrs_H", "h_fwrs"),
        "L-FWRs": ("fwrs_L", "l_fwrs"),
        # 13: Paired variable domains
        "VH+VL": ("vh_vl", "paired_variable_domains"),
        # 14-15: All CDRs/FWRs
        "All-CDRs": ("all_cdrs", "all_cdrs"),
        "All-FWRs": ("all_fwrs", "all_fwrs"),
        # 16: Full (alias for VH+VL for compatibility)
        "Full": ("vh_vl", "full_sequence"),
    }

    print(f"\nCreating {len(fragments)} fragment-specific CSV files...")

    # NOTE: Metadata moved to manifest.yml to maintain CSV compatibility
    # All fragments are standard CSVs (no comment headers) for HuggingFace compatibility

    for fragment_name, (column_name, _sequence_alias) in fragments.items():
        output_path = output_dir / f"{fragment_name}_jain.csv"

        # Create fragment-specific CSV with standardized column names
        fragment_df = pd.DataFrame(
            {
                "id": df["id"],
                "sequence": df[column_name],
                "label": df["label"],
                "elisa_flags": df["elisa_flags"],
                "source": df["source"],
            }
        )

        # Write standard CSV (matches Shehata/Harvey format)
        fragment_df.to_csv(output_path, index=False)

        # Show stats
        min_len = fragment_df["sequence"].str.len().min()
        max_len = fragment_df["sequence"].str.len().max()

        specific = (fragment_df["label"] == 0).sum()
        nonspecific = (fragment_df["label"] == 1).sum()
        mild = fragment_df["label"].isna().sum()

        print(
            f"  ✓ {fragment_name:12s} → {output_path.name:30s} "
            f"(len: {min_len:3d}-{max_len:3d} aa, {specific}/{nonspecific}/{mild})"
        )

    print(f"\n✓ All fragments saved to: {output_dir}/")


def create_manifest(output_dir: Path, source_path: Path, script_path: Path) -> None:
    """
    Create manifest.yml for provenance tracking.

    Args:
        output_dir: Directory containing fragments
        source_path: Path to source CSV
        script_path: Path to this script
    """
    import hashlib

    root = Path.cwd().resolve()

    # Calculate SHA256 of source file
    with open(source_path, "rb") as f:
        source_sha = hashlib.sha256(f.read()).hexdigest()

    source_abs = source_path.resolve()
    script_abs = script_path.resolve()

    try:
        source_rel = source_abs.relative_to(root)
    except ValueError:
        source_rel = source_abs

    try:
        script_rel = script_abs.relative_to(root)
    except ValueError:
        script_rel = script_abs

    manifest_content = f"""# Jain Dataset Fragment Provenance Manifest
# Generated: 2025-11-06
# Format: Standard CSV (no comment headers) for HuggingFace compatibility

source_file: {source_rel}
source_sha256: {source_sha}
script: {script_rel}

csv_format:
  type: standard
  columns: [id, sequence, label, elisa_flags, source]
  comment_headers: false
  note: All fragments are plain CSVs compatible with pd.read_csv(path)

labeling_rule: ELISA-based
  - elisa_flags = 0     → Specific (label 0)
  - elisa_flags = 1-3   → Mild (label NaN) - excluded from training
  - elisa_flags >= 4    → Non-specific (label 1)

expected_counts:
  total: 137
  specific: 94
  nonspecific: 22
  mild: 21
  test_set: 116  # excludes mild

fragments_include_nan_labels: true
  note: |
    Fragments contain all 137 antibodies including 21 "mild" with label=NaN.
    For training, use canonical/ files which exclude mild antibodies.
    To filter: df = df[df['label'].notna()]

compatibility:
  - Matches Shehata/Harvey fragment format (standard CSVs)
  - Compatible with HuggingFace datasets library
  - No special read parameters required (pd.read_csv works directly)

note: |
  This replaces the old fragments (67/27/43 distribution) that used
  flags_total labeling from the paper. The ELISA-based labeling is
  the single source of truth (SSOT) for all jain data.

  Previous version (commit 09d6121) included comment headers that broke
  standard CSV parsers. Current version uses standard CSV format.
"""

    manifest_path = output_dir / "manifest.yml"
    with open(manifest_path, "w") as f:
        f.write(manifest_content)

    print(f"\n✓ Manifest created: {manifest_path}")


def main() -> int:
    """Main processing pipeline."""
    # Paths
    csv_path = Path("test_datasets/jain/processed/jain_with_private_elisa_FULL.csv")
    output_dir = Path("test_datasets/jain/fragments")
    script_path = Path(__file__)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print(
            "This script requires jain_with_private_elisa_FULL.csv (ELISA-based SSOT)."
        )
        return 1

    print("=" * 70)
    print("Jain Dataset: Fragment Extraction (ELISA-based SSOT)")
    print("=" * 70)
    print(f"\nInput:  {csv_path}")
    print(f"Output: {output_dir}/")
    print("Method: ANARCI (IMGT numbering scheme)")
    print("Labels: ELISA flags (0→specific, 1-3→mild, ≥4→non-specific)")
    print()
    print(
        "⚠  This will REPLACE old fragments (67/27/43) with correct labels (94/22/21)"
    )
    print()

    # Process dataset
    df_annotated = process_jain_dataset(str(csv_path))

    # Create fragment CSVs
    create_fragment_csvs(df_annotated, output_dir)

    # Create manifest for provenance
    create_manifest(output_dir, csv_path, script_path)

    # Validation summary
    print("\n" + "=" * 70)
    print("Fragment Extraction Summary")
    print("=" * 70)

    print(f"\nAnnotated antibodies: {len(df_annotated)}")
    print("Label distribution (ELISA-based SSOT):")

    specific = (df_annotated["label"] == 0).sum()
    nonspecific = (df_annotated["label"] == 1).sum()
    mild = df_annotated["label"].isna().sum()

    print(f"  Specific (0):     {specific} ({specific / len(df_annotated) * 100:.1f}%)")
    print(
        f"  Non-specific (1): {nonspecific} ({nonspecific / len(df_annotated) * 100:.1f}%)"
    )
    print(f"  Mild (NaN):       {mild} ({mild / len(df_annotated) * 100:.1f}%)")
    print(f"  Test set:         {specific + nonspecific} (excludes mild)")

    print("\nFragment files created: 16")
    print(f"Output directory: {output_dir.absolute()}")

    print("\n" + "=" * 70)
    print("✓ Jain Fragment Regeneration Complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Run integration tests to verify the 94/22/21 distribution")
    print(
        "  2. Commit changes with a clear provenance message "
        "(legacy 67/27/43 fragments remain available in git history at commit 09d6121)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
