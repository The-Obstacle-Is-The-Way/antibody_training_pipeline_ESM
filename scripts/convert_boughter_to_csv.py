#!/usr/bin/env python3
"""
Boughter Dataset Stage 1: DNA Translation & CSV Conversion

Processes raw Boughter DNA FASTA files, translates to protein, applies
Novo Nordisk flagging strategy, and outputs combined CSV.

Usage:
    python3 preprocessing/convert_boughter_to_csv.py

Outputs:
    test_datasets/boughter.csv - Combined dataset with Novo flagging
    test_datasets/boughter/translation_failures.log - Failed sequences
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


def parse_fasta_dna(fasta_path: Path) -> List[str]:
    """
    Parse DNA FASTA file and return list of DNA sequences.

    Args:
        fasta_path: Path to FASTA file (.txt or .dat)

    Returns:
        List of DNA sequence strings
    """
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def parse_numreact_flags(flag_path: Path) -> List[int]:
    """
    Parse NumReact flag file (0-7 scale with header).

    Format:
        reacts
        1
        0
        5

    Returns:
        List of integer flags (0-7)
    """
    lines = flag_path.read_text().strip().splitlines()

    # Remove header if present
    if lines[0].lower() == "reacts":
        lines = lines[1:]

    return [int(line.strip()) for line in lines]


def parse_yn_flags(flag_path: Path) -> List[int]:
    """
    Parse Y/N flag file and convert to NumReact equivalent.

    Y (polyreactive) -> 7 (max flags)
    N (non-polyreactive) -> 0 (no flags)

    Returns:
        List of integer flags
    """
    lines = flag_path.read_text().strip().splitlines()
    flags = []

    for line in lines:
        line = line.strip().upper()
        if line == "Y":
            flags.append(7)  # Treat Y as maximally polyreactive
        elif line == "N":
            flags.append(0)  # Treat N as specific
        else:
            raise ValueError(f"Invalid Y/N flag: {line}")

    return flags


def translate_dna_to_protein(dna_seq: str) -> Optional[str]:
    """
    Translate DNA sequence to protein using standard genetic code.

    Args:
        dna_seq: DNA nucleotide sequence string

    Returns:
        Translated amino acid sequence, or None if translation fails
    """
    try:
        # Clean sequence (remove any N's or non-standard bases)
        dna_seq = dna_seq.upper()

        # Create Bio.Seq object
        seq = Seq(dna_seq)

        # Translate using standard genetic code (table=1)
        # to_stop=False to get full sequence
        protein = seq.translate(table=1, to_stop=False)

        return str(protein)

    except Exception as e:
        print(f"Translation failed: {e}")
        return None


def validate_translation(protein_seq: str) -> bool:
    """
    Validate that translation produced reasonable antibody sequence.

    Strict validation to reject junk sequences before ANARCI annotation.
    ANARCI will fail on sequences with stop codons or excessive unknowns.

    Checks:
    1. Length in expected range for variable domains (90-200 aa)
    2. No stop codons (*)
    3. Limited unknown amino acids (<5% X's from N bases)
    4. Contains standard amino acids

    Returns:
        True if valid, False otherwise
    """
    if not protein_seq:
        return False

    # Check length - variable domains are typically 110-130 aa
    # Allow 90-200 to handle signal peptides and slight variations
    if len(protein_seq) < 90 or len(protein_seq) > 200:
        return False

    # Reject sequences with stop codons
    if "*" in protein_seq:
        return False

    # Reject sequences with excessive unknown amino acids (X from N bases)
    x_count = protein_seq.count("X")
    x_ratio = x_count / len(protein_seq)
    if x_ratio > 0.05:  # Reject if >5% unknown
        return False

    # Check that sequence contains mostly standard amino acids
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_chars = sum(1 for aa in protein_seq if aa in standard_aa or aa == "X")
    if valid_chars < len(protein_seq) * 0.95:
        return False

    return True


def apply_novo_flagging(num_flags: int) -> Dict[str, any]:
    """
    Apply Novo Nordisk flagging strategy from Sakhnini et al. 2025.

    Rules:
    - 0 flags → label 0 (specific), INCLUDE in training
    - 1-3 flags → EXCLUDE from training (mild polyreactivity)
    - >3 flags (4-7) → label 1 (non-specific), INCLUDE in training

    Args:
        num_flags: Number of ELISA flags (0-7)

    Returns:
        Dictionary with label, category, and inclusion status
    """
    if num_flags == 0:
        return {"label": 0, "flag_category": "specific", "include_in_training": True}
    elif 1 <= num_flags <= 3:
        return {"label": None, "flag_category": "mild", "include_in_training": False}
    elif num_flags >= 4:
        return {
            "label": 1,
            "flag_category": "non_specific",
            "include_in_training": True,
        }
    else:
        raise ValueError(f"Invalid flag count: {num_flags}")


def process_subset(
    subset_name: str,
    heavy_path: Path,
    light_path: Path,
    flag_path: Path,
    flag_format: str,
) -> List[Dict]:
    """
    Process a single subset: translate DNA, pair sequences, apply flagging.

    Args:
        subset_name: Name of subset (e.g., 'flu', 'hiv_nat')
        heavy_path: Path to heavy chain DNA FASTA
        light_path: Path to light chain DNA FASTA
        flag_path: Path to flag file
        flag_format: 'numreact' or 'yn'

    Returns:
        List of dictionaries with processed antibody data
    """
    print(f"\nProcessing subset: {subset_name}")

    # Parse input files
    heavy_dna = parse_fasta_dna(heavy_path)
    light_dna = parse_fasta_dna(light_path)

    if flag_format == "numreact":
        flags = parse_numreact_flags(flag_path)
    elif flag_format == "yn":
        flags = parse_yn_flags(flag_path)
    else:
        raise ValueError(f"Unknown flag format: {flag_format}")

    # Validate counts match
    counts = [len(heavy_dna), len(light_dna), len(flags)]
    if len(set(counts)) != 1:
        raise ValueError(
            f"{subset_name}: Sequence count mismatch - "
            f"Heavy: {counts[0]}, Light: {counts[1]}, Flags: {counts[2]}"
        )

    print(f"  Sequences: {len(heavy_dna)}")

    results = []
    failures = []

    for idx in range(len(heavy_dna)):
        # Generate sequential ID
        seq_id = f"{subset_name}_{idx + 1:06d}"

        # Translate DNA → protein
        heavy_protein = translate_dna_to_protein(heavy_dna[idx])
        light_protein = translate_dna_to_protein(light_dna[idx])

        # Validate translations
        if not heavy_protein or not light_protein:
            failures.append(f"{seq_id}: Translation failed")
            continue

        if not validate_translation(heavy_protein) or not validate_translation(
            light_protein
        ):
            failures.append(f"{seq_id}: Invalid protein sequence")
            continue

        # Apply Novo flagging strategy
        flagging = apply_novo_flagging(flags[idx])

        # Create result record
        results.append(
            {
                "id": seq_id,
                "subset": subset_name,
                "heavy_seq": heavy_protein,
                "light_seq": light_protein,
                "num_flags": flags[idx],
                "flag_category": flagging["flag_category"],
                "label": flagging["label"],
                "include_in_training": flagging["include_in_training"],
                "source": "boughter2020",
            }
        )

    print(f"  Successful: {len(results)}")
    print(f"  Failures: {len(failures)}")

    if failures:
        print(f"  Failed IDs: {', '.join(failures[:5])}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")

    return results, failures


def print_dataset_stats(df: pd.DataFrame):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 70)
    print("Boughter Dataset - Stage 1 Complete")
    print("=" * 70)

    print(f"\nTotal sequences across all subsets: {len(df)}")

    print("\nBreakdown by subset:")
    for subset in sorted(df["subset"].unique()):
        subset_df = df[df["subset"] == subset]
        print(f"  {subset:12s}: {len(subset_df):4d} sequences")

    print("\nFlag distribution:")
    for flag in sorted(df["num_flags"].unique()):
        count = len(df[df["num_flags"] == flag])
        pct = count / len(df) * 100
        print(f"  {flag} flags: {count:4d} ({pct:5.2f}%)")

    print("\nNovo flagging strategy results:")
    for category in ["specific", "mild", "non_specific"]:
        cat_df = df[df["flag_category"] == category]
        count = len(cat_df)
        pct = count / len(df) * 100
        included = len(cat_df[cat_df["include_in_training"]])
        print(
            f"  {category:15s}: {count:4d} ({pct:5.2f}%) - "
            f"{included} included in training"
        )

    training_df = df[df["include_in_training"]]
    print(f"\nTraining set size: {len(training_df)} sequences")
    print(f"Excluded (mild 1-3 flags): {len(df[~df['include_in_training']])} sequences")

    if len(training_df) > 0:
        label_dist = training_df["label"].value_counts()
        print("\nTraining set label balance:")
        for label in sorted(label_dist.index):
            count = label_dist[label]
            label_name = "Specific (0)" if label == 0 else "Non-specific (1)"
            pct = count / len(training_df) * 100
            print(f"  {label_name}: {count:4d} ({pct:5.2f}%)")


def main():
    """Main processing pipeline."""
    # Define dataset structure
    # Raw data is in boughter_raw/ (not committed to git)
    base_dir = Path("test_datasets/boughter_raw")

    subsets = {
        "flu": {
            "heavy_dna": base_dir / "flu_fastaH.txt",
            "light_dna": base_dir / "flu_fastaL.txt",
            "flags": base_dir / "flu_NumReact.txt",
            "flag_format": "numreact",
        },
        "hiv_nat": {
            "heavy_dna": base_dir / "nat_hiv_fastaH.txt",
            "light_dna": base_dir / "nat_hiv_fastaL.txt",
            "flags": base_dir / "nat_hiv_NumReact.txt",
            "flag_format": "numreact",
        },
        "hiv_cntrl": {
            "heavy_dna": base_dir / "nat_cntrl_fastaH.txt",
            "light_dna": base_dir / "nat_cntrl_fastaL.txt",
            "flags": base_dir / "nat_cntrl_NumReact.txt",
            "flag_format": "numreact",
        },
        "hiv_plos": {
            "heavy_dna": base_dir / "plos_hiv_fastaH.txt",
            "light_dna": base_dir / "plos_hiv_fastaL.txt",
            "flags": base_dir / "plos_hiv_YN.txt",
            "flag_format": "yn",
        },
        "gut_hiv": {
            "heavy_dna": base_dir / "gut_hiv_fastaH.txt",
            "light_dna": base_dir / "gut_hiv_fastaL.txt",
            "flags": base_dir / "gut_hiv_NumReact.txt",
            "flag_format": "numreact",
        },
        "mouse_iga": {
            "heavy_dna": base_dir / "mouse_fastaH.dat",
            "light_dna": base_dir / "mouse_fastaL.dat",
            "flags": base_dir / "mouse_YN.txt",
            "flag_format": "yn",
        },
    }

    # Process all subsets
    all_results = []
    all_failures = []

    for subset_name, paths in subsets.items():
        results, failures = process_subset(
            subset_name,
            paths["heavy_dna"],
            paths["light_dna"],
            paths["flags"],
            paths["flag_format"],
        )
        all_results.extend(results)
        all_failures.extend(failures)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print statistics
    print_dataset_stats(df)

    # Save output
    output_path = Path("test_datasets/boughter.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Output saved to: {output_path}")

    # Save failure log if any
    if all_failures:
        failure_log = Path("test_datasets/boughter_raw/translation_failures.log")
        failure_log.write_text("\n".join(all_failures))
        print(f"✓ Failure log saved to: {failure_log}")

    print("\n" + "=" * 70)
    print("Stage 1 Complete - Ready for Stage 2 (ANARCI annotation)")
    print("=" * 70)


if __name__ == "__main__":
    main()
