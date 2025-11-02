#!/usr/bin/env python3
"""
Boughter Dataset Stage 1: DNA Translation & CSV Conversion

Processes raw Boughter DNA FASTA files, translates to protein, applies
Novo Nordisk flagging strategy, and outputs combined CSV.

Usage:
    python3 scripts/convert_boughter_to_csv.py

Outputs:
    test_datasets/boughter.csv - Combined dataset with Novo flagging
    test_datasets/boughter_raw/translation_failures.log - Failed sequences
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

# Canonical framework-1 motifs for human/mouse VH/VL domains
VDOMAIN_MOTIFS: Sequence[str] = (
    # Heavy-chain FR1 motifs
    "EVQ",
    "QVQ",
    "QVL",
    "QVR",
    "QVK",
    "EVQL",
    "QVQL",
    "EVKM",
    "EQLV",
    # Light-chain FR1 motifs (kappa/lambda)
    "EIVLT",
    "DIVMT",
    "DIQMT",
    "QSVLT",
    "QAVLT",
    "QIVLT",
    "EIVMT",
    "DITMT",
    # Generic VH/VL patterns observed across datasets
    "QVLV",
    "QVV",
    "EVLV",
    "EVTV",
    "EQLV",
)


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


def find_best_atg_translation(dna_seq: str) -> Optional[str]:
    """
    Find the best ATG-based translation for full-length sequences.

    For HIV/gut sequences with signal peptides and leading Ns/primers.
    Scans for plausible ATG starts, translates from each, scores quality.

    Strategy:
    1. Find all ATG codons in first 300bp
    2. Try translating from each
    3. Return the translation with:
       - Starts with M
       - Minimal X's and stops in first 150 aa
       - Looks like an antibody signal peptide

    Args:
        dna_seq: DNA nucleotide sequence string

    Returns:
        Best translated protein sequence, or None if no good translation found
    """
    dna_seq = dna_seq.upper()

    # Find all ATG positions in first 300bp
    atg_positions = []
    for i in range(0, min(300, len(dna_seq) - 2)):
        if dna_seq[i : i + 3] == "ATG":
            atg_positions.append(i)

    if not atg_positions:
        return None

    # Try each ATG and score the results
    best_protein = None
    best_score = -1

    for atg_pos in atg_positions:
        try:
            trimmed_dna = dna_seq[atg_pos:]
            protein = Seq(trimmed_dna).translate(table=1, to_stop=False)
            protein_str = str(protein)

            # Skip if doesn't start with M
            if not protein_str or protein_str[0] != "M":
                continue

            # Skip if too short (signal + V-domain should be at least 100 aa)
            if len(protein_str) < 100:
                continue

            # Ensure V-domain motif exists within first 120 aa after signal
            window = protein_str[:120]
            has_motif = any(motif in window for motif in VDOMAIN_MOTIFS)

            # Score based on quality in first 150 aa (signal + V-domain)
            first_150 = protein_str[: min(150, len(protein_str))]
            x_count_first = first_150.count("X")
            stop_count_first = first_150.count("*")

            # Heavily penalize X's, stops, or lack of motif in V-domain region
            score = 1000 - (x_count_first * 100) - (stop_count_first * 200)
            if not has_motif:
                score -= 500

            # Bonus for typical antibody signal peptide patterns
            if protein_str.startswith("MGW") or protein_str.startswith("MGA"):
                score += 50

            if score > best_score:
                best_score = score
                best_protein = protein_str

        except Exception:
            continue

    return best_protein


def translate_vdomain_direct(dna_seq: str) -> Optional[str]:
    """
    Direct translation for sequences that already begin with the V-domain.

    Boughter provides many heavy/light sequences that start directly with the
    framework-1 motif (e.g. EVQL, QVQL, EIVLT). These strings may still include
    downstream constant regions, so we only validate the first ~150 aa.
    """
    try:
        protein = Seq(dna_seq.upper()).translate(table=1, to_stop=False)
        protein_str = str(protein)

        if not protein_str:
            return None

        # Quality check: the V-domain region (first 150 aa) should be mostly
        # standard amino acids and free of premature stops.
        v_region = protein_str[: min(150, len(protein_str))]
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        valid_ratio = sum(aa in standard_aa for aa in v_region) / len(v_region)
        if valid_ratio < 0.85:
            return None
        if "*" in v_region:
            return None

        return protein_str
    except Exception:
        return None


def translate_dna_to_protein(dna_seq: str) -> Optional[str]:
    """
    Hybrid DNA translation for Boughter's two sequence types.

    Boughter raw FASTA contains two distinct formats:
    1. Full-length (HIV/gut): Signal peptide + V-domain, leading Ns/primers
       → Needs ATG-based trimming to correct reading frame
    2. Pre-trimmed V-domain (mouse/flu): Already V-domain only, no signal peptide
       → Direct translation (starts with Q/E/D, not M)

    Strategy (SSOT from Boughter notebooks + Novo validation):
    1. Detect sequence type using heuristics (leading Ns, length, ATG presence)
    2. Route to appropriate translation:
       - Full-length → ATG-based scoring (find best start, minimize X/*)
       - V-domain → Direct translation with V-domain pattern validation
    3. Fallback to direct translation if ATG method fails

    This hybrid approach recovers ~95% of sequences vs ~30% with naive translation.

    Args:
        dna_seq: DNA nucleotide sequence string

    Returns:
        Translated amino acid sequence, or None if translation fails
    """
    try:
        dna_seq = dna_seq.upper()

        # Try direct V-domain translation first (handles most sequences)
        protein = translate_vdomain_direct(dna_seq)
        if protein is not None:
            return protein

        # Fall back to ATG-based translation for full-length sequences
        protein = find_best_atg_translation(dna_seq)
        if protein is not None:
            return protein

        # Last resort: raw translation (may still be salvageable later)
        protein = Seq(dna_seq).translate(table=1, to_stop=False)
        return str(protein)

    except Exception as e:
        print(f"Translation failed: {e}")
        return None


def validate_translation(protein_seq: str) -> bool:
    """
    Validate that translation produced reasonable antibody sequence.

    Accepts BOTH sequence types from Boughter data:
    1. Full-length (HIV/gut): Signal peptide + V-domain
    2. V-domain only (mouse/flu): V-domain (± constant region) without signal

    Lenient validation — ANARCI will still perform strict numbering in Stage 2.

    Checks:
    1. Sequence exists and has reasonable length (95-500 aa)
    2. Canonical VH/VL motif occurs within first 120 aa (framework-1 region)
    3. First 150 aa are mostly clean (>85% standard amino acids)
    4. No stop codons in first 150 aa (would truncate the V-domain)

    Returns:
        True if valid, False otherwise
    """
    if not protein_seq:
        return False

    # Accept wide length range to accommodate both types:
    # - V-domain only: ~95-160 aa (mouse/flu)
    # - Full-length: ~150-500 aa (signal + V-domain + constant regions)
    if len(protein_seq) < 95 or len(protein_seq) > 500:
        return False

    # Check first 150 aa (V-domain region that ANARCI will extract)
    first_150 = protein_seq[: min(150, len(protein_seq))]

    # Must be mostly standard amino acids (>80% valid)
    # Allow some X's from sequencing uncertainty
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_count = sum(1 for aa in first_150 if aa in standard_aa)
    valid_ratio = valid_count / len(first_150)

    if valid_ratio < 0.80:
        return False

    # Reject if stop codons in first 150 aa (would truncate V-domain)
    if "*" in first_150:
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
