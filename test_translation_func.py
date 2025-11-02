#!/usr/bin/env python3
"""Test the actual translation function on a failing sequence."""

import sys
sys.path.append('scripts')

from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
from typing import Optional

# Copy the functions from convert_boughter_to_csv.py
def find_first_atg(dna_seq: str) -> Optional[int]:
    """Find the first in-frame ATG (start codon) in DNA sequence."""
    dna_seq = dna_seq.upper()
    for i in range(0, min(300, len(dna_seq) - 2)):
        if dna_seq[i:i+3] == "ATG":
            return i
    return None


def translate_dna_to_protein(dna_seq: str) -> Optional[str]:
    """Translate DNA sequence to protein, trimming to first ATG start codon."""
    try:
        dna_seq = dna_seq.upper()
        atg_pos = find_first_atg(dna_seq)

        if atg_pos is not None:
            trimmed_dna = dna_seq[atg_pos:]
            protein = Seq(trimmed_dna).translate(table=1, to_stop=False)
            protein_str = str(protein)

            # Validate: should start with M (methionine) and have minimal X's
            if protein_str[0] == 'M' and protein_str.count('X') < len(protein_str) * 0.05:
                return protein_str

        # Fallback: if ATG trimming failed or produced bad result,
        # try raw translation (for sequences that don't need trimming)
        protein = Seq(dna_seq).translate(table=1, to_stop=False)
        return str(protein)

    except Exception as e:
        print(f"Translation failed: {e}")
        return None


def validate_translation(protein_seq: str) -> bool:
    """Validate that translation produced reasonable antibody sequence."""
    if not protein_seq:
        return False

    if len(protein_seq) < 50 or len(protein_seq) > 500:
        return False

    if protein_seq[0] != 'M':
        return False

    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_count = sum(1 for aa in protein_seq if aa in standard_aa)
    valid_ratio = valid_count / len(protein_seq)

    if valid_ratio < 0.80:
        return False

    return True


# Test on hiv_nat_000001
fasta_path = Path("test_datasets/boughter_raw/nat_hiv_fastaH.txt")
sequences = list(SeqIO.parse(fasta_path, "fasta"))
seq1 = sequences[0]
dna = str(seq1.seq)

print(f"Testing sequence: {seq1.id}")
print(f"DNA length: {len(dna)}\n")

# Run translation
protein = translate_dna_to_protein(dna)

print(f"Translation result:")
print(f"  Returned protein: {protein is not None}")
if protein:
    print(f"  Length: {len(protein)}")
    print(f"  First 60 aa: {protein[:60]}")
    print(f"  X count: {protein.count('X')}")
    print(f"  Starts with M: {protein[0] == 'M'}")

    # Run validation
    is_valid = validate_translation(protein)
    print(f"\nValidation result: {is_valid}")

    if not is_valid:
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        valid_count = sum(1 for aa in protein if aa in standard_aa)
        valid_ratio = valid_count / len(protein)
        print(f"  Length check (50-500): {50 <= len(protein) <= 500}")
        print(f"  Starts with M: {protein[0] == 'M'}")
        print(f"  Valid AA ratio: {valid_ratio:.3f} (need >=0.80)")
