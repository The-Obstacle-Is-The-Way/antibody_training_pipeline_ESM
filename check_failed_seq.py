#!/usr/bin/env python3
"""Check why sequences are failing validation."""

from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path

# Read one of the failed sequences
fasta_path = Path("test_datasets/boughter_raw/nat_hiv_fastaH.txt")
sequences = list(SeqIO.parse(fasta_path, "fasta"))

# Check hiv_nat_000001 (which failed)
seq1 = sequences[0]
dna = str(seq1.seq).upper()

print(f"Sequence ID: {seq1.id}")
print(f"DNA length: {len(dna)}")
print(f"First 200 bp: {dna[:200]}")

# Find first ATG
atg_pos = dna.find("ATG")
print(f"\nFirst ATG at position: {atg_pos}")

if atg_pos >= 0:
    trimmed = dna[atg_pos:]
    protein = Seq(trimmed).translate(table=1, to_stop=False)
    protein_str = str(protein)

    print(f"\nProtein from ATG:")
    print(f"  Length: {len(protein_str)}")
    print(f"  First char: '{protein_str[0]}'")
    print(f"  First 60 aa: {protein_str[:60]}")
    print(f"  X count: {protein_str.count('X')}")
    print(f"  * count: {protein_str.count('*')}")

    # Count valid AA
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_count = sum(1 for aa in protein_str if aa in standard_aa)
    valid_ratio = valid_count / len(protein_str)
    print(f"  Valid AA ratio: {valid_ratio:.3f}")

    # Check why it failed validation
    print(f"\nValidation checks:")
    print(f"  Length OK (50-500)? {50 <= len(protein_str) <= 500}")
    print(f"  Starts with M? {protein_str[0] == 'M'}")
    print(f"  >80% valid AA? {valid_ratio >= 0.80}")

    # Show where the X's and *'s are
    print(f"\nLast 60 aa: {protein_str[-60:]}")
