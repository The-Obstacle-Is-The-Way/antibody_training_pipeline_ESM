#!/usr/bin/env python3
"""Compare mouse vs HIV raw sequences to understand the difference."""

from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path

print("="*70)
print("MOUSE SEQUENCES (failing with new strict code)")
print("="*70)
mouse_h = list(SeqIO.parse("test_datasets/boughter_raw/mouse_fastaH.dat", "fasta"))
print(f"Total: {len(mouse_h)}")
print(f"\nFirst 3 sequences:")
for i in range(3):
    seq = mouse_h[i]
    dna = str(seq.seq)
    print(f"\n{i+1}. ID: {seq.id}")
    print(f"   Length: {len(dna)} bp")
    print(f"   First 150 bp: {dna[:150]}")
    print(f"   Has leading Ns: {dna[:20].count('N') > 5}")

    # Try translating
    try:
        protein = Seq(dna).translate(table=1, to_stop=False)
        print(f"   Naive translation first 40 aa: {str(protein)[:40]}")
        print(f"   Translation length: {len(protein)} aa")
    except:
        print(f"   Translation failed")

print("\n" + "="*70)
print("HIV SEQUENCES (passing with new code)")
print("="*70)
hiv_h = list(SeqIO.parse("test_datasets/boughter_raw/nat_hiv_fastaH.txt", "fasta"))
print(f"Total: {len(hiv_h)}")
print(f"\nFirst 3 sequences:")
for i in range(3):
    seq = hiv_h[i]
    dna = str(seq.seq)
    print(f"\n{i+1}. ID: {seq.id}")
    print(f"   Length: {len(dna)} bp")
    print(f"   First 150 bp: {dna[:150]}")
    print(f"   Has leading Ns: {dna[:20].count('N') > 5}")

    # Try translating
    try:
        protein = Seq(dna).translate(table=1, to_stop=False)
        print(f"   Naive translation first 40 aa: {str(protein)[:40]}")
        print(f"   Translation length: {len(protein)} aa")
    except:
        print(f"   Translation failed")
