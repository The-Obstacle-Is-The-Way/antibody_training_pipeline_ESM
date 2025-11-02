#!/usr/bin/env python3
"""Test if ATG trimming would fix the translation problem."""

from Bio.Seq import Seq

# Test sequence from nat_hiv_fastaH.txt line 2
raw_dna = "NNNNNNNNNNNNNNTNNNNNNNTNCGATTTAGGTGACACTATAGAATAACATCCACTTTGCCTTTCTCTCCACAGGTGTCCACTCCCAGGTCCAACTGCACCTCGGTTCTATCGATTGAATTCCACCATGGGATGGTCATGTATCATCCTTTTTCTAGTAGCAACTGCAACCGGTGTACATTCCCAGGTGCAGCTGCAGGAGTCGGGCCCAAGACTGGTGAAGCCTTCGGAGACCTTGTCCCTCACCTGCACTGTGTCTGGTGGCTCCATCAGTCATTACTACTGGAGCTGGATCCGGCAGTCCCCAGGGAAGGGACTGGAGTGGATTGGATATCTCTATGATAGTGGGAGGGCCGGTTACAGCCCCTCCCTCAAGAGTCGAACCACCATATCGGCAGACACGTCAAACAACCAGTTGTCCCTGAAGGTGACCTCTGTGACCGCCGCAGACACGGCCGTCTATTACTGTGCGAGACATGAGGCCCCCCGGTACAGCTATGCCTTCCGCAGGTACTACCATTATGGTCTGGACGTCTGGGGCCAGGGGACCATGGTCACCGTCTCCTCA"

# Method 1: Current pipeline - translate raw sequence
current = Seq(raw_dna).translate(table=1, to_stop=False)
print("CURRENT METHOD (translate all):")
print(f"  First 60 aa: {str(current)[:60]}")
print(f"  Stop codons: {str(current).count('*')}")
print(f"  X's: {str(current).count('X')}")

# Method 2: Find first ATG and translate from there
atg_pos = raw_dna.find("ATG")
if atg_pos >= 0:
    from_atg = Seq(raw_dna[atg_pos:]).translate(table=1, to_stop=False)
    print(f"\nFIXED METHOD (from first ATG at position {atg_pos}):")
    print(f"  First 60 aa: {str(from_atg)[:60]}")
    print(f"  Stop codons: {str(from_atg).count('*')}")
    print(f"  X's: {str(from_atg).count('X')}")

    # Check if this would pass ANARCI
    print(f"\n  Would likely pass ANARCI? Signal peptide present: {'MGWSCIILFLVATATGVHS' in str(from_atg)}")
    print(f"  Variable domain start (QVQL): {'QVQL' in str(from_atg)}")
