from pathlib import Path

from Bio import SeqIO


def parse_fasta_dna(fasta_path: Path) -> list[str]:
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


def parse_numreact_flags(flag_path: Path) -> list[int]:
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


def parse_yn_flags(flag_path: Path) -> list[int]:
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
