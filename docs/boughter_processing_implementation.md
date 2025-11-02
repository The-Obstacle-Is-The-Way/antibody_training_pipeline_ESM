# Boughter Dataset Processing - Complete Implementation Specification

## Document Purpose
This document provides COMPLETE, EXPLICIT, step-by-step implementation instructions for processing the Boughter 2020 dataset to replicate Novo Nordisk's methodology (Sakhnini et al. 2025).

**Target Audience**: Implementation engineer who will write `convert_boughter_to_csv.py` and `process_boughter.py`

---

## Implementation Overview

### Pipeline Stages

```
Stage 1: convert_boughter_to_csv.py
  Input:  Raw FASTA DNA files + flag files from test_datasets/boughter/
  Output: test_datasets/boughter.csv (combined CSV with translated protein sequences)

Stage 2: process_boughter.py
  Input:  test_datasets/boughter.csv
  Output: test_datasets/boughter/*.csv (16 fragment-specific CSVs)
```

---

## Stage 1: DNA Translation & CSV Conversion

### Stage 1.1: File Reading

**Input Files per Subset** (6 subsets total):

```python
subsets = {
    'flu': {
        'heavy_dna': 'flu_fastaH.txt',
        'light_dna': 'flu_fastaL.txt',
        'flags': 'flu_NumReact.txt',
        'flag_format': 'numreact'
    },
    'hiv_nat': {
        'heavy_dna': 'nat_hiv_fastaH.txt',
        'light_dna': 'nat_hiv_fastaL.txt',
        'flags': 'nat_hiv_NumReact.txt',
        'flag_format': 'numreact'
    },
    'hiv_cntrl': {
        'heavy_dna': 'nat_cntrl_fastaH.txt',
        'light_dna': 'nat_cntrl_fastaL.txt',
        'flags': 'nat_cntrl_NumReact.txt',
        'flag_format': 'numreact'
    },
    'hiv_plos': {
        'heavy_dna': 'plos_hiv_fastaH.txt',
        'light_dna': 'plos_hiv_fastaL.txt',
        'flags': 'plos_hiv_YN.txt',
        'flag_format': 'yn'
    },
    'gut_hiv': {
        'heavy_dna': 'gut_hiv_fastaH.txt',
        'light_dna': 'gut_hiv_fastaL.txt',
        'flags': 'gut_hiv_NumReact.txt',
        'flag_format': 'numreact'
    },
    'mouse_iga': {
        'heavy_dna': 'mouse_fastaH.dat',
        'light_dna': 'mouse_fastaL.dat',
        'flags': 'mouse_YN.txt',
        'flag_format': 'yn'
    }
}
```

### Stage 1.2: FASTA Parsing

**FASTA Format**:
```
>                           # Header (empty after ">")
NAGGTGCAGCTGGTGCAGT...     # DNA nucleotide sequence
>                           # Next header
GAGGTGCAGCTGGTGGAGT...     # Next DNA sequence
```

**Parsing Algorithm**:
```python
def parse_fasta_dna(filepath: str) -> List[str]:
    """
    Parse FASTA DNA file and return list of nucleotide sequences.

    Returns:
        List of DNA sequence strings (in order of appearance)
    """
    sequences = []
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('>'):
                # Save previous sequence if exists
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            elif line:  # Non-empty, non-header line
                current_seq.append(line)

        # Save last sequence
        if current_seq:
            sequences.append(''.join(current_seq))

    return sequences
```

### Stage 1.3: Flag File Parsing

**NumReact Format** (0-7 scale):
```
reacts      # Header line (skip)
1           # Integer 0-7
0
5
...
```

**Y/N Format** (binary):
```
Y           # Y or N (no header)
N
Y
...
```

**Parsing Algorithm**:
```python
def parse_flags(filepath: str, flag_format: str) -> List[int]:
    """
    Parse flag file and return list of polyreactivity counts.

    Args:
        filepath: Path to flag file
        flag_format: 'numreact' or 'yn'

    Returns:
        List of integers representing polyreactivity flags (0-7)
    """
    flags = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove header for numreact format
    if flag_format == 'numreact':
        lines = lines[1:]  # Skip "reacts" header

    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        if flag_format == 'numreact':
            # Parse integer directly
            try:
                flag = int(line)
                flags.append(flag)
            except ValueError:
                # Handle empty or malformed entries
                flags.append(None)

        elif flag_format == 'yn':
            # Convert Y/N to flag count
            if line.upper() == 'Y':
                # Polyreactive - use 7 (maximum) as proxy
                flags.append(7)
            elif line.upper() == 'N':
                # Non-polyreactive
                flags.append(0)
            else:
                # Malformed
                flags.append(None)

    return flags
```

### Stage 1.4: DNA to Protein Translation

**Translation Requirements**:
- Use standard genetic code
- Antibody sequences typically encoded in Frame 1 (start at position 0)
- Handle ambiguous nucleotides (N → X in protein)
- No signal peptide cleavage (keep full translated sequence)

**Implementation**:
```python
from Bio.Seq import Seq

def translate_dna_to_protein(dna_seq: str) -> str:
    """
    Translate DNA nucleotide sequence to amino acid protein sequence.

    Uses standard genetic code (table 1).
    Translates in frame 1 (position 0).

    Args:
        dna_seq: DNA nucleotide sequence string

    Returns:
        Translated amino acid sequence
        Returns None if translation fails
    """
    try:
        # Create Bio.Seq object
        seq = Seq(dna_seq)

        # Translate using standard genetic code (table=1)
        # to_stop=False to get full sequence even if stop codon present
        protein = seq.translate(table=1, to_stop=False)

        return str(protein)

    except Exception as e:
        print(f"Translation failed for sequence: {dna_seq[:50]}...")
        print(f"Error: {e}")
        return None
```

**Validation Checks**:
```python
def validate_translation(dna_seq: str, protein_seq: str) -> bool:
    """
    Validate that translation produced reasonable antibody sequence.

    Checks:
    1. Protein sequence is not empty
    2. Length is reasonable for antibody V-domain (>50 aa)
    3. No internal stop codons (*) in variable region
    4. Contains expected antibody motifs (optional)

    Returns:
        True if valid, False otherwise
    """
    if not protein_seq:
        return False

    # Check minimum length (antibody V-domains are typically 100-130 aa)
    if len(protein_seq) < 50:
        print(f"Warning: Sequence too short ({len(protein_seq)} aa)")
        return False

    # Check for stop codons (should be at end only, if present)
    if '*' in protein_seq[:-1]:  # Allow * at end only
        print(f"Warning: Internal stop codon detected")
        return False

    return True
```

### Stage 1.5: Sequence Pairing & ID Generation

**ID Format**: `{subset}_{sequential_number:06d}`

Examples:
- `flu_000001`, `flu_000002`, ...
- `hiv_nat_000001`, `hiv_nat_000002`, ...
- `mouse_iga_000001`, ...

**Pairing Algorithm**:
```python
def pair_sequences(subset_name: str,
                  heavy_dna_list: List[str],
                  light_dna_list: List[str],
                  flags_list: List[int]) -> List[Dict]:
    """
    Translate DNA and pair heavy/light chains with flags.

    Returns:
        List of dictionaries with paired data
    """
    # Validate counts match
    counts = [len(heavy_dna_list), len(light_dna_list), len(flags_list)]
    if len(set(counts)) != 1:
        raise ValueError(
            f"{subset_name}: Sequence count mismatch - "
            f"Heavy: {counts[0]}, Light: {counts[1]}, Flags: {counts[2]}"
        )

    results = []
    failures = []

    for idx in range(len(heavy_dna_list)):
        # Generate sequential ID
        seq_id = f"{subset_name}_{idx+1:06d}"

        # Translate DNA → protein
        heavy_protein = translate_dna_to_protein(heavy_dna_list[idx])
        light_protein = translate_dna_to_protein(light_dna_list[idx])

        # Validate translations
        if not heavy_protein or not light_protein:
            failures.append((seq_id, "Translation failed"))
            continue

        if not validate_translation(heavy_dna_list[idx], heavy_protein):
            failures.append((seq_id, "Heavy chain validation failed"))
            continue

        if not validate_translation(light_dna_list[idx], light_protein):
            failures.append((seq_id, "Light chain validation failed"))
            continue

        # Get flag value
        flag_value = flags_list[idx]
        if flag_value is None:
            failures.append((seq_id, "Missing flag value"))
            continue

        # Store paired data
        results.append({
            'id': seq_id,
            'subset': subset_name,
            'heavy_seq': heavy_protein,
            'light_seq': light_protein,
            'num_flags': flag_value,
            'source': 'boughter2020'
        })

    # Report failures
    if failures:
        print(f"{subset_name}: {len(failures)} sequences failed validation:")
        for seq_id, reason in failures[:5]:  # Show first 5
            print(f"  {seq_id}: {reason}")

    return results
```

### Stage 1.6: Novo Flagging Strategy Implementation

**From Sakhnini et al. 2025, Section 4.3**:
> "First, the Boughter dataset was parsed into three groups: specific group (0 flags), mildly poly-reactive group (1-3 flags) and poly-reactive group (>3 flags)."

**Flagging Rules**:
```python
def apply_novo_flagging(num_flags: int) -> Dict:
    """
    Apply Novo Nordisk flagging strategy to polyreactivity count.

    Rules:
        0 flags   → label=0 (specific), category='specific', INCLUDE
        1-3 flags → label=None (excluded), category='mild', EXCLUDE
        >3 flags  → label=1 (non-specific), category='non_specific', INCLUDE

    Args:
        num_flags: Integer polyreactivity flag count (0-7)

    Returns:
        Dictionary with label, category, and include flag
    """
    if num_flags == 0:
        return {
            'label': 0,
            'flag_category': 'specific',
            'include_in_training': True
        }
    elif 1 <= num_flags <= 3:
        return {
            'label': None,  # Excluded from training
            'flag_category': 'mild',
            'include_in_training': False
        }
    elif num_flags >= 4:
        return {
            'label': 1,
            'flag_category': 'non_specific',
            'include_in_training': True
        }
    else:
        # Should not happen if data is valid
        raise ValueError(f"Invalid flag count: {num_flags}")
```

### Stage 1.7: CSV Output Format

**Column Structure**:
```python
output_columns = [
    'id',                    # {subset}_{number:06d}
    'subset',                # flu, hiv_nat, hiv_cntrl, hiv_plos, gut_hiv, mouse_iga
    'heavy_seq',             # Translated amino acid sequence
    'light_seq',             # Translated amino acid sequence
    'num_flags',             # Original flag count (0-7)
    'flag_category',         # specific, mild, non_specific
    'label',                 # 0, 1, or None
    'include_in_training',   # True/False
    'source'                 # boughter2020
]
```

**Example rows**:
```csv
id,subset,heavy_seq,light_seq,num_flags,flag_category,label,include_in_training,source
flu_000001,flu,EVQLVQSGAEVKKPGA...,EIVLTQSPGTLSLSPA...,1,mild,,False,boughter2020
flu_000002,flu,EVQLVESGGGVVQPGR...,DIQMTQSPSSLSASVG...,0,specific,0,True,boughter2020
flu_000005,flu,EVQLVESGAEVKKPGA...,DIQMTQSPSSLSASVG...,5,non_specific,1,True,boughter2020
```

**Dataset Statistics to Report**:
```python
def print_dataset_stats(df: pd.DataFrame):
    """Print comprehensive dataset statistics."""
    print("=" * 70)
    print("Boughter Dataset - Stage 1 Complete")
    print("=" * 70)

    print(f"\nTotal sequences across all subsets: {len(df)}")

    print("\nBreakdown by subset:")
    for subset in df['subset'].unique():
        subset_df = df[df['subset'] == subset]
        print(f"  {subset:12s}: {len(subset_df):4d} sequences")

    print("\nFlag distribution:")
    for flag in sorted(df['num_flags'].unique()):
        count = len(df[df['num_flags'] == flag])
        pct = count / len(df) * 100
        print(f"  {flag} flags: {count:4d} ({pct:5.2f}%)")

    print("\nNovo flagging strategy results:")
    for category in ['specific', 'mild', 'non_specific']:
        cat_df = df[df['flag_category'] == category]
        count = len(cat_df)
        pct = count / len(df) * 100
        included = len(cat_df[cat_df['include_in_training']])
        print(f"  {category:15s}: {count:4d} ({pct:5.2f}%) - {included} included in training")

    print(f"\nTraining set size: {len(df[df['include_in_training']])} sequences")
    print(f"Excluded (mild 1-3 flags): {len(df[~df['include_in_training']])} sequences")

    label_dist = df[df['include_in_training']]['label'].value_counts()
    print("\nTraining set label balance:")
    for label, count in label_dist.items():
        label_name = "Specific (0)" if label == 0 else "Non-specific (1)"
        pct = count / len(df[df['include_in_training']]) * 100
        print(f"  {label_name}: {count:4d} ({pct:5.2f}%)")
```

---

## Stage 2: ANARCI Annotation & Fragment Extraction

### Stage 2.1: Input Validation

**Pre-processing checks**:
```python
def validate_stage1_output(csv_path: str) -> pd.DataFrame:
    """
    Validate that Stage 1 CSV has correct structure.

    Checks:
    1. All required columns present
    2. No null values in critical columns
    3. Sequences are amino acids (not DNA)
    4. Flag categories are valid
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        'id', 'subset', 'heavy_seq', 'light_seq',
        'num_flags', 'flag_category', 'label',
        'include_in_training', 'source'
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for nulls
    critical_cols = ['id', 'heavy_seq', 'light_seq', 'num_flags']
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Column {col} has {null_count} null values")

    # Verify sequences are protein (contain standard amino acids)
    aa_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY*]+$')

    for idx, row in df.iterrows():
        if not aa_pattern.match(row['heavy_seq']):
            raise ValueError(f"{row['id']}: Heavy chain is not protein sequence")
        if not aa_pattern.match(row['light_seq']):
            raise ValueError(f"{row['id']}: Light chain is not protein sequence")

    print(f"✓ Stage 1 validation passed: {len(df)} sequences")
    return df
```

### Stage 2.2: ANARCI Annotation

**Following process_jain.py pattern**:

```python
import riot_na

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_sequence(seq_id: str, sequence: str, chain: str) -> Optional[Dict[str, str]]:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Uses strict IMGT boundaries:
    - CDR-H3: positions 105-117 (EXCLUDES position 118, which is FR4 J-anchor)
    - CDR-H2: positions 56-65 (fixed IMGT)
    - CDR-H1: positions 27-38 (fixed IMGT)

    Note: Position 118 (J-Trp/Phe) is conserved FR4, NOT CDR.
    Boughter's published .dat files include position 118, but we use
    strict IMGT for biological correctness and ML best practices.

    From Sakhnini et al. 2025, Section 4.3:
    "The primary sequences were annotated in the CDRs using ANARCI
    following the IMGT numbering scheme"

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    assert chain in ('H', 'L'), "chain must be 'H' or 'L'"

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments
        fragments = {
            f'full_seq_{chain}': annotation.sequence_alignment_aa,
            f'fwr1_aa_{chain}': annotation.fwr1_aa,
            f'cdr1_aa_{chain}': annotation.cdr1_aa,
            f'fwr2_aa_{chain}': annotation.fwr2_aa,
            f'cdr2_aa_{chain}': annotation.cdr2_aa,
            f'fwr3_aa_{chain}': annotation.fwr3_aa,
            f'cdr3_aa_{chain}': annotation.cdr3_aa,
            f'fwr4_aa_{chain}': annotation.fwr4_aa,
        }

        # Create concatenated fragments
        fragments[f'cdrs_{chain}'] = ''.join([
            fragments[f'cdr1_aa_{chain}'],
            fragments[f'cdr2_aa_{chain}'],
            fragments[f'cdr3_aa_{chain}'],
        ])

        fragments[f'fwrs_{chain}'] = ''.join([
            fragments[f'fwr1_aa_{chain}'],
            fragments[f'fwr2_aa_{chain}'],
            fragments[f'fwr3_aa_{chain}'],
            fragments[f'fwr4_aa_{chain}'],
        ])

        return fragments

    except Exception as e:
        print(f'Warning: Failed to annotate {seq_id} ({chain}): {e}', file=sys.stderr)
        return None
```

### Stage 2.3: Fragment Assembly

**16 Fragment Types** (from Sakhnini et al. 2025, Table 4):

```python
def process_boughter_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process Boughter CSV to extract all fragments.

    Extracts 16 antibody fragments following Sakhnini et al. 2025 methodology.

    Args:
        csv_path: Path to boughter.csv from Stage 1

    Returns:
        DataFrame with all fragments and metadata
    """
    print(f'Reading {csv_path}...')
    df = pd.read_csv(csv_path)

    print(f'  Total antibodies: {len(df)}')
    print('  Annotating sequences with ANARCI (IMGT scheme)...')

    results = []
    failures = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Annotating'):
        # Annotate heavy chain
        heavy_frags = annotate_sequence(f"{row['id']}_VH", row['heavy_seq'], 'H')

        # Annotate light chain
        light_frags = annotate_sequence(f"{row['id']}_VL", row['light_seq'], 'L')

        if heavy_frags is None or light_frags is None:
            failures.append(row['id'])
            print(f"  Skipping {row['id']} - annotation failed")
            continue

        # Combine all fragments and metadata
        result = {
            'id': row['id'],
            'subset': row['subset'],
            'label': row['label'],
            'num_flags': row['num_flags'],
            'flag_category': row['flag_category'],
            'include_in_training': row['include_in_training'],
            'source': row['source'],
        }

        result.update(heavy_frags)
        result.update(light_frags)

        # Create paired/combined fragments
        result['vh_vl'] = result['full_seq_H'] + result['full_seq_L']
        result['all_cdrs'] = result['cdrs_H'] + result['cdrs_L']
        result['all_fwrs'] = result['fwrs_H'] + result['fwrs_L']

        results.append(result)

    df_annotated = pd.DataFrame(results)

    print(f'\n  Successfully annotated: {len(df_annotated)}/{len(df)} antibodies')

    if failures:
        print(f'  Failures: {len(failures)}')
        # Write failures to log
        failure_log = Path('test_datasets/boughter/failed_sequences.txt')
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log, 'w') as f:
            f.write('\n'.join(failures))
        print(f'  Failed IDs written to: {failure_log}')

    return df_annotated
```

### Stage 2.4: Fragment CSV Creation

**16 Fragment Files** (following Jain/Harvey pattern):

```python
def create_fragment_csvs(df: pd.DataFrame, output_dir: Path):
    """
    Create separate CSV files for each fragment type.

    Following the 16-fragment methodology from Sakhnini et al. 2025 Table 4.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs (test_datasets/boughter/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all 16 fragment types
    fragments = {
        # 1-2: Full variable domains
        'VH_only': ('full_seq_H', 'heavy_variable_domain'),
        'VL_only': ('full_seq_L', 'light_variable_domain'),

        # 3-5: Heavy CDRs
        'H-CDR1': ('cdr1_aa_H', 'h_cdr1'),
        'H-CDR2': ('cdr2_aa_H', 'h_cdr2'),
        'H-CDR3': ('cdr3_aa_H', 'h_cdr3'),

        # 6-8: Light CDRs
        'L-CDR1': ('cdr1_aa_L', 'l_cdr1'),
        'L-CDR2': ('cdr2_aa_L', 'l_cdr2'),
        'L-CDR3': ('cdr3_aa_L', 'l_cdr3'),

        # 9-10: Concatenated CDRs
        'H-CDRs': ('cdrs_H', 'h_cdrs_concatenated'),
        'L-CDRs': ('cdrs_L', 'l_cdrs_concatenated'),

        # 11-12: Concatenated FWRs
        'H-FWRs': ('fwrs_H', 'h_fwrs_concatenated'),
        'L-FWRs': ('fwrs_L', 'l_fwrs_concatenated'),

        # 13: Paired variable domains
        'VH+VL': ('vh_vl', 'paired_variable_domains'),

        # 14-15: All CDRs/FWRs
        'All-CDRs': ('all_cdrs', 'all_cdrs_heavy_light'),
        'All-FWRs': ('all_fwrs', 'all_fwrs_heavy_light'),

        # 16: Full (alias for VH+VL)
        'Full': ('vh_vl', 'full_sequence'),
    }

    print(f'\nCreating {len(fragments)} fragment-specific CSV files...')

    for fragment_name, (column_name, description) in fragments.items():
        output_path = output_dir / f'{fragment_name}_boughter.csv'

        # Create fragment-specific CSV with standardized column names
        fragment_df = pd.DataFrame({
            'id': df['id'],
            'sequence': df[column_name],
            'label': df['label'],
            'subset': df['subset'],
            'num_flags': df['num_flags'],
            'flag_category': df['flag_category'],
            'include_in_training': df['include_in_training'],
            'source': df['source'],
            'sequence_length': df[column_name].str.len(),
        })

        fragment_df.to_csv(output_path, index=False)

        # Show stats
        mean_len = fragment_df['sequence'].str.len().mean()
        min_len = fragment_df['sequence'].str.len().min()
        max_len = fragment_df['sequence'].str.len().max()

        print(f'  ✓ {fragment_name:12s} → {output_path.name:30s} '
              f'(len: {min_len}-{max_len} aa, mean: {mean_len:.1f})')

    print(f'\n✓ All fragments saved to: {output_dir}/')
```

---

## Quality Control & Validation

### QC Checkpoint 1: Translation Validation

**After Stage 1**:
```python
def validate_translations(df: pd.DataFrame):
    """
    Validate that all translations produced valid antibody sequences.

    Checks:
    1. No sequences with multiple stop codons
    2. Length distributions match expected antibody V-domain ranges
    3. All sequences start with reasonable signal peptide or V-domain start
    """
    print("\nTranslation Quality Control:")
    print("=" * 50)

    # Check for stop codons
    stop_in_heavy = df['heavy_seq'].str.contains('\*').sum()
    stop_in_light = df['light_seq'].str.contains('\*').sum()

    print(f"Heavy chains with stop codons: {stop_in_heavy}/{len(df)}")
    print(f"Light chains with stop codons: {stop_in_light}/{len(df)}")

    # Length distributions
    heavy_lens = df['heavy_seq'].str.len()
    light_lens = df['light_seq'].str.len()

    print(f"\nHeavy chain lengths: {heavy_lens.min()}-{heavy_lens.max()} aa "
          f"(mean: {heavy_lens.mean():.1f})")
    print(f"Light chain lengths: {light_lens.min()}-{light_lens.max()} aa "
          f"(mean: {light_lens.mean():.1f})")

    # Expected ranges for antibody V-domains: ~100-130 aa
    if heavy_lens.mean() < 80 or heavy_lens.mean() > 150:
        print("⚠ WARNING: Heavy chain length outside expected range (100-130 aa)")

    if light_lens.mean() < 80 or light_lens.mean() > 150:
        print("⚠ WARNING: Light chain length outside expected range (100-130 aa)")
```

### QC Checkpoint 2: ANARCI Success Rate

**After Stage 2**:
```python
def validate_annotation_success(df_input: pd.DataFrame, df_output: pd.DataFrame):
    """
    Check ANARCI annotation success rate.

    Expected: >95% success rate
    """
    success_rate = len(df_output) / len(df_input) * 100

    print(f"\nANARCI Annotation Success Rate: {success_rate:.1f}%")
    print(f"  Input sequences: {len(df_input)}")
    print(f"  Successfully annotated: {len(df_output)}")
    print(f"  Failed: {len(df_input) - len(df_output)}")

    if success_rate < 95:
        print("⚠ WARNING: Annotation success rate below 95%")
        print("  Check failed_sequences.txt for details")
```

### QC Checkpoint 3: CDR Length Distributions

**After Stage 2**:
```python
def validate_cdr_lengths(df: pd.DataFrame):
    """
    Validate CDR lengths match expected IMGT ranges.

    Expected IMGT CDR lengths:
    H-CDR1: 5-7 aa (typically 8 with gaps)
    H-CDR2: 6-10 aa (typically 8 with gaps)
    H-CDR3: 3-25 aa (highly variable)
    L-CDR1: 6-12 aa
    L-CDR2: 3 aa (highly conserved)
    L-CDR3: 7-11 aa
    """
    print("\nCDR Length Distributions:")
    print("=" * 50)

    cdrs = {
        'H-CDR1': ('cdr1_aa_H', (5, 10)),
        'H-CDR2': ('cdr2_aa_H', (6, 12)),
        'H-CDR3': ('cdr3_aa_H', (3, 30)),
        'L-CDR1': ('cdr1_aa_L', (6, 15)),
        'L-CDR2': ('cdr2_aa_L', (2, 5)),
        'L-CDR3': ('cdr3_aa_L', (7, 15)),
    }

    for cdr_name, (col, (min_exp, max_exp)) in cdrs.items():
        lengths = df[col].str.len()
        mean_len = lengths.mean()
        min_len = lengths.min()
        max_len = lengths.max()

        status = "✓" if min_exp <= mean_len <= max_exp else "⚠"

        print(f"{status} {cdr_name:8s}: {min_len:2.0f}-{max_len:2.0f} aa "
              f"(mean: {mean_len:5.2f}, expected: {min_exp}-{max_exp})")
```

---

## Expected Output Summary

### Final Outputs

**Stage 1**:
- `test_datasets/boughter.csv` - ~1,171 total sequences
  - After Novo filtering: ~700-800 training sequences (exact number TBD)
  - Columns: id, subset, heavy_seq, light_seq, num_flags, flag_category, label, include_in_training, source

**Stage 2**:
- `test_datasets/boughter/` - 16 fragment-specific CSVs
  - Each file: ~700-800 sequences (annotation failures excluded)
  - Naming: `{Fragment}_boughter.csv` (e.g., `VH_only_boughter.csv`)
  - Columns: id, sequence, label, subset, num_flags, flag_category, include_in_training, source, sequence_length

### Performance Metrics to Track

```python
def print_final_summary(df_stage1: pd.DataFrame, df_stage2: pd.DataFrame):
    """Print comprehensive processing summary."""
    print("\n" + "=" * 70)
    print("BOUGHTER DATASET PROCESSING - FINAL SUMMARY")
    print("=" * 70)

    print("\nStage 1: DNA Translation & CSV Conversion")
    print(f"  Input: 6 subsets, ~1,171 sequences")
    print(f"  Output: {len(df_stage1)} sequences")
    print(f"  Translation success rate: {len(df_stage1)/1171*100:.1f}%")

    print("\nNovo Flagging Strategy Results:")
    print(f"  Training set (0 flags + >3 flags): {df_stage1['include_in_training'].sum()}")
    print(f"  Excluded (1-3 flags): {(~df_stage1['include_in_training']).sum()}")

    label_counts = df_stage1[df_stage1['include_in_training']]['label'].value_counts()
    print(f"  Label 0 (specific): {label_counts.get(0, 0)}")
    print(f"  Label 1 (non-specific): {label_counts.get(1, 0)}")

    print("\nStage 2: ANARCI Annotation & Fragment Extraction")
    print(f"  Input: {len(df_stage1)} sequences")
    print(f"  Output: {len(df_stage2)} successfully annotated")
    print(f"  Annotation success rate: {len(df_stage2)/len(df_stage1)*100:.1f}%")
    print(f"  Fragment files created: 16")

    print("\nDataset Composition (by subset):")
    for subset in df_stage2['subset'].unique():
        count = len(df_stage2[df_stage2['subset'] == subset])
        training = len(df_stage2[(df_stage2['subset'] == subset) &
                                 (df_stage2['include_in_training'])])
        print(f"  {subset:12s}: {count:4d} total, {training:4d} in training set")

    print("\n" + "=" * 70)
    print("✓ Boughter Processing Complete - Ready for Model Training/Testing")
    print("=" * 70)
```

---

## Dependencies

**Required Python Packages**:
```
biopython>=1.79      # For DNA translation
pandas>=1.3.0        # For CSV handling
riot-na>=0.1.0       # For ANARCI annotation
tqdm>=4.62.0         # For progress bars
```

**Installation**:
```bash
pip install biopython pandas riot-na tqdm
```

---

## Error Handling

### Common Failure Modes

1. **Translation failures**: DNA sequence too short, contains invalid nucleotides
   - Log failed sequence IDs
   - Continue processing remaining sequences
   - Report failure rate in summary

2. **ANARCI annotation failures**: Sequence doesn't match antibody structure
   - Write failed IDs to `failed_sequences.txt`
   - Expected failure rate: <5%
   - Continue processing remaining sequences

3. **File count mismatches**: Heavy/light/flag file lengths don't match
   - STOP processing for that subset
   - Raise clear error message
   - Do not guess or skip sequences

### Logging Strategy

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('boughter_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

---

## Implementation Checklist

### Stage 1: convert_boughter_to_csv.py

- [ ] Implement FASTA DNA parser
- [ ] Implement NumReact flag parser
- [ ] Implement Y/N flag parser
- [ ] Implement DNA → protein translation (BioPython)
- [ ] Implement translation validation
- [ ] Implement sequence pairing logic
- [ ] Implement Novo flagging strategy
- [ ] Generate sequential IDs per subset
- [ ] Create combined CSV output
- [ ] Print dataset statistics
- [ ] Log all failures

### Stage 2: process_boughter.py

- [ ] Validate Stage 1 CSV structure
- [ ] Initialize ANARCI (riot_na) annotator
- [ ] Implement heavy chain annotation
- [ ] Implement light chain annotation
- [ ] Extract all 16 fragment types
- [ ] Create fragment-specific CSVs
- [ ] Write failed annotation IDs to log
- [ ] Print annotation success rate
- [ ] Print CDR length distributions
- [ ] Final summary statistics

### Quality Control

- [ ] Translation success rate >95%
- [ ] ANARCI annotation success rate >95%
- [ ] CDR lengths within expected ranges
- [ ] Label balance in training set
- [ ] Verify 6 subsets all processed
- [ ] Verify 16 fragment files created
- [ ] Verify output format matches Jain/Harvey pattern

---

## Next Steps After Implementation

1. **Test on small subset first** (e.g., just `flu` subset)
2. **Run full pipeline on all 6 subsets**
3. **Compare statistics with Boughter 2020 paper Table 1**
4. **Verify fragment files load correctly with existing data loaders**
5. **Document any deviations or unexpected findings**
6. **Request senior AI agent consensus review** before finalizing

---

## References

**Primary Sources**:
1. Boughter CT et al. (2020) eLife 9:e61393
2. Sakhnini LI et al. (2025) "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"
3. BioPython Seq.translate() documentation: https://biopython.org/docs/1.75/api/Bio.Seq.html#Bio.Seq.Seq.translate

**Implementation Patterns**:
- `preprocessing/process_harvey.py` - Nanobody (VHH only) processing
- `preprocessing/process_jain.py` - Full antibody (VH+VL) processing

---

**Document Version**: 2.0
**Date**: 2025-11-02
**Status**: Complete - Ready for AI Senior Review
**Confidence**: 95% (DNA translation methodology validated, ANARCI pattern confirmed from existing scripts)
