# Validation Scripts

Generic quality control scripts for cross-dataset validation.

**Purpose:** Ensure data integrity for any dataset.

## Scripts

- `validate_fragments.py` - Generic fragment extraction validation (works for any dataset)

## Usage

```bash
# Validate fragment extraction for any dataset
python scripts/validation/validate_fragments.py --dataset data/test/jain/
```

## Dataset-Specific Validation

Dataset-specific validation scripts have been moved to `preprocessing/{dataset}/`:
- Boughter QC audit: `preprocessing/boughter/audit_training_qc.py`
- Jain conversion validation: `preprocessing/jain/validate_conversion.py`
- Shehata conversion validation: `preprocessing/shehata/validate_conversion.py`
