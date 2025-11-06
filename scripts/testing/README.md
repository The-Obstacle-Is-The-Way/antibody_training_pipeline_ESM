# Testing Scripts

Educational demos and examples for using the model API.

**Purpose:** Show usage patterns for assay-specific thresholds.

## Scripts

- `demo_assay_specific_thresholds.py` - Tutorial demonstrating assay-specific threshold usage

## Usage

```bash
# Demo script showing ELISA vs PSR threshold usage
python scripts/testing/demo_assay_specific_thresholds.py
```

## Dataset-Specific Tests

Dataset-specific test scripts have been moved to `preprocessing/{dataset}/`:
- Jain parity test: `preprocessing/jain/test_novo_parity.py`
- Harvey PSR test: `preprocessing/harvey/test_psr_threshold.py`
