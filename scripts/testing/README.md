# Model Testing Scripts

Scripts for testing trained models against benchmark datasets.

**Purpose:** Validate model performance and reproduce Novo Nordisk results.

## Scripts

- `test_jain_novo_parity.py` - Test on Jain dataset (ELISA), verify Novo parity
- `test_harvey_psr_threshold.py` - Test on Harvey dataset (PSR) with calibrated threshold
- `demo_assay_specific_thresholds.py` - Demo script showing assay-specific threshold usage

## Usage

```bash
# Test on Jain (ELISA)
python scripts/testing/test_jain_novo_parity.py

# Test on Harvey (PSR) - use tmux wrapper for long-running job
./scripts/run_harvey_test_tmux.sh
```
