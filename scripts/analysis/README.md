# Analysis Scripts

Scripts for analyzing model behavior and optimizing parameters.

**Purpose:** Investigate model predictions and threshold calibration.

## Scripts

- `analyze_threshold_optimization.py` - Finds optimal decision thresholds for ELISA vs PSR datasets

## Usage

```bash
python scripts/analysis/analyze_threshold_optimization.py
```

This script:
1. Loads trained model
2. Extracts prediction probabilities for Jain and Shehata
3. Finds optimal thresholds that match Novo's confusion matrices
4. Shows why single threshold can't work for both ELISA and PSR
