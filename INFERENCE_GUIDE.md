# Antibody Non-Specificity Prediction: Inference Guide

## Overview
This guide details how to use the `antibody-predict` CLI to screen antibody sequences for non-specificity (polyreactivity). It explains input requirements, command usage, and how to interpret the results.

## 1. Input Specification

The tool requires a **CSV file** as input.

### File Requirements
*   **Format:** Standard Comma-Separated Values (`.csv`).
*   **Location:** The file can exist anywhere on your filesystem (relative or absolute paths are accepted).

### Column Requirements
*   **`sequence` (Default):** By default, the tool looks for a column named `sequence`.
    *   **Customizable:** If your column is named differently (e.g., `vh_sequence`, `heavy_chain`), you can specify this using the `sequence_column` argument (see below).
    *   **Content:** The amino acid sequence of the antibody (Variable Heavy domain / VH).
    *   **Case:** Case-insensitive (sequences are automatically normalized to uppercase).
    *   **Cleaning:** Whitespace and standard gaps are handled, but pure amino acid sequences are preferred.
*   **Other Columns:** Any other columns (e.g., `id`, `name`, `notes`) will be **preserved** in the output file.

### Example Input File (`my_candidates.csv`)
```csv
id,sequence,description
mAb-001,EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR,Primary candidate
mAb-002,QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR,Backup sequence
```

---

## 2. Running Inference

Inference is run via the command line using `uv`.

### Basic Command
```bash
uv run antibody-predict \
    input_file="path/to/your/input.csv" \
    output_file="path/to/save/results.csv" \
    classifier.path="path/to/trained_model.pkl"
```

### Custom Sequence Column
If your CSV uses a different name for the sequence column (e.g., `vh_sequence`):
```bash
uv run antibody-predict \
    input_file="data/my_data.csv" \
    sequence_column="vh_sequence" \
    classifier.path="..."
```

### Assay-Specific Thresholds
You can optimize the decision threshold for specific assays (improving accuracy):
```bash
# For PSR (Polyspecificity Reagent) assays (Threshold = 0.5495)
uv run antibody-predict ... assay_type="PSR"

# For ELISA assays (Threshold = 0.5)
uv run antibody-predict ... assay_type="ELISA"
```

### Arguments Breakdown

| Argument | Description | Required? | Example |
| :--- | :--- | :--- | :--- |
| `input_file` | Path to the CSV containing sequences. | **Yes** | `data/batch_1.csv` |
| `output_file` | Path where the predictions will be written. | No | `results/batch_1_pred.csv` |
| `classifier.path` | Path to the trained model checkpoint (`.pkl` or `.joblib`). | **Yes** | `experiments/checkpoints/model.pkl` |
| `sequence_column` | Name of the column containing sequences. | No (Default: `sequence`) | `vh_seq` |
| `assay_type` | `PSR` or `ELISA` to use calibrated thresholds. | No | `PSR` |
| `threshold` | Manual probability threshold (0.0 - 1.0). | No (Default: 0.5) | `0.8` |
| `model.name` | The ESM model architecture to use. | No | `facebook/esm1v_t33_650M_UR90S_1` |

### File System Flexibility
*   **Inputs:** You do not need to put files in a special "input" folder. You can point to any file on your disk (e.g., `/Users/scientist/downloads/sequences.csv`).
*   **Outputs:** You define where the output goes. If the directory structure exists, the file will be created there.

---

## 3. Output Specification

The output file is a **CSV** that contains **all original columns** plus two new inference columns.

### New Columns

| Column Name | Type | Description |
| :--- | :--- | :--- |
| **`prediction`** | String | The binary classification: <br>• `specific`: Likely safe (Low Polyreactivity)<br>• `non-specific`: Likely unsafe (High Polyreactivity) |
| **`probability`** | Float | The confidence score (0.0 - 1.0) that the antibody is **non-specific**. <br>• Closer to `0.0` = More Specific<br>• Closer to `1.0` = More Non-Specific |

### Example Output File
```csv
id,sequence,description,prediction,probability
mAb-001,EVQL...YCAR,Primary candidate,specific,0.04
mAb-002,QVQL...YCAR,Backup sequence,non-specific,0.89
```

---

## 4. Troubleshooting

| Error Message | Cause | Solution |
| :--- | :--- | :--- |
| `ValueError: Input CSV must contain a 'sequence' column.` | The input CSV header is missing or named incorrectly. | Rename your sequence column to `sequence`. |
| `FileNotFoundError: ...` | The `input_file` path is incorrect. | Check that the file exists and the path is correct relative to your current directory. |
| `torch.cuda.OutOfMemoryError` | GPU ran out of memory. | The batch size might be too large, or the sequences too long. (CLI currently defaults to auto-config). |
