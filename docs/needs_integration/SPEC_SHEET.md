# Specification Sheet: CLI Inference Pipeline

## 1. Overview

This document outlines the plan to build a command-line interface (CLI) for the antibody non-specificity prediction pipeline. The goal is to create a user-friendly tool that allows researchers to get predictions for their own antibody sequences using pre-trained models from this project. The development will follow Test-Driven Development (TDD) principles and adhere to the coding standards outlined in `AGENTS.md`.

## 2. Core Features

The CLI will provide the following functionalities:

*   **Model Selection:** Allow users to select a pre-trained ESM model from an available "model zoo."
*   **Classifier Head Selection:** Allow users to select a compatible classifier head.
*   **CSV Input:** Accept a CSV file as input, containing antibody sequences.
*   **Prediction Output:** Generate a CSV file with the original data plus the added prediction and probability scores.
*   **Configuration:** Use the existing Hydra configuration system for managing models, classifiers, and other parameters.

## 3. Command-Line Interface (CLI) Design

A new CLI entry point will be created at `src/antibody_training_esm/cli/predict.py`. The command will be invoked as follows:

```bash
uv run antibody-predict input_file=path/to/your/input.csv output_file=path/to/your/predictions.csv
```

### Arguments:

*   `input_file`: Path to the input CSV file.
*   `output_file`: Path to save the output CSV file with predictions.
*   Other Hydra overrides for model and classifier selection.

## 4. Input CSV Format

The input CSV file must contain a column named `sequence` which holds the antibody amino acid sequences. Other columns will be preserved in the output.

**Example `input.csv`:**

```csv
id,sequence,source
antibody1,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS,lab1
antibody2,QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS,lab2
```

## 5. Output CSV Format

The output CSV will contain all the columns from the input file, with two additional columns:

*   `prediction`: The predicted class (`specific` or `non-specific`).
*   `probability`: The model's confidence score for the prediction (a float between 0 and 1).

**Example `output.csv`:**

```csv
id,sequence,source,prediction,probability
antibody1,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS,lab1,non-specific,0.85
antibody2,QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS,lab2,specific,0.23
```

## 6. Model Zoo and Configuration

The "model zoo" will be managed through Hydra configurations. Pre-trained models and classifier heads will be defined in YAML files within the `src/antibody_training_esm/conf` directory. This allows for easy extension and selection of different models and classifiers.

A new `predict.yaml` configuration file will be created in `src/antibody_training_esm/conf` to define the default prediction settings.

## 7. Implementation Details

The implementation will be broken down into the following steps:

1.  **Create `predict.py`:** Create the new CLI entry point in `src/antibody_training_esm/cli/predict.py`. This script will handle command-line argument parsing and orchestrate the prediction process.
2.  **Develop Prediction Logic:** Implement the core prediction logic in `src/antibody_training_esm/core/prediction.py`. This module will be responsible for:
    *   Loading the selected ESM model and classifier head.
    *   Reading and validating the input CSV.
    *   Generating embeddings for the antibody sequences.
    *   Making predictions using the classifier.
    *   Returning the results in a structured format.
3.  **Add Unit Tests:** Create a new test file `tests/unit/test_prediction.py` to test the prediction logic in isolation.
4.  **Add Integration Tests:** Create a new test file `tests/integration/test_predict_cli.py` to test the `predict.py` CLI script end-to-end.
5.  **Update Documentation:** Update the `README.md` and `docs/` to include instructions on how to use the new prediction CLI.
6.  **Pre-Commit Checks:** Run all necessary pre-commit checks to ensure code quality and test coverage.
7.  **Submit:** Submit the final changes with a descriptive commit message.
