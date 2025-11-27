"""
Track B Reproducibility Script

Implements Phase B: Baseline Reproducibility for Biophysical Descriptors.
Reproduction of "Track B" from Sakhnini et al. 2025 using the Biopython Trio.

Target Metrics (from Paper Table S2):
*   Theoretical pI ALONE: 65.2% Accuracy (single descriptor baseline)

This script:
1. Loads Boughter (Train) and Jain (Test) datasets
2. extracts 3 biophysical descriptors: Charge@pH6, Charge@pH7.4, Theoretical pI
3. Trains a Logistic Regression model (StandardScaler + LogReg)
4. Evaluates using 10-fold CV and Hold-out Test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from antibody_training_esm.core.biophysical import BiophysicalExtractor
from antibody_training_esm.datasets.boughter import load_boughter_data
from antibody_training_esm.datasets.jain import load_jain_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("reproduce_track_b")


def clean_dataset(df: pd.DataFrame, sequence_col: str = "VH_sequence") -> pd.DataFrame:
    """
    Clean dataset for biophysical extraction.
    Removes sequences with 'X' (ambiguous) or empty sequences.
    """
    initial_len = len(df)

    # Filter empty
    df = df[df[sequence_col].str.len() > 0].copy()

    # Filter 'X' (BiophysicalExtractor requires exact AAs)
    # Note: BoughterDataset already filters '*' (stop codons)
    df = df[~df[sequence_col].str.contains("X", na=False)].copy()

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info(f"  Removed {dropped} sequences with 'X' or empty")

    return df


def run_reproducibility_study(output_dir: str = "experiments/benchmarks") -> None:
    """Run the full reproducibility study."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "track_b_baseline.json"

    logger.info("=" * 60)
    logger.info("PHASE B: BIOPHYSICAL BASELINE REPRODUCIBILITY")
    logger.info("=" * 60)

    # 1. Load Data
    logger.info("1. Loading Datasets...")

    # Boughter (Train)
    # Use include_mild=False to replicate paper's training set (0 vs 4+ flags)
    df_train = load_boughter_data(include_mild=False)
    df_train = clean_dataset(df_train, "VH_sequence")
    logger.info(f"  Train (Boughter): {len(df_train)} sequences")

    # Jain (Test)
    df_test = load_jain_data(stage="parity")  # Use Novo parity set (86 antibodies)
    # Ensure column mapping for Jain (it has VH_sequence)
    if "VH_sequence" not in df_test.columns and "sequence" in df_test.columns:
        df_test["VH_sequence"] = df_test["sequence"]

    df_test = clean_dataset(df_test, "VH_sequence")
    logger.info(f"  Test (Jain): {len(df_test)} sequences")

    # 2. Extract Features
    logger.info("\n2. Extracting Biophysical Features...")
    extractor = BiophysicalExtractor()

    logger.info("  Extracting training features...")
    X_train = extractor.extract_batch_features(df_train["VH_sequence"].tolist())
    y_train = df_train["label"].values

    logger.info("  Extracting test features...")
    X_test = extractor.extract_batch_features(df_test["VH_sequence"].tolist())
    y_test = df_test["label"].values

    feature_names = extractor.feature_names
    logger.info(f"  Features: {feature_names}")

    # 3. Train Model (CV)
    logger.info("\n3. Training & Cross-Validation...")

    # Pipeline: Scale -> LogReg
    # Note: Paper uses simple LogReg. Scaling is essential for convergence.
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    logger.info(
        f"  10-fold CV Accuracy (Boughter): {mean_cv:.4f} (+/- {std_cv * 2:.4f})"
    )

    # 4. Final Evaluation
    logger.info("\n4. Final Evaluation on Jain (Novo Parity)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    logger.info(f"  Test Accuracy (Jain): {test_acc:.4f}")
    logger.info(f"  Test ROC-AUC (Jain): {test_auc:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)

    # Check coefficients to see importance (pI vs others)
    logreg = pipeline.named_steps["logisticregression"]
    coefs = logreg.coef_[0]

    logger.info("\n  Feature Coefficients:")
    for name, coef in zip(feature_names, coefs, strict=True):
        logger.info(f"    {name}: {coef:.4f}")

    # 5. Save Results
    results = {
        "cv_accuracy_mean": float(mean_cv),
        "cv_accuracy_std": float(std_cv),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "coefficients": {
            name: float(coef) for name, coef in zip(feature_names, coefs, strict=True)
        },
        "report": report,
        "dataset_sizes": {"train": len(df_train), "test": len(df_test)},
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Validation against paper claims
    target_acc = 0.652
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION VERDICT")
    logger.info("=" * 60)
    logger.info(f"Target (pI only): ~{target_acc:.3f}")
    logger.info(f"Actual (3 descriptors): {mean_cv:.3f} (CV), {test_acc:.3f} (Test)")

    if 0.60 <= mean_cv <= 0.70:
        logger.info("✅ SUCCESS: Results match expected baseline range (60-70%)")
    elif mean_cv > 0.70:
        logger.warning("❓ UNEXPECTED HIGH PERFORMANCE: >70%. Check for data leakage.")
    else:
        logger.warning(
            "❌ FAILURE: <60%. Check biophysical calculation or data quality."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Track B Reproducibility Study")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    run_reproducibility_study(args.output_dir)
