#!/usr/bin/env python3
"""
Train model for Ginkgo 2025 Antibody Developability Competition.

This script:
1. Loads GDPa1 training dataset with predefined folds
2. Trains ESM2-650M + Ridge regressor using 5-fold CV
3. Generates submission CSVs for competition

Usage:
    # Default training (from Hydra config)
    uv run python scripts/train_ginkgo_competition.py

    # Override parameters
    uv run python scripts/train_ginkgo_competition.py regressor.alpha=0.5 hardware.device=cuda

    # Multiple runs with different alphas
    uv run python scripts/train_ginkgo_competition.py --multirun regressor.alpha=0.1,0.5,1.0,5.0,10.0
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from antibody_training_esm.core.trainer import train_model_with_predefined_folds
from antibody_training_esm.datasets.ginkgo import GinkgoDataset, GinkgoTestSet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_submission_csvs(
    model_path: str,
    oof_predictions: np.ndarray,
    train_dataset: GinkgoDataset,
    test_dataset: GinkgoTestSet,
    output_dir: str,
    property_name: str = "PR_CHO",
) -> tuple[Path, Path]:
    """
    Generate two competition submission CSVs.

    Args:
        model_path: Path to trained regressor model
        oof_predictions: Out-of-fold predictions on training set
        train_dataset: GinkgoDataset instance (for antibody names)
        test_dataset: GinkgoTestSet instance
        output_dir: Directory to save CSV files
        property_name: Property being predicted (e.g., "PR_CHO")

    Returns:
        Tuple of (cv_csv_path, test_csv_path)
    """
    from antibody_training_esm.core.regressor import AntibodyRegressor

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === 1. Generate CV predictions CSV ===
    train_df = train_dataset.load_data()

    cv_submission = pd.DataFrame(
        {
            "antibody_name": train_df["antibody_name"],
            property_name: oof_predictions,
            "hierarchical_cluster_IgG_isotype_stratified_fold": train_df["fold"],
        }
    )

    cv_csv_path = output_path / f"ginkgo_cv_predictions_{property_name}.csv"
    cv_submission.to_csv(cv_csv_path, index=False)

    logger.info(f"✅ Saved CV predictions to: {cv_csv_path}")

    # === 2. Generate test set predictions CSV ===
    model = AntibodyRegressor.load(model_path)

    test_sequences, antibody_names = test_dataset.get_sequences()

    # NOTE: Caching not implemented yet for regression pipeline
    test_predictions = model.predict(test_sequences)

    test_submission = pd.DataFrame(
        {
            "antibody_name": antibody_names,
            property_name: test_predictions,
        }
    )

    test_csv_path = output_path / f"ginkgo_test_predictions_{property_name}.csv"
    test_submission.to_csv(test_csv_path, index=False)

    logger.info(f"✅ Saved test predictions to: {test_csv_path}")

    return cv_csv_path, test_csv_path


@hydra.main(
    version_base=None,
    config_path="../src/antibody_training_esm/conf",
    config_name="ginkgo_competition",
)
def main(cfg: DictConfig) -> None:
    """Train Ginkgo competition model and generate submissions."""

    logger.info("=" * 80)
    logger.info("GINKGO 2025 ANTIBODY DEVELOPABILITY COMPETITION")
    logger.info("=" * 80)

    # Load training dataset
    logger.info("Loading GDPa1 training dataset...")
    train_dataset = GinkgoDataset(
        assay_file=cfg.data.train_assay_file,
        fold_file=cfg.data.train_fold_file,
        target_property=cfg.data.target_property,
    )

    sequences, labels, folds = train_dataset.get_sequences_and_labels()

    logger.info(f"✅ Loaded {len(sequences)} antibodies")
    logger.info(f"   Target property: {cfg.data.target_property}")
    logger.info(f"   Label range: [{labels.min():.4f}, {labels.max():.4f}]")
    logger.info(f"   Folds: {sorted(set(folds))}")

    # Train model with predefined CV folds
    logger.info("=" * 80)
    logger.info("TRAINING MODEL WITH 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 80)

    output_dir = Path(cfg.output.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cv_results, oof_predictions = train_model_with_predefined_folds(
        sequences=sequences,
        labels=labels,
        fold_assignments=folds,
        config=dict(cfg),  # Convert DictConfig to dict
        output_dir=output_dir,
    )

    # Log results
    logger.info("=" * 80)
    logger.info("📊 CROSS-VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(
        f"Overall Spearman: {cv_results['overall_spearman']:.4f} "
        f"(p-value: {cv_results['overall_spearman_pval']:.4e})"
    )
    logger.info(
        f"Mean CV Spearman: {cv_results['cv_spearman_mean']:.4f} "
        f"± {cv_results['cv_spearman_std']:.4f}"
    )

    for fold_metric in cv_results["fold_metrics"]:
        logger.info(
            f"  Fold {fold_metric['fold']}: {fold_metric['spearman']:.4f} "
            f"(n={fold_metric['n_samples']})"
        )

    # Generate submission CSVs
    logger.info("=" * 80)
    logger.info("📄 GENERATING SUBMISSION CSVs")
    logger.info("=" * 80)

    test_dataset = GinkgoTestSet(test_file=cfg.data.test_file)

    model_path = output_dir / "ginkgo_regressor.pkl"

    cv_csv, test_csv = generate_submission_csvs(
        model_path=str(model_path),
        oof_predictions=oof_predictions,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=cfg.output.submission_dir,
        property_name=cfg.data.target_property,
    )

    # Final instructions
    logger.info("=" * 80)
    logger.info("🎯 SUBMISSION FILES READY")
    logger.info("=" * 80)
    logger.info(f"✅ CV predictions: {cv_csv}")
    logger.info(f"✅ Test predictions: {test_csv}")
    logger.info("")
    logger.info("📤 Next steps:")
    logger.info(
        "1. Go to: https://huggingface.co/spaces/ginkgo-datapoints/abdev-benchmark"
    )
    logger.info("2. Click 'Submit' tab")
    logger.info("3. Upload both CSV files")
    logger.info("4. Check leaderboard for CV Spearman correlation")
    logger.info("")
    logger.info("🎯 Target to beat: 0.892 Spearman (current leader)")
    logger.info(f"📊 Your CV score: {cv_results['overall_spearman']:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
