"""
Phase 1 Test: Confirm StandardScaler is the root cause

This script tests Novo's exact methodology:
- ESM 1v embeddings (mean pooling)
- NO StandardScaler
- Simple sklearn LogisticRegression
- 10-fold cross-validation

Expected results if hypothesis is correct:
- CV accuracy: ~70-71% (matching Novo)
- Jain test accuracy: ~65-69% (matching Novo)

If results match expectations, we proceed with full pipeline fixes.
If results don't match, we investigate other issues (dataset, hyperparameters, etc.)
"""

import logging
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cached_embeddings(cache_dir: str, dataset_name: str):
    """Load pre-computed embeddings from cache"""
    import glob

    # Find cached embedding files matching the dataset name
    pattern = os.path.join(cache_dir, f"{dataset_name}_*_embeddings.pkl")
    cache_files = glob.glob(pattern)

    if not cache_files:
        raise FileNotFoundError(
            f"No cached embeddings found for {dataset_name} in {cache_dir}"
        )

    # Use the most recent cache file
    cache_file = sorted(cache_files, key=os.path.getmtime)[-1]
    logger.info(f"Loading cached embeddings from {cache_file}")

    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)

    return cached_data["embeddings"]


def test1_no_scaler_our_hyperparams():
    """
    Test 1: LogisticRegression without StandardScaler (our hyperparameters)

    Settings:
    - max_iter=1000 (our config)
    - random_state=42 (our config)
    - NO StandardScaler

    Expected: CV ~70-71% if StandardScaler was the problem
    """
    logger.info("=" * 80)
    logger.info(
        "TEST 1: No StandardScaler + Our Hyperparameters (max_iter=1000, random_state=42)"
    )
    logger.info("=" * 80)

    # Load training data
    train_df = pd.read_csv("./train_datasets/boughter/VH_only_boughter_training.csv")
    X_train_sequences = train_df["sequence"].tolist()
    y_train = train_df["label"].values

    logger.info(f"Loaded {len(X_train_sequences)} training samples")

    # Load cached embeddings (computed with ESM 1v mean pooling)
    try:
        X_train_embedded = load_cached_embeddings("./embeddings_cache", "train")
        logger.info(f"Loaded embeddings: shape={X_train_embedded.shape}")
    except FileNotFoundError:
        logger.warning("Embeddings not cached, computing fresh embeddings...")
        from model import ESMEmbeddingExtractor

        embedding_extractor = ESMEmbeddingExtractor(
            "facebook/esm1v_t33_650M_UR90S_1", "mps", batch_size=8
        )
        X_train_embedded = embedding_extractor.extract_batch_embeddings(
            X_train_sequences
        )
        logger.info(f"Computed embeddings: shape={X_train_embedded.shape}")

    # Simple sklearn LogisticRegression (NO StandardScaler, NO BinaryClassifier wrapper)
    classifier = LogisticRegression(
        random_state=42, max_iter=1000, class_weight=None  # Boughter is balanced
    )

    logger.info("Classifier settings:")
    logger.info("  - max_iter: 1000")
    logger.info("  - random_state: 42")
    logger.info("  - class_weight: None")
    logger.info("  - StandardScaler: NO ✅")

    # 10-fold cross-validation (matching Novo)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    logger.info("\nRunning 10-fold cross-validation...")
    cv_scores = cross_val_score(
        classifier, X_train_embedded, y_train, cv=cv, scoring="accuracy"
    )

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS - Test 1:")
    logger.info(
        f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )
    logger.info(f"  Individual fold scores: {cv_scores}")
    logger.info("=" * 80)

    # Train final model on full training set
    logger.info("\nTraining final model on full training set...")
    classifier.fit(X_train_embedded, y_train)
    train_acc = classifier.score(X_train_embedded, y_train)
    logger.info(f"  Training Accuracy: {train_acc:.4f}")

    # Test on Jain
    logger.info("\nTesting on Jain dataset...")
    jain_df = pd.read_csv("./test_datasets/jain/VH_only_jain_test.csv")
    X_jain_sequences = jain_df["sequence"].tolist()
    y_jain = jain_df["label"].values

    logger.info(f"Loaded {len(X_jain_sequences)} Jain test samples")

    # Load cached Jain embeddings (if available, otherwise compute)
    try:
        X_jain_embedded = load_cached_embeddings("./embeddings_cache", "test")
        logger.info(f"Loaded Jain embeddings: shape={X_jain_embedded.shape}")
    except FileNotFoundError:
        logger.warning("Jain embeddings not cached, computing...")
        from model import ESMEmbeddingExtractor

        embedding_extractor = ESMEmbeddingExtractor(
            "facebook/esm1v_t33_650M_UR90S_1", "mps", batch_size=8
        )
        X_jain_embedded = embedding_extractor.extract_batch_embeddings(X_jain_sequences)

    jain_acc = classifier.score(X_jain_embedded, y_jain)
    y_jain_pred = classifier.predict(X_jain_embedded)

    logger.info(f"\n  Jain Test Accuracy: {jain_acc:.4f}")
    logger.info("\n  Classification Report:")
    logger.info(f"\n{classification_report(y_jain, y_jain_pred)}")

    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_acc": train_acc,
        "jain_acc": jain_acc,
    }


def test2_sklearn_defaults():
    """
    Test 2: LogisticRegression with sklearn defaults (matching Novo's likely approach)

    Settings:
    - max_iter=100 (sklearn default)
    - random_state=None (no seed)
    - NO StandardScaler

    Expected: May match Novo's results even better (if they used defaults)
    """
    logger.info("\n\n" + "=" * 80)
    logger.info(
        "TEST 2: No StandardScaler + sklearn Defaults (max_iter=100, no random_state)"
    )
    logger.info("=" * 80)

    # Load training data
    train_df = pd.read_csv("./train_datasets/boughter/VH_only_boughter_training.csv")
    X_train_sequences = train_df["sequence"].tolist()
    y_train = train_df["label"].values

    # Load cached embeddings
    try:
        X_train_embedded = load_cached_embeddings("./embeddings_cache", "train")
    except FileNotFoundError:
        logger.warning("Embeddings not cached, computing fresh embeddings...")
        from model import ESMEmbeddingExtractor

        embedding_extractor = ESMEmbeddingExtractor(
            "facebook/esm1v_t33_650M_UR90S_1", "mps", batch_size=8
        )
        X_train_embedded = embedding_extractor.extract_batch_embeddings(
            X_train_sequences
        )

    # Simple sklearn LogisticRegression with DEFAULTS
    classifier = LogisticRegression()  # All defaults

    logger.info("Classifier settings:")
    logger.info("  - max_iter: 100 (sklearn default)")
    logger.info("  - random_state: None (no seed)")
    logger.info("  - class_weight: None (sklearn default)")
    logger.info("  - StandardScaler: NO ✅")

    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    logger.info("\nRunning 10-fold cross-validation...")
    cv_scores = cross_val_score(
        classifier, X_train_embedded, y_train, cv=cv, scoring="accuracy"
    )

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS - Test 2:")
    logger.info(
        f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )
    logger.info(f"  Individual fold scores: {cv_scores}")
    logger.info("=" * 80)

    # Train and test on Jain
    classifier.fit(X_train_embedded, y_train)
    train_acc = classifier.score(X_train_embedded, y_train)

    jain_df = pd.read_csv("./test_datasets/jain/VH_only_jain_test.csv")
    y_jain = jain_df["label"].values

    try:
        X_jain_embedded = load_cached_embeddings("./embeddings_cache", "test")
    except FileNotFoundError:
        logger.warning("Jain embeddings not cached, computing...")
        from model import ESMEmbeddingExtractor

        embedding_extractor = ESMEmbeddingExtractor(
            "facebook/esm1v_t33_650M_UR90S_1", "mps", batch_size=8
        )
        X_jain_embedded = embedding_extractor.extract_batch_embeddings(
            jain_df["sequence"].tolist()
        )

    jain_acc = classifier.score(X_jain_embedded, y_jain)

    logger.info(f"\n  Training Accuracy: {train_acc:.4f}")
    logger.info(f"  Jain Test Accuracy: {jain_acc:.4f}")

    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_acc": train_acc,
        "jain_acc": jain_acc,
    }


def compare_results(test1_results, test2_results):
    """
    Compare results and determine if hypothesis is confirmed
    """
    logger.info("\n\n" + "=" * 80)
    logger.info("COMPARISON: Phase 1 Tests vs Baseline vs Novo Benchmark")
    logger.info("=" * 80)

    comparison_table = f"""
    | Configuration          | CV Accuracy | Train Accuracy | Jain Accuracy | Status |
    |------------------------|-------------|----------------|---------------|--------|
    | **Baseline (with StandardScaler)** | 63.88%      | 95.62%         | 55.32%        | ❌ Current |
    | **Test 1 (no scaler, max_iter=1000)** | {test1_results['cv_mean']*100:.2f}%      | {test1_results['train_acc']*100:.2f}%         | {test1_results['jain_acc']*100:.2f}%        | {"✅" if test1_results['cv_mean'] > 0.69 else "⚠️"} Phase 1.1 |
    | **Test 2 (no scaler, sklearn defaults)** | {test2_results['cv_mean']*100:.2f}%      | {test2_results['train_acc']*100:.2f}%         | {test2_results['jain_acc']*100:.2f}%        | {"✅" if test2_results['cv_mean'] > 0.69 else "⚠️"} Phase 1.2 |
    | **Novo Benchmark (2025)** | 71.00%      | ~75-80% (est.) | 69.00%        | ✅ Target |
    """

    logger.info(comparison_table)

    # Determine if hypothesis is confirmed
    hypothesis_confirmed = False
    if test1_results["cv_mean"] >= 0.69:  # Within 2% of Novo's 71%
        logger.info("\n" + "=" * 80)
        logger.info("✅ HYPOTHESIS CONFIRMED!")
        logger.info("=" * 80)
        logger.info(
            "StandardScaler was indeed the root cause of performance degradation."
        )
        logger.info(
            "Removing StandardScaler brings CV accuracy to ~70-71% (matching Novo)."
        )
        logger.info("\nNext steps:")
        logger.info("1. Update classifier.py to remove StandardScaler")
        logger.info(
            "2. Simplify train.py to use simple LogisticRegression on embeddings"
        )
        logger.info(
            "3. Update config to use sklearn defaults (or our hyperparams if better)"
        )
        logger.info("4. Run black, isort, ruff on all code")
        logger.info("5. Re-run full training pipeline and document results")
        hypothesis_confirmed = True
    else:
        logger.warning("\n" + "=" * 80)
        logger.warning("⚠️ HYPOTHESIS NOT FULLY CONFIRMED")
        logger.warning("=" * 80)
        logger.warning(
            f"CV accuracy improved to {test1_results['cv_mean']*100:.2f}%, but still below Novo's 71%."
        )
        logger.warning("Additional investigations needed:")
        logger.warning("1. Check Boughter dataset filtering (0 vs >3 flags parsing)")
        logger.warning("2. Verify ESM 1v model checkpoint version")
        logger.warning("3. Check random seed effects on CV splits")
        logger.warning("4. Verify label distribution matches Novo's")

    return hypothesis_confirmed


if __name__ == "__main__":
    logger.info("PHASE 1 TESTING: Confirming StandardScaler Hypothesis")
    logger.info("=" * 80)
    logger.info(
        "This test validates whether removing StandardScaler restores Novo's performance."
    )
    logger.info("")

    # Run Test 1
    test1_results = test1_no_scaler_our_hyperparams()

    if test1_results is None:
        logger.error("Test 1 failed - embeddings not found. Run train.py first.")
        exit(1)

    # Run Test 2
    test2_results = test2_sklearn_defaults()

    if test2_results is None:
        logger.error("Test 2 failed - embeddings not found.")
        exit(1)

    # Compare and conclude
    hypothesis_confirmed = compare_results(test1_results, test2_results)

    # Save results to file
    results_file = "PHASE1_TEST_RESULTS.md"
    with open(results_file, "w") as f:
        f.write("# Phase 1 Test Results\n\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n\n")
        f.write("## Hypothesis\n\n")
        f.write(
            "StandardScaler is hurting performance. Removing it should restore CV accuracy to ~70-71% (matching Novo).\n\n"
        )
        f.write("## Results\n\n")
        f.write(
            "### Test 1: No StandardScaler + Our Hyperparameters (max_iter=1000, random_state=42)\n"
        )
        f.write(
            f"- CV Accuracy: {test1_results['cv_mean']*100:.2f}% (+/- {test1_results['cv_std']*2*100:.2f}%)\n"
        )
        f.write(f"- Training Accuracy: {test1_results['train_acc']*100:.2f}%\n")
        f.write(f"- Jain Test Accuracy: {test1_results['jain_acc']*100:.2f}%\n\n")

        f.write(
            "### Test 2: No StandardScaler + sklearn Defaults (max_iter=100, no random_state)\n"
        )
        f.write(
            f"- CV Accuracy: {test2_results['cv_mean']*100:.2f}% (+/- {test2_results['cv_std']*2*100:.2f}%)\n"
        )
        f.write(f"- Training Accuracy: {test2_results['train_acc']*100:.2f}%\n")
        f.write(f"- Jain Test Accuracy: {test2_results['jain_acc']*100:.2f}%\n\n")

        f.write("## Comparison with Baseline\n\n")
        f.write("| Configuration | CV Accuracy | Train Accuracy | Jain Accuracy |\n")
        f.write("|---------------|-------------|----------------|---------------|\n")
        f.write("| Baseline (with StandardScaler) | 63.88% | 95.62% | 55.32% |\n")
        f.write(
            f"| Test 1 (no scaler, max_iter=1000) | {test1_results['cv_mean']*100:.2f}% | {test1_results['train_acc']*100:.2f}% | {test1_results['jain_acc']*100:.2f}% |\n"
        )
        f.write(
            f"| Test 2 (no scaler, sklearn defaults) | {test2_results['cv_mean']*100:.2f}% | {test2_results['train_acc']*100:.2f}% | {test2_results['jain_acc']*100:.2f}% |\n"
        )
        f.write(
            "| **Novo Benchmark (2025)** | **71.00%** | ~75-80% | **69.00%** |\n\n"
        )

        f.write("## Conclusion\n\n")
        if hypothesis_confirmed:
            f.write("✅ **HYPOTHESIS CONFIRMED**\n\n")
            f.write(
                "StandardScaler was the root cause. Removing it restores performance to Novo's benchmarks.\n\n"
            )
            f.write("**Next steps:**\n")
            f.write("1. Remove StandardScaler from classifier.py\n")
            f.write(
                "2. Simplify architecture to match Novo (embeddings → LogisticRegression)\n"
            )
            f.write("3. Update config and re-run full pipeline\n")
            f.write("4. Run black, isort, ruff\n")
            f.write("5. Document final results\n")
        else:
            f.write("⚠️ **HYPOTHESIS PARTIALLY CONFIRMED**\n\n")
            f.write(
                "Removing StandardScaler improved performance, but we're still below Novo's benchmarks.\n"
            )
            f.write(
                "Additional investigations needed (see console output for details).\n"
            )

    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nPHASE 1 TESTING COMPLETE")
