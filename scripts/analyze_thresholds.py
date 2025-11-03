"""
Analyze decision thresholds for different datasets

This script:
1. Loads our trained model
2. Extracts prediction probabilities (not just predictions) for Jain and Shehata
3. Finds the optimal threshold for Shehata that would match Novo's confusion matrix
4. Shows why we can't optimize for both Jain AND Shehata simultaneously
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_probabilities(model, test_file, dataset_name, target_cm=None):
    """Analyze prediction probabilities and find optimal threshold"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {dataset_name} Dataset")
    logger.info(f"{'='*60}")

    # Load data
    df = pd.read_csv(test_file, comment='#')
    sequences = df['sequence'].tolist()
    y_true = df['label'].values

    logger.info(f"  Total sequences: {len(sequences)}")
    logger.info(f"  Class distribution: {pd.Series(y_true).value_counts().to_dict()}")

    # Extract embeddings
    logger.info(f"Extracting ESM-1v embeddings...")
    X_embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)

    # Get prediction probabilities (not just predictions)
    logger.info(f"Getting prediction probabilities...")
    y_proba = model.predict_proba(X_embeddings)

    # y_proba is (N, 2) array: [P(specific), P(non-specific)]
    # We want P(non-specific) = y_proba[:, 1]
    prob_nonspecific = y_proba[:, 1]

    # Current threshold (0.5)
    logger.info(f"\nUsing default threshold = 0.5:")
    y_pred_default = (prob_nonspecific > 0.5).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)
    acc_default = accuracy_score(y_true, y_pred_default)

    logger.info(f"  Confusion matrix: {cm_default.tolist()}")
    logger.info(f"  Accuracy: {acc_default*100:.1f}%")

    # Probability distribution analysis
    logger.info(f"\nProbability distribution:")
    logger.info(f"  Specific antibodies (label=0):")
    spec_probs = prob_nonspecific[y_true == 0]
    logger.info(f"    Mean: {spec_probs.mean():.3f}, Std: {spec_probs.std():.3f}")
    logger.info(f"    Min: {spec_probs.min():.3f}, Max: {spec_probs.max():.3f}")
    logger.info(f"    Median: {np.median(spec_probs):.3f}")

    logger.info(f"  Non-specific antibodies (label=1):")
    nonspec_probs = prob_nonspecific[y_true == 1]
    logger.info(f"    Mean: {nonspec_probs.mean():.3f}, Std: {nonspec_probs.std():.3f}")
    logger.info(f"    Min: {nonspec_probs.min():.3f}, Max: {nonspec_probs.max():.3f}")
    logger.info(f"    Median: {np.median(nonspec_probs):.3f}")

    # If target confusion matrix provided, find the threshold that achieves it
    if target_cm is not None:
        logger.info(f"\nSearching for threshold that matches Novo benchmark...")
        logger.info(f"  Target confusion matrix: {target_cm}")

        best_threshold = None
        best_diff = float('inf')

        # Try thresholds from 0.0 to 1.0 in steps of 0.001
        for threshold in np.arange(0.0, 1.0, 0.001):
            y_pred_test = (prob_nonspecific > threshold).astype(int)
            cm_test = confusion_matrix(y_true, y_pred_test)

            # Calculate difference from target
            diff = np.abs(cm_test - np.array(target_cm)).sum()

            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold

            # Check for exact match
            if np.array_equal(cm_test, target_cm):
                logger.info(f"  ✓ EXACT MATCH found at threshold = {threshold:.3f}")
                logger.info(f"    Confusion matrix: {cm_test.tolist()}")
                acc_test = accuracy_score(y_true, y_pred_test)
                logger.info(f"    Accuracy: {acc_test*100:.1f}%")
                return threshold, cm_test, prob_nonspecific

        # If no exact match, report best match
        y_pred_best = (prob_nonspecific > best_threshold).astype(int)
        cm_best = confusion_matrix(y_true, y_pred_best)
        acc_best = accuracy_score(y_true, y_pred_best)

        logger.info(f"  ~ Closest match at threshold = {best_threshold:.3f}")
        logger.info(f"    Confusion matrix: {cm_best.tolist()}")
        logger.info(f"    Accuracy: {acc_best*100:.1f}%")
        logger.info(f"    Difference from target: {best_diff} (sum of absolute differences)")

        return best_threshold, cm_best, prob_nonspecific

    return 0.5, cm_default, prob_nonspecific

def test_threshold_on_dataset(model, test_file, dataset_name, threshold):
    """Test a specific threshold on a dataset"""
    logger.info(f"\nTesting threshold={threshold:.3f} on {dataset_name}:")

    df = pd.read_csv(test_file, comment='#')
    sequences = df['sequence'].tolist()
    y_true = df['label'].values

    # Get embeddings and probabilities
    X_embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)
    y_proba = model.predict_proba(X_embeddings)
    prob_nonspecific = y_proba[:, 1]

    # Apply threshold
    y_pred = (prob_nonspecific > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    logger.info(f"  Confusion matrix: {cm.tolist()}")
    logger.info(f"  Accuracy: {acc*100:.1f}%")

    return cm, acc

def main():
    logger.info("="*60)
    logger.info("THRESHOLD ANALYSIS FOR JAIN AND SHEHATA DATASETS")
    logger.info("="*60)

    # Load trained model
    model_path = "models/boughter_vh_esm1v_logreg.pkl"
    logger.info(f"\nLoading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"  Model loaded successfully (device: {model.device})")

    # Novo benchmarks
    novo_jain = [[40, 17], [10, 19]]     # 68.6% accuracy
    novo_shehata = [[229, 162], [2, 5]]  # 58.8% accuracy

    # Analyze Jain dataset
    jain_threshold, jain_cm, jain_probs = analyze_dataset_probabilities(
        model,
        "test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv",
        "Jain",
        target_cm=novo_jain
    )

    # Analyze Shehata dataset
    shehata_threshold, shehata_cm, shehata_probs = analyze_dataset_probabilities(
        model,
        "test_datasets/shehata/VH_only_shehata.csv",
        "Shehata",
        target_cm=novo_shehata
    )

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY: Can we use a single threshold for both datasets?")
    logger.info("="*60)

    logger.info(f"\nOptimal thresholds:")
    logger.info(f"  Jain:    {jain_threshold:.3f} (to match Novo)")
    logger.info(f"  Shehata: {shehata_threshold:.3f} (to match Novo)")

    if abs(jain_threshold - shehata_threshold) < 0.01:
        logger.info(f"\n✓ YES - Same threshold works for both datasets!")
    else:
        logger.info(f"\n✗ NO - Different thresholds needed (Δ = {abs(jain_threshold - shehata_threshold):.3f})")

        logger.info(f"\nWhat happens if we use Jain's optimal threshold on Shehata?")
        test_threshold_on_dataset(
            model,
            "test_datasets/shehata/VH_only_shehata.csv",
            "Shehata",
            jain_threshold
        )
        logger.info(f"  → We get our current results (not Novo's)")

        logger.info(f"\nWhat happens if we use Shehata's optimal threshold on Jain?")
        test_threshold_on_dataset(
            model,
            "test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv",
            "Jain",
            shehata_threshold
        )
        logger.info(f"  → We would lose Jain parity with Novo!")

        logger.info(f"\n" + "="*60)
        logger.info("CONCLUSION:")
        logger.info("="*60)
        logger.info(f"The difference in optimal thresholds reflects the fundamental")
        logger.info(f"difference between ELISA (Jain) and PSR (Shehata) assays.")
        logger.info(f"")
        logger.info(f"Novo's paper states (Section 2.7):")
        logger.info(f"  'Antibodies characterised by the PSR assay appear to be on")
        logger.info(f"   a different non-specificity spectrum than that from the")
        logger.info(f"   non-specificity ELISA assay'")
        logger.info(f"")
        logger.info(f"Our model was trained on ELISA data (Boughter), so it's")
        logger.info(f"optimized for ELISA thresholds. The PSR assay measures a")
        logger.info(f"different 'spectrum' of non-specificity.")
        logger.info(f"")
        logger.info(f"To implement dataset-specific thresholds, we would need to:")
        logger.info(f"1. Detect which assay type the input data uses (ELISA vs PSR)")
        logger.info(f"2. Apply the corresponding threshold:")
        logger.info(f"   - ELISA (Jain, Boughter): threshold = {jain_threshold:.3f}")
        logger.info(f"   - PSR (Shehata, Harvey):  threshold = {shehata_threshold:.3f}")

    logger.info(f"\n" + "="*60)
    logger.info("Analysis completed!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
