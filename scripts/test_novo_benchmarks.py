"""
Test our trained model against Novo Nordisk benchmarks on Harvey and Shehata datasets
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_metrics(y_true, y_pred, dataset_name):
    """Calculate and print performance metrics"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Handle potential division by zero
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        precision = recall = f1 = specificity = 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"Results for {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  {cm}")
    logger.info(f"\n                Predicted")
    logger.info(f"                Spec  Non-spec")
    logger.info(f"True    Spec    {cm[0,0]:5d}  {cm[0,1]:5d}     ({cm[0,0] + cm[0,1]} specific)")
    logger.info(f"        Non-spec{cm[1,0]:5d}  {cm[1,1]:5d}     ({cm[1,0] + cm[1,1]} non-specific)")
    logger.info(f"\nTotal sequences: {len(y_true)}")
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Accuracy:    {accuracy*100:.1f}% ({np.sum(y_pred == y_true)}/{len(y_true)})")
    logger.info(f"  Sensitivity: {recall*100:.1f}% (TP/(TP+FN))")
    logger.info(f"  Specificity: {specificity*100:.1f}% (TN/(TN+FP))")
    logger.info(f"  Precision:   {precision*100:.1f}% (TP/(TP+FP))")
    logger.info(f"  F1-Score:    {f1*100:.1f}%")
    logger.info(f"{'='*60}\n")

    return cm, accuracy, precision, recall, f1, specificity

def test_dataset(model, test_file, dataset_name):
    """Test model on a dataset"""
    logger.info(f"Loading {dataset_name} dataset from {test_file}...")
    df = pd.read_csv(test_file, comment='#')

    logger.info(f"  Total sequences: {len(df)}")
    logger.info(f"  Class distribution: {df['label'].value_counts().to_dict()}")

    # Extract sequences and labels
    sequences = df['sequence'].tolist()
    y_true = df['label'].values

    # Extract embeddings
    logger.info(f"Extracting ESM-1v embeddings for {len(sequences)} sequences...")
    logger.info("  (This may take a while for large datasets...)")
    X_embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)

    # Run inference
    logger.info(f"Running inference...")
    y_pred = model.predict(X_embeddings)

    # Calculate and print metrics
    metrics = print_metrics(y_true, y_pred, dataset_name)

    return metrics

def compare_to_novo(cm, dataset_name, novo_cm):
    """Compare our results to Novo benchmarks"""
    logger.info(f"\nComparison to Novo Benchmark for {dataset_name}:")
    logger.info(f"  Our confusion matrix:   {cm.tolist()}")
    logger.info(f"  Novo confusion matrix:  {novo_cm}")

    # Calculate differences
    diff = cm - np.array(novo_cm)
    logger.info(f"  Difference (ours - Novo): {diff.tolist()}")

    # Check if matrices match
    if np.array_equal(cm, novo_cm):
        logger.info(f"  ✓ EXACT MATCH with Novo benchmark!")
    else:
        our_acc = (cm[0,0] + cm[1,1]) / cm.sum()
        novo_acc = (novo_cm[0][0] + novo_cm[1][1]) / np.array(novo_cm).sum()
        diff_acc = abs(our_acc - novo_acc) * 100
        logger.info(f"  Accuracy difference: {diff_acc:.1f} percentage points")
        if diff_acc < 2.0:
            logger.info(f"  ✓ Close match (within 2%)")
        elif diff_acc < 5.0:
            logger.info(f"  ~ Reasonable match (within 5%)")
        else:
            logger.info(f"  ✗ Significant difference (>{5}%)")

def main():
    logger.info("="*60)
    logger.info("Testing Novo Nordisk Benchmark Reproduction")
    logger.info("="*60)

    # Load trained model
    model_path = "models/boughter_vh_esm1v_logreg.pkl"
    logger.info(f"\nLoading trained model from {model_path}...")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"  Model loaded successfully")
    logger.info(f"  Device: {model.device}")
    logger.info(f"  Model name: {model.model_name}")

    # Novo benchmarks (from Figure S14)
    novo_benchmarks = {
        'Jain': [[40, 17], [10, 19]],      # 86 antibodies, 68.6% accuracy
        'Shehata': [[229, 162], [2, 5]],   # 398 antibodies, 58.8% accuracy
        'Harvey': [[19778, 49962], [4186, 67633]]  # 141,559 nanobodies, 61.7% accuracy
    }

    results = {}

    # Test on Shehata (smaller dataset, faster)
    logger.info("\n" + "="*60)
    logger.info("TESTING ON SHEHATA DATASET")
    logger.info("="*60)
    shehata_metrics = test_dataset(
        model,
        "test_datasets/shehata/VH_only_shehata.csv",
        "Shehata"
    )
    results['Shehata'] = shehata_metrics
    compare_to_novo(shehata_metrics[0], "Shehata", novo_benchmarks['Shehata'])

    # Test on Harvey (very large dataset, will take time)
    logger.info("\n" + "="*60)
    logger.info("TESTING ON HARVEY DATASET")
    logger.info("="*60)
    logger.info("WARNING: Harvey dataset has 141,000+ sequences. This will take significant time.")
    logger.info("Proceeding with full dataset test...\n")

    harvey_metrics = test_dataset(
        model,
        "test_datasets/harvey/VHH_only_harvey.csv",
        "Harvey"
    )
    results['Harvey'] = harvey_metrics
    compare_to_novo(harvey_metrics[0], "Harvey", novo_benchmarks['Harvey'])

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("\nComparison to Novo Benchmarks:")
    logger.info(f"\n{'Dataset':<12} {'Our Accuracy':<15} {'Novo Accuracy':<15} {'Difference'}")
    logger.info("-" * 60)

    for dataset, metrics in results.items():
        cm, acc, _, _, _, _ = metrics
        novo_cm = np.array(novo_benchmarks[dataset])
        novo_acc = (novo_cm[0,0] + novo_cm[1,1]) / novo_cm.sum()
        diff = abs(acc - novo_acc) * 100

        logger.info(f"{dataset:<12} {acc*100:>6.1f}%         {novo_acc*100:>6.1f}%         {diff:>5.1f}pp")

    logger.info("\n" + "="*60)
    logger.info("Testing completed!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
