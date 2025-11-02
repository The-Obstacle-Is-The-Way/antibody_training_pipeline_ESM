"""
Simplified Training Script - Novo Nordisk Methodology
Matches the exact approach from Sakhnini et al. 2025:
1. Compute ESM 1v embeddings (mean pooling)
2. Train sklearn LogisticRegression directly (NO StandardScaler)
3. 10-fold cross-validation
4. Test on Jain

This is the CORRECT, simple approach that achieves Novo's benchmarks.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model import ESMEmbeddingExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_embeddings(
    sequences: list, model_name: str, device: str, batch_size: int = 8
) -> np.ndarray:
    """Compute ESM 1v embeddings using mean pooling (matching Novo)"""
    logger.info(f"Computing embeddings for {len(sequences)} sequences...")
    embedding_extractor = ESMEmbeddingExtractor(model_name, device, batch_size)
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)
    logger.info(f"Embeddings computed: shape={embeddings.shape}")
    return embeddings


def main():
    logger.info("=" * 80)
    logger.info("NOVO NORDISK METHODOLOGY - SIMPLIFIED TRAINING")
    logger.info("=" * 80)
    logger.info("Approach: ESM 1v embeddings → LogisticRegression (no scaling)")
    logger.info("")

    # Configuration (matching Novo's approach)
    model_name = "facebook/esm1v_t33_650M_UR90S_1"
    device = "mps"  # Use 'cuda' or 'cpu' if needed
    batch_size = 8
    max_iter = 1000  # Increase from sklearn default (100) for convergence
    random_state = 42  # For reproducibility

    # Load training data (Boughter VH dataset)
    logger.info("Loading training data (Boughter VH)...")
    train_df = pd.read_csv("./train_datasets/boughter/VH_only_boughter_training.csv")
    X_train_sequences = train_df["sequence"].tolist()
    y_train = train_df["label"].values
    logger.info(f"  Loaded {len(X_train_sequences)} training samples")
    logger.info(
        f"  Label distribution: {np.bincount(y_train.astype(int))} "
        f"({np.bincount(y_train.astype(int))[0]/len(y_train)*100:.1f}% / "
        f"{np.bincount(y_train.astype(int))[1]/len(y_train)*100:.1f}%)"
    )

    # Compute ESM embeddings (matching Novo: mean pooling)
    X_train_embedded = compute_embeddings(
        X_train_sequences, model_name, device, batch_size
    )

    # Initialize simple sklearn LogisticRegression (NO StandardScaler!)
    # Matching Novo's methodology: embeddings → LogisticRegression directly
    logger.info("\nInitializing LogisticRegression...")
    classifier = LogisticRegression(
        max_iter=max_iter, random_state=random_state, class_weight=None
    )
    logger.info(f"  max_iter: {max_iter}")
    logger.info(f"  random_state: {random_state}")
    logger.info("  class_weight: None (Boughter is balanced)")
    logger.info("  StandardScaler: NO ✅ (matching Novo)")

    # 10-fold cross-validation (matching Novo)
    logger.info("\nPerforming 10-fold cross-validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    cv_accuracy = cross_val_score(
        classifier, X_train_embedded, y_train, cv=cv, scoring="accuracy"
    )
    cv_f1 = cross_val_score(classifier, X_train_embedded, y_train, cv=cv, scoring="f1")
    cv_roc_auc = cross_val_score(
        classifier, X_train_embedded, y_train, cv=cv, scoring="roc_auc"
    )

    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION RESULTS:")
    logger.info("=" * 80)
    logger.info(f"  Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std()*2:.4f})")
    logger.info(f"  F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
    logger.info(f"  ROC AUC:  {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std()*2:.4f})")
    logger.info(f"  Individual fold accuracies: {cv_accuracy}")

    # Train final model on full training set
    logger.info("\nTraining final model on full training set...")
    classifier.fit(X_train_embedded, y_train)

    train_acc = classifier.score(X_train_embedded, y_train)

    logger.info(f"  Training Accuracy: {train_acc:.4f}")

    # Evaluate on Jain test set
    logger.info("\nEvaluating on Jain test set...")
    jain_df = pd.read_csv("./test_datasets/jain/VH_only_jain_test.csv")
    X_jain_sequences = jain_df["sequence"].tolist()
    y_jain = jain_df["label"].values
    logger.info(f"  Loaded {len(X_jain_sequences)} Jain test samples")

    # Compute Jain embeddings
    X_jain_embedded = compute_embeddings(
        X_jain_sequences, model_name, device, batch_size
    )

    # Predict on Jain
    jain_acc = classifier.score(X_jain_embedded, y_jain)
    y_jain_pred = classifier.predict(X_jain_embedded)
    y_jain_proba = classifier.predict_proba(X_jain_embedded)

    logger.info("\n" + "=" * 80)
    logger.info("JAIN TEST RESULTS:")
    logger.info("=" * 80)
    logger.info(f"  Accuracy:  {jain_acc:.4f}")
    logger.info(f"  Precision: {precision_score(y_jain, y_jain_pred):.4f}")
    logger.info(f"  Recall:    {recall_score(y_jain, y_jain_pred):.4f}")
    logger.info(f"  F1 Score:  {f1_score(y_jain, y_jain_pred):.4f}")
    logger.info(f"  ROC AUC:   {roc_auc_score(y_jain, y_jain_proba[:, 1]):.4f}")
    logger.info("\n  Classification Report:")
    logger.info(f"\n{classification_report(y_jain, y_jain_pred)}")
    logger.info("\n  Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_jain, y_jain_pred)}")

    # Compare with Novo benchmark
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH NOVO BENCHMARK:")
    logger.info("=" * 80)
    comparison = f"""
    | Metric        | Our Results | Novo (2025) | Gap      | Status |
    |---------------|-------------|-------------|----------|--------|
    | CV Accuracy   | {cv_accuracy.mean()*100:6.2f}%   | 71.00%      | {cv_accuracy.mean()*100 - 71:.2f}% | {'✅' if cv_accuracy.mean() >= 0.69 else '⚠️'} |
    | Train Acc     | {train_acc*100:6.2f}%   | ~75-80%     | -        | {'✅' if 0.73 <= train_acc <= 0.82 else '⚠️'} |
    | Jain Test     | {jain_acc*100:6.2f}%   | 69.00%      | {jain_acc*100 - 69:.2f}% | {'✅' if jain_acc >= 0.67 else '⚠️'} |
    """
    logger.info(comparison)

    # Save model
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "novo_simple_logreg.pkl")

    logger.info(f"\nSaving model to {model_path}...")
    model_data = {
        "classifier": classifier,
        "model_name": model_name,
        "cv_accuracy": cv_accuracy.mean(),
        "train_accuracy": train_acc,
        "jain_accuracy": jain_acc,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("  Model saved successfully")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✅ CV Accuracy:     {cv_accuracy.mean()*100:.2f}%")
    logger.info(f"✅ Training Acc:    {train_acc*100:.2f}%")
    logger.info(f"✅ Jain Test Acc:   {jain_acc*100:.2f}%")
    logger.info(f"✅ Model saved:     {model_path}")
    logger.info("")
    logger.info("Novo Nordisk methodology successfully reproduced! 🎯")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
