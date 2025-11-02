#!/usr/bin/env python3
"""
Test Expert's Exact Configuration

Following external guidance with:
- C=0.1 (strong prior for high-dim ESM features)
- penalty='l2', solver='lbfgs'
- max_iter=2000, tol=1e-6 (stricter convergence)
- class_weight=None (dataset is already balanced!)
- StandardScaler preprocessing
"""

import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from model import ESMEmbeddingExtractor
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config(C, penalty, solver, max_iter, tol, class_weight, config_name):
    """Test a single configuration with 10-fold CV"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {config_name}")
    logger.info(f"{'='*70}")
    logger.info(f"C={C}, penalty={penalty}, solver={solver}")
    logger.info(f"max_iter={max_iter}, tol={tol}, class_weight={class_weight}")

    # Load config and data
    with open('config_boughter.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_file = config['data']['train_file']
    train_df = pd.read_csv(train_file)

    logger.info(f"Dataset: {len(train_df)} antibodies")

    # Check class balance
    labels = train_df[config['data']['label_column']].values
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    logger.info(f"Balance: {counts[0]/(counts[0]+counts[1])*100:.1f}% / {counts[1]/(counts[0]+counts[1])*100:.1f}%")

    # Extract embeddings
    logger.info("Extracting ESM embeddings...")
    sequences = train_df[config['data']['sequence_column']].tolist()

    model_name = config['model']['name']
    device = config['model']['device']
    batch_size = config['training']['batch_size']

    extractor = ESMEmbeddingExtractor(model_name, device, batch_size)
    embeddings = extractor.extract_batch_embeddings(sequences)

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Preprocessing with StandardScaler
    logger.info("Applying StandardScaler...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(embeddings)

    # Create classifier
    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    # 10-fold stratified CV
    logger.info("Running 10-fold stratified CV...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_results = cross_validate(
        clf, X_scaled, labels,
        cv=cv,
        scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
        return_train_score=True,
        n_jobs=-1
    )

    # Results
    results = {
        'config_name': config_name,
        'C': C,
        'penalty': penalty,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol,
        'class_weight': class_weight,
        'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_f1_mean': cv_results['test_f1'].mean(),
        'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        'overfitting_gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {config_name}")
    logger.info(f"{'='*70}")
    logger.info(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    logger.info(f"Train Accuracy: {results['train_accuracy_mean']:.4f}")
    logger.info(f"Overfitting Gap: {results['overfitting_gap']:.4f}")
    logger.info(f"F1: {results['cv_f1_mean']:.4f}, ROC-AUC: {results['cv_roc_auc_mean']:.4f}")
    logger.info(f"{'='*70}\n")

    return results

if __name__ == '__main__':
    all_results = []

    logger.info("\n" + "="*70)
    logger.info("TESTING EXPERT-GUIDED CONFIGURATIONS")
    logger.info("="*70)
    logger.info("Baseline: 67.5% ± 8.9% (C=1.0, class_weight='balanced')")
    logger.info("Target: 71.0% (Novo benchmark)")
    logger.info("="*70 + "\n")

    # Preset A: Expert's strong-prior config
    results_a = test_config(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        max_iter=2000,
        tol=1e-6,
        class_weight=None,  # KEY DIFFERENCE!
        config_name="Preset A (Expert strong-prior)"
    )
    all_results.append(results_a)

    # Test with C=0.01 too (our sweep showed it was slightly better)
    results_a2 = test_config(
        C=0.01,
        penalty='l2',
        solver='lbfgs',
        max_iter=2000,
        tol=1e-6,
        class_weight=None,
        config_name="Preset A variant (C=0.01)"
    )
    all_results.append(results_a2)

    # Preset B: L1 regularization
    results_b = test_config(
        C=0.1,
        penalty='l1',
        solver='liblinear',
        max_iter=2000,
        tol=1e-6,
        class_weight=None,
        config_name="Preset B (L1 sparse)"
    )
    all_results.append(results_b)

    # Preset C: ElasticNet
    results_c = test_config(
        C=0.1,
        penalty='elasticnet',
        solver='saga',
        max_iter=5000,  # Expert recommends more iterations for saga
        tol=1e-6,
        class_weight=None,
        config_name="Preset C (ElasticNet)"
    )
    all_results.append(results_c)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('cv_accuracy_mean', ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"expert_config_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)

    logger.info("\n" + "="*70)
    logger.info("FINAL COMPARISON")
    logger.info("="*70)
    logger.info(f"\n{results_df[['config_name', 'cv_accuracy_mean', 'cv_accuracy_std', 'overfitting_gap']].to_string(index=False)}\n")

    best = results_df.iloc[0]
    logger.info(f"="*70)
    logger.info(f"BEST CONFIGURATION: {best['config_name']}")
    logger.info(f"="*70)
    logger.info(f"CV Accuracy: {best['cv_accuracy_mean']:.4f} ± {best['cv_accuracy_std']:.4f}")
    logger.info(f"Improvement over baseline (67.5%): {(best['cv_accuracy_mean'] - 0.675)*100:+.2f}%")
    logger.info(f"Gap to Novo (71.0%): {(0.71 - best['cv_accuracy_mean'])*100:.2f}%")
    logger.info(f"="*70)

    logger.info(f"\nResults saved to: {output_file}")
