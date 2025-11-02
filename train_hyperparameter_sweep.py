#!/usr/bin/env python3
"""
Hyperparameter Sweep for Boughter LogisticRegression Model

Goal: Find optimal C, penalty, and solver to match Novo's 71% 10-CV accuracy

Current performance: 67.5% ± 8.9%
Target: 71.0% (Novo's benchmark)
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from model import ESMEmbeddingExtractor
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterSweep:
    """Run systematic hyperparameter sweep for LogisticRegression"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load training data
        train_file = self.config['data']['train_file']
        logger.info(f"Loading training data from {train_file}")
        self.train_df = pd.read_csv(train_file)

        # Extract embeddings once (cache)
        self.embeddings = None
        self.labels = None

    def extract_embeddings(self):
        """Extract ESM embeddings once for all experiments"""
        if self.embeddings is not None:
            return

        logger.info("Extracting ESM embeddings...")
        sequences = self.train_df[self.config['data']['sequence_column']].tolist()
        self.labels = self.train_df[self.config['data']['label_column']].values

        # Initialize embedding extractor
        model_name = self.config['model']['name']
        device = self.config['model']['device']
        batch_size = self.config['training']['batch_size']

        extractor = ESMEmbeddingExtractor(model_name, device, batch_size)
        self.embeddings = extractor.extract_batch_embeddings(sequences)

        logger.info(f"Extracted embeddings shape: {self.embeddings.shape}")
        logger.info(f"Labels shape: {self.labels.shape}")

    def test_hyperparameters(self, C, penalty, solver, class_weight='balanced'):
        """Test a single hyperparameter configuration"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: C={C}, penalty={penalty}, solver={solver}, class_weight={class_weight}")
        logger.info(f"{'='*70}")

        # Scale embeddings
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.embeddings)

        # Create classifier with specified hyperparameters
        try:
            clf = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=1000,
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1  # Use all cores
            )

            # 10-fold cross-validation (matching Novo's methodology)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            cv_results = cross_validate(
                clf, X_scaled, self.labels,
                cv=cv,
                scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
                return_train_score=True,
                n_jobs=-1
            )

            # Calculate statistics
            results = {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'class_weight': class_weight,
                'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
                'cv_accuracy_std': cv_results['test_accuracy'].std(),
                'cv_f1_mean': cv_results['test_f1'].mean(),
                'cv_f1_std': cv_results['test_f1'].std(),
                'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
                'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
                'cv_precision_mean': cv_results['test_precision'].mean(),
                'cv_recall_mean': cv_results['test_recall'].mean(),
                'train_accuracy_mean': cv_results['train_accuracy'].mean(),
                'train_accuracy_std': cv_results['train_accuracy'].std(),
                'overfitting_gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
            }

            logger.info(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
            logger.info(f"Train Accuracy: {results['train_accuracy_mean']:.4f} ± {results['train_accuracy_std']:.4f}")
            logger.info(f"Overfitting Gap: {results['overfitting_gap']:.4f}")
            logger.info(f"F1: {results['cv_f1_mean']:.4f}, ROC-AUC: {results['cv_roc_auc_mean']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Failed with error: {e}")
            return {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'class_weight': class_weight,
                'error': str(e)
            }

    def run_sweep(self, output_dir='hyperparameter_sweep_results'):
        """Run full hyperparameter sweep"""
        os.makedirs(output_dir, exist_ok=True)

        # Extract embeddings once
        self.extract_embeddings()

        # Define sweep grid
        sweep_grid = [
            # Priority 1: C sweep with default solver (L2 penalty)
            {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},  # Current default
            {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'},
            {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'},

            # Priority 2: Test L1 regularization with compatible solver
            {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'},
            {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'},
            {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'},

            # Priority 3: Test saga solver (supports all penalties)
            {'C': 0.01, 'penalty': 'l2', 'solver': 'saga'},
            {'C': 0.1, 'penalty': 'l2', 'solver': 'saga'},
            {'C': 1.0, 'penalty': 'l2', 'solver': 'saga'},
        ]

        results_list = []

        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING HYPERPARAMETER SWEEP")
        logger.info(f"Total configurations to test: {len(sweep_grid)}")
        logger.info(f"Baseline: 67.5% ± 8.9% (current)")
        logger.info(f"Target: 71.0% (Novo benchmark)")
        logger.info(f"{'='*70}\n")

        for i, params in enumerate(sweep_grid, 1):
            logger.info(f"\n[{i}/{len(sweep_grid)}] Testing configuration...")
            result = self.test_hyperparameters(**params)
            results_list.append(result)

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(f"{output_dir}/sweep_results_{timestamp}.csv", index=False)

        # Final summary
        results_df = pd.DataFrame(results_list)

        # Remove failed configurations
        valid_results = results_df[~results_df['cv_accuracy_mean'].isna()].copy()

        # Sort by CV accuracy
        valid_results = valid_results.sort_values('cv_accuracy_mean', ascending=False)

        logger.info(f"\n{'='*70}")
        logger.info(f"HYPERPARAMETER SWEEP COMPLETE")
        logger.info(f"{'='*70}\n")

        logger.info("TOP 5 CONFIGURATIONS:")
        logger.info(f"\n{valid_results.head(5)[['C', 'penalty', 'solver', 'cv_accuracy_mean', 'cv_accuracy_std', 'overfitting_gap']].to_string()}\n")

        # Best configuration
        best = valid_results.iloc[0]
        logger.info(f"\n{'='*70}")
        logger.info(f"BEST CONFIGURATION:")
        logger.info(f"{'='*70}")
        logger.info(f"C: {best['C']}")
        logger.info(f"Penalty: {best['penalty']}")
        logger.info(f"Solver: {best['solver']}")
        logger.info(f"CV Accuracy: {best['cv_accuracy_mean']:.4f} ± {best['cv_accuracy_std']:.4f}")
        logger.info(f"Improvement over baseline: {(best['cv_accuracy_mean'] - 0.675) * 100:+.2f}%")
        logger.info(f"Gap to Novo (71%): {(0.71 - best['cv_accuracy_mean']) * 100:.2f}%")
        logger.info(f"{'='*70}\n")

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"{output_dir}/final_sweep_results_{timestamp}.csv", index=False)

        # Save best config as YAML
        best_config = {
            'C': float(best['C']),
            'penalty': str(best['penalty']),
            'solver': str(best['solver']),
            'class_weight': str(best['class_weight']),
            'max_iter': 1000,
            'random_state': 42,
            'cv_accuracy': float(best['cv_accuracy_mean']),
            'cv_accuracy_std': float(best['cv_accuracy_std'])
        }

        with open(f"{output_dir}/best_config_{timestamp}.yaml", 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)

        logger.info(f"Results saved to {output_dir}/")

        return valid_results


if __name__ == '__main__':
    sweep = HyperparameterSweep('config_boughter.yaml')
    results = sweep.run_sweep()
