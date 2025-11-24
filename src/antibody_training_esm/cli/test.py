#!/usr/bin/env python3
"""
Test CLI for Antibody Classification Pipeline

Professional command-line interface for testing trained antibody classifiers:
1. Load trained models from pickle files
2. Evaluate on test datasets with performance metrics
3. Generate confusion matrices and comprehensive logging

Usage:
    antibody-test --model experiments/checkpoints/antibody_classifier.pkl --data sample_data.csv
    antibody-test --config test_config.yaml
    antibody-test --model m1.pkl m2.pkl --data d1.csv d2.csv
"""

import argparse
import sys

from antibody_training_esm.cli.testing.config import (
    TestConfig,
    create_sample_test_config,
    load_config_file,
)
from antibody_training_esm.cli.testing.tester import ModelTester


def main() -> int:
    """Main entry point for antibody-test CLI"""
    parser = argparse.ArgumentParser(
        description="Testing for antibody classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single model on single dataset (auto-detects threshold from dataset name)
    antibody-test --model experiments/checkpoints/antibody_classifier.pkl --data sample_data.csv

    # Test on PSR dataset with auto-detected threshold (0.5495 for Harvey/Shehata)
    antibody-test --model model.pkl --data data/test/harvey/fragments/VHH_only_harvey.csv

    # Test multiple models on multiple datasets
    antibody-test --model experiments/checkpoints/model1.pkl experiments/checkpoints/model2.pkl --data dataset1.csv dataset2.csv

    # Use configuration file
    antibody-test --config test_config.yaml

    # Override device, batch size, and threshold
    antibody-test --config test_config.yaml --device cuda --batch-size 64 --threshold 0.6

    # Create sample configuration
    antibody-test --create-config
        """,
    )

    parser.add_argument(
        "--model", nargs="+", help="Path(s) to trained model pickle files"
    )
    parser.add_argument("--data", nargs="+", help="Path(s) to test dataset CSV files")
    parser.add_argument("--config", help="Path to test configuration YAML file")
    parser.add_argument(
        "--output-dir",
        default="./experiments/benchmarks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for inference: auto (CUDA > MPS > CPU), or explicit (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for embedding extraction (overrides config)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Manual decision threshold override (default: auto-detect from dataset name). "
        "Use 0.5 for ELISA datasets (Boughter, Jain) or 0.5495 for PSR datasets (Harvey, Shehata).",
    )
    parser.add_argument(
        "--sequence-column",
        type=str,
        help="Column name for sequences in dataset (default: 'sequence', overrides config)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        help="Column name for labels in dataset (default: 'label', overrides config)",
    )
    parser.add_argument(
        "--create-config", action="store_true", help="Create sample configuration file"
    )

    args = parser.parse_args()

    # Create sample config if requested
    if args.create_config:
        create_sample_test_config()
        return 0

    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        if not args.model or not args.data:
            parser.error("Either --config or both --model and --data must be specified")

        config = TestConfig(
            model_paths=args.model, data_paths=args.data, output_dir=args.output_dir
        )

    # Override config with command line arguments
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.threshold:
        config.threshold = args.threshold
    if args.sequence_column:
        config.sequence_column = args.sequence_column
    if args.label_column:
        config.label_column = args.label_column

    # Run testing
    try:
        tester = ModelTester(config)
        results = tester.run_comprehensive_test()

        print(f"\n{'=' * 60}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"Results saved to: {config.output_dir}")

        # Print summary
        for dataset_name, dataset_results in results.items():
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
            for model_name, model_results in dataset_results.items():
                print(f"Model: {model_name}")
                if "test_scores" in model_results:
                    for metric, value in model_results["test_scores"].items():
                        print(f"  {metric}: {value:.4f}")

        return 0

    except KeyboardInterrupt:
        print("Error during testing: Interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during testing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
