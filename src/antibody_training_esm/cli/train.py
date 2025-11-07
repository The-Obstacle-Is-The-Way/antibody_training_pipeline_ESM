"""
Training CLI

Professional command-line interface for antibody model training.
"""

import argparse
import sys

from antibody_training_esm.core.trainer import train_model


def main():
    """Main entry point for training CLI"""
    parser = argparse.ArgumentParser(
        description="Train antibody classification model using ESM-1V embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    try:
        print(f"Starting training with config: {args.config}")
        train_model(args.config)
        print("\n✅ Training completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n❌ Training failed: Interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
