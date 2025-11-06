"""
Preprocessing CLI

Professional command-line interface for dataset preprocessing.
"""

import argparse
import sys


def main():
    """Main entry point for preprocessing CLI"""
    parser = argparse.ArgumentParser(
        description="Preprocess antibody datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=["jain", "harvey", "shehata", "boughter"],
        help="Dataset to preprocess",
    )

    parser.add_argument(
        "--step",
        "-s",
        type=str,
        help="Specific preprocessing step to run",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for preprocessed data",
    )

    args = parser.parse_args()

    try:
        print("\n⚠️  The 'antibody-preprocess' CLI is not implemented")
        print(
            "\nDataset preprocessing is handled by specialized scripts, not this CLI."
        )
        print(
            "These scripts are the authoritative source of truth for data transformation."
        )
        print(f"\nFor {args.dataset} dataset, use:")

        script_paths = {
            "jain": "preprocessing/jain/step2_preprocess_p5e_s2.py",
            "harvey": "preprocessing/harvey/step2_extract_fragments.py",
            "shehata": "preprocessing/shehata/step2_extract_fragments.py",
            "boughter": "preprocessing/boughter/stage2_stage3_annotation_qc.py",
        }

        script = script_paths.get(args.dataset)
        if script:
            print(f"  python {script}")

        print("\nWhy use scripts instead of this CLI?")
        print("  • Scripts are Single Source of Truth (SSOT) for preprocessing")
        print(
            "  • Each dataset has unique requirements (DNA translation, PSR thresholds, etc.)"
        )
        print("  • Scripts maintain bit-for-bit parity with published methods")
        print("  • CLI is for loading preprocessed data, not creating it")

        print("\nFor more information:")
        print("  • See src/antibody_training_esm/datasets/README.md")
        print("  • See docs/boughter/boughter_data_sources.md (dataset-specific)")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
