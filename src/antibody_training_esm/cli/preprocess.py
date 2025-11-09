"""
Preprocessing CLI

Professional command-line interface for dataset preprocessing.
"""

import argparse
import sys


def main() -> int:
    """
    Main entry point for preprocessing CLI.

    This CLI does NOT run preprocessing - it only provides guidance on which
    preprocessing scripts to use. Preprocessing is handled by specialized
    scripts that are the Single Source of Truth (SSOT).
    """
    parser = argparse.ArgumentParser(
        description="Antibody dataset preprocessing guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: This CLI does NOT run preprocessing. It provides guidance on which
preprocessing scripts to use. Each dataset has unique requirements and the
scripts maintain bit-for-bit parity with published methods.
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=["jain", "harvey", "shehata", "boughter"],
        help="Dataset to get preprocessing guidance for",
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

    except KeyboardInterrupt:
        print("\n❌ Error: Interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
