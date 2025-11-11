"""
Training CLI - Hydra Entry Point

Professional command-line interface for antibody model training.
Uses Hydra for configuration management and supports dynamic overrides.

Usage:
    # Default config
    antibody-train

    # With overrides
    antibody-train hardware.device=cuda training.batch_size=16

    # Multi-run sweep
    antibody-train --multirun classifier.C=0.1,1.0,10.0

    # Help
    antibody-train --help
"""

from antibody_training_esm.core.trainer import main as hydra_main


def main() -> None:
    """
    Main entry point for training CLI

    Delegates to Hydra-decorated main() in core.trainer.
    This provides automatic config composition, override support,
    and multi-run sweeps.

    Note:
        This function does not return an exit code (Hydra handles that).
        Use try/except at a higher level if you need custom error handling.
    """
    # Delegate to Hydra entry point
    # Hydra automatically parses sys.argv and handles all CLI logic
    hydra_main()


if __name__ == "__main__":
    main()
