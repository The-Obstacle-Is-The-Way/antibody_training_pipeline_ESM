import sys
from pathlib import Path
from typing import cast

import hydra
import pandas as pd
from omegaconf import DictConfig
from pydantic import ValidationError

from antibody_training_esm.core.config import SEQUENCE_PREVIEW_LENGTH
from antibody_training_esm.core.prediction import Predictor, run_prediction
from antibody_training_esm.models.prediction import AssayType, PredictionRequest


def predict_sequence_cli(
    sequence: str, threshold: float, assay_type: AssayType | None, cfg: DictConfig
) -> None:
    """CLI prediction with Pydantic validation."""
    config_path = getattr(cfg.classifier, "config_path", None)
    # Respect explicit model.device override; fall back to hardware.device
    requested_device = getattr(cfg.model, "device", None) or getattr(
        getattr(cfg, "hardware", None), "device", None
    )

    # Instantiate predictor (loading model)
    try:
        predictor = Predictor(
            model_name=cfg.model.name,
            classifier_path=cfg.classifier.path,
            device=requested_device,
            config_path=config_path,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        request = PredictionRequest(
            sequence=sequence,
            threshold=threshold,
            assay_type=assay_type,
        )
        result = predictor.predict_single(request)

        # Print formatted output
        print(
            f"Sequence: {result.sequence[:SEQUENCE_PREVIEW_LENGTH]}..."
            if len(result.sequence) > SEQUENCE_PREVIEW_LENGTH
            else f"Sequence: {result.sequence}"
        )
        print(f"Prediction: {result.prediction}")
        print(f"Probability: {result.probability:.2%}")

    except ValidationError as e:
        print("❌ Validation Error:")
        for error in e.errors():
            # loc is a tuple, e.g. ('sequence',)
            loc = error["loc"][0] if error["loc"] else "root"
            print(f"  - {loc}: {error['msg']}")
        sys.exit(1)


@hydra.main(config_path="../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the prediction CLI."""

    # Check for single sequence prediction mode
    sequence = getattr(cfg, "sequence", None)
    if sequence:
        threshold = getattr(cfg, "threshold", 0.5)
        assay_type = cast(AssayType | None, getattr(cfg, "assay_type", None))
        predict_sequence_cli(sequence, threshold, assay_type, cfg)
        return

    # Validate required arguments for batch mode
    if cfg.input_file is None:
        raise ValueError(
            "Input file must be specified via command-line override: `input_file=...`"
        )

    if cfg.classifier.path is None:
        raise ValueError(
            "Classifier path must be specified via command-line override:\n"
            "  classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl\n"
            "  # OR for production models (.npz):\n"
            "  classifier.path=experiments/.../model.npz classifier.config_path=.../model_config.json\n"
            "\nExample usage:\n"
            "  uv run antibody-predict \\\n"
            "      input_file=data/test.csv \\\n"
            "      output_file=predictions.csv \\\n"
            "      classifier.path=path/to/model.pkl"
        )
    classifier_path = Path(cfg.classifier.path)
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier file not found at {classifier_path}. "
            "Train a model (e.g., `make train`) or download a published checkpoint first."
        )

    try:
        # Load input data
        input_df = pd.read_csv(cfg.input_file)

        # Run prediction
        output_df = run_prediction(input_df, cfg)

        # Save output data
        output_df.to_csv(cfg.output_file, index=False)

        print(f"Predictions saved to {cfg.output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {cfg.input_file}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
