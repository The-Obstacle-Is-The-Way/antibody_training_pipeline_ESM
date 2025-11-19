from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from antibody_training_esm.core.prediction import run_prediction


@hydra.main(config_path="../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the prediction CLI."""
    # Validate required arguments
    if cfg.input_file is None:
        raise ValueError(
            "Input file must be specified via command-line override: `input_file=...`"
        )

    if cfg.classifier.path is None:
        raise ValueError(
            "Classifier path must be specified via command-line override:\n"
            "  classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl\n"
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
