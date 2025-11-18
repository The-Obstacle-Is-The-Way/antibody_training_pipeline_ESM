
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd

from antibody_training_esm.core.prediction import run_prediction


@hydra.main(config_path="../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the prediction CLI."""
    try:
        if cfg.input_file is None:
            raise ValueError("Input file must be specified.")

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
