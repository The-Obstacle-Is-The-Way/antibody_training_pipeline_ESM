"""
This module contains the Gradio app for the antibody non-specificity prediction pipeline.
"""

from pathlib import Path

import gradio as gr
import hydra
from omegaconf import DictConfig

from antibody_training_esm.core.prediction import Predictor


def launch_gradio_app(cfg: DictConfig) -> None:
    """
    Launches the Gradio web UI for antibody prediction.

    This function sets up a Gradio interface that allows users to input an
    antibody sequence and receive a prediction for its non-specificity.

    Args:
        cfg: The Hydra configuration object.
    """
    if cfg.classifier.path is None:
        raise ValueError(
            "Classifier path must be specified via command-line override:\n"
            "  classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
        )
    classifier_path = Path(cfg.classifier.path)
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier file not found at {classifier_path}. "
            "Train a model (e.g., `make train`) or download a published checkpoint first."
        )

    # Instantiate the predictor
    predictor = Predictor(
        model_name=cfg.model.name,
        classifier_path=cfg.classifier.path,
    )

    def predict_sequence(sequence: str) -> tuple[str, float]:
        """
        Prediction function for the Gradio interface.

        Args:
            sequence: The antibody sequence to predict.

        Returns:
            A tuple containing the prediction string and the probability.
        """
        result = predictor.predict_single(sequence)
        return result["prediction"], result["probability"]

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict_sequence,
        inputs=gr.Textbox(lines=5, label="Antibody Sequence"),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Number(label="Probability of Non-Specificity"),
        ],
        title="Antibody Non-Specificity Predictor",
        description="Enter an antibody sequence to predict its non-specificity.",
    )

    # Launch the app
    iface.launch()


@hydra.main(config_path="../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the Gradio app."""
    launch_gradio_app(cfg)


if __name__ == "__main__":
    main()
