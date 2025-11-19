"""
This module contains the Gradio app for the antibody non-specificity prediction pipeline.
"""

import platform
from pathlib import Path

import gradio as gr
import hydra
import torch
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
    # Robust Device & Threading Configuration
    # -------------------------------------------------------------------------
    # 1. Determine the optimal device for inference
    #    - Prefer CUDA if available (Linux/Windows GPU boxes)
    #    - Force CPU on macOS if MPS is detected to avoid Gradio+MPS SegFaults
    #    - Default to configured value otherwise
    device = cfg.model.get("device", "cpu")

    if platform.system() == "Darwin" and device == "mps":
        print(
            "WARNING: macOS detected. Forcing CPU for Gradio app stability (MPS workaround)."
        )
        device = "cpu"

    # 2. Configure Threading to prevent OpenMP SegFaults on macOS
    #    - On macOS/CPU, PyTorch's OpenMP runtime can crash inside Gradio threads.
    #    - We restrict it to 1 thread to ensure stability.
    #    - Linux/CUDA systems remain untouched and can use full parallelism.
    if platform.system() == "Darwin" and device == "cpu":
        print(
            "WARNING: macOS/CPU detected. Setting torch.set_num_threads(1) to prevent OpenMP crashes."
        )
        torch.set_num_threads(1)

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
    config_path = getattr(cfg.classifier, "config_path", None)
    predictor = Predictor(
        model_name=cfg.model.name,
        classifier_path=cfg.classifier.path,
        device=device,
        config_path=config_path,
    )

    def validate_input(sequence: str) -> None:
        """
        Validates that the input sequence contains only valid amino acids.
        """
        if not sequence:
            raise ValueError("Input sequence cannot be empty.")

        # Standard 20 amino acids + X (unknown)
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        invalid_chars = set(sequence) - valid_chars

        if invalid_chars:
            raise ValueError(
                f"Invalid characters found: {', '.join(sorted(invalid_chars))}"
            )

    def predict_sequence(sequence: str) -> tuple[str, str]:
        """
        Prediction function for the Gradio interface.

        Args:
            sequence: The antibody sequence to predict.

        Returns:
            A tuple containing the prediction string and the formatted probability.
        """
        try:
            # Clean input
            cleaned_seq = sequence.strip().upper()

            # Validate
            validate_input(cleaned_seq)

            # Log request (observability)
            print(f"Processing sequence length: {len(cleaned_seq)}")

            # Predict
            result = predictor.predict_single(cleaned_seq)

            # Format probability
            prob_percent = f"{result['probability']:.1%}"

            return result["prediction"], prob_percent

        except ValueError as e:
            raise gr.Error(str(e)) from e
        except Exception as e:
            raise gr.Error(f"Prediction failed: {str(e)}") from e

    # Example sequences
    examples = [
        [
            "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS"
        ],
        [
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
        ],
    ]

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict_sequence,
        inputs=gr.Textbox(
            lines=5,
            label="Antibody Sequence (VH or VL)",
            placeholder="Paste amino acid sequence here (e.g., QVQL...)",
            info="Supported characters: Standard amino acids (ACDEFGHIKLMNPQRSTVWY).",
        ),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Probability of Non-Specificity"),
        ],
        title="Antibody Non-Specificity Predictor",
        description=(
            "Enter an antibody Variable Heavy (VH) or Variable Light (VL) sequence "
            "to predict its non-specificity (polyreactivity)."
        ),
        article=f"Model: {cfg.model.name} | Device: {device}",
        examples=examples,
        cache_examples=True,
        flagging_mode="never",
        analytics_enabled=False,
    )

    # Enable queueing for concurrency management
    iface.queue(default_concurrency_limit=2, max_size=10)

    # Launch the app with hardened settings
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )


@hydra.main(config_path="../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the Gradio app."""
    launch_gradio_app(cfg)


if __name__ == "__main__":
    main()
