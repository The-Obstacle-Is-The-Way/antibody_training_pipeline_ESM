"""
This module contains the Gradio app for the antibody non-specificity prediction pipeline.
"""

import logging
import platform
from pathlib import Path

import gradio as gr
import hydra
import torch
from omegaconf import DictConfig
from pydantic import ValidationError

from antibody_training_esm.core.prediction import Predictor
from antibody_training_esm.models.prediction import PredictionRequest

# Configure logging
logger = logging.getLogger(__name__)


def launch_gradio_app(cfg: DictConfig) -> None:
    """
    Launches the Gradio web UI for antibody prediction.

    This function sets up a Gradio interface that allows users to input an
    antibody sequence and receive a prediction for its non-specificity.

    Args:
        cfg: The Hydra configuration object.
    """
    # Set log level from config
    logging.basicConfig(
        level=getattr(logging, cfg.gradio.log_level.upper(), logging.INFO)
    )

    # Robust Device & Threading Configuration
    # -------------------------------------------------------------------------
    # 1. Determine the optimal device for inference
    #    - Prefer CUDA if available (Linux/Windows GPU boxes)
    #    - Force CPU on macOS if MPS is detected to avoid Gradio+MPS SegFaults
    #    - Default to configured value otherwise
    device = cfg.model.get("device", "cpu")

    if platform.system() == "Darwin" and device == "mps":
        logger.warning(
            "macOS detected. Forcing CPU for Gradio app stability (MPS workaround)."
        )
        device = "cpu"

    # 2. Configure Threading to prevent OpenMP SegFaults on macOS
    #    - On macOS/CPU, PyTorch's OpenMP runtime can crash inside Gradio threads.
    #    - We restrict it to 1 thread to ensure stability.
    #    - Linux/CUDA systems remain untouched and can use full parallelism.
    if platform.system() == "Darwin" and device == "cpu":
        logger.warning(
            "macOS/CPU detected. Setting torch.set_num_threads(1) to prevent OpenMP crashes."
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

    # Warm-up: Run a dummy prediction to load the model into memory eagerly
    try:
        logger.info("Warming up model with dummy prediction...")
        predictor.predict_single("QVQL")
        logger.info("Model warmed up and ready.")
    except Exception as e:
        logger.warning(f"Model warm-up failed (non-fatal): {e}")

    def predict_sequence(sequence: str) -> tuple[str, str]:
        """
        Prediction function for the Gradio interface.

        Args:
            sequence: The antibody sequence to predict.

        Returns:
            A tuple containing the prediction string and the formatted probability.
        """
        try:
            # Validate with Pydantic (replaces old validate_input)
            request = PredictionRequest(sequence=sequence)

            # Log request (observability)
            logger.info(f"Processing: length={len(request.sequence)}")

            # Predict (returns PydanticResult)
            result = predictor.predict_single(request)

            # Format probability
            prob_percent = f"{result.probability:.1%}"

            return result.prediction, prob_percent

        except ValidationError as e:
            # Extract first error message for user-friendly display
            error_msg = e.errors()[0]["msg"]
            raise gr.Error(error_msg) from e
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during inference")
            raise gr.Error(
                "Server overloaded (GPU OOM). Please try again in a moment."
            ) from e
        except Exception as e:
            logger.exception("Unexpected prediction failure")
            raise gr.Error(f"Prediction failed: {str(e)}") from e

    # Example sequences (Diverse set)
    examples = [
        [
            "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS"
        ],  # Standard VH
        [
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
        ],  # Standard VL
        [
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARSWGQGTLVTVSS"
        ],  # Short VH (Herceptin-like)
    ]

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict_sequence,
        inputs=gr.TextArea(
            lines=7,
            max_lines=20,
            max_length=2000,
            label="Antibody Sequence (VH or VL)",
            placeholder="Paste amino acid sequence here (e.g., QVQL...)",
            info="Supported characters: Standard amino acids (ACDEFGHIKLMNPQRSTVWY).",
            show_copy_button=True,
        ),
        outputs=[
            gr.Textbox(label="Prediction", show_copy_button=True),
            gr.Textbox(label="Probability of Non-Specificity", show_copy_button=True),
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
        submit_btn="Predict Non-Specificity",
    )

    # Enable queueing for concurrency management
    """
    Queue Configuration:
    - concurrency_limit: Based on available VRAM (approx 3GB per ESM-1v inference).
    - max_size: Prevents unbounded queue growth under load.
    """
    iface.queue(
        default_concurrency_limit=cfg.gradio.queue.concurrency_limit,
        max_size=cfg.gradio.queue.max_size,
    )

    # Launch the app with hardened settings
    iface.launch(
        server_name=cfg.gradio.server_name,
        server_port=cfg.gradio.server_port,
        share=cfg.gradio.share,
        show_api=False,
    )


@hydra.main(config_path="../conf", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the Gradio app."""
    launch_gradio_app(cfg)


if __name__ == "__main__":
    main()
