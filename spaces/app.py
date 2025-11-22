"""
Hugging Face Spaces Gradio App for Antibody Non-Specificity Prediction

Simplified deployment version (no Hydra, no complex dependencies).
Works on HF Spaces free CPU tier.

Local app (src/antibody_training_esm/cli/app.py) remains unchanged.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to Python path for local imports (HF Spaces doesn't install package)
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import torch
from pydantic import ValidationError

from antibody_training_esm.core.prediction import Predictor
from antibody_training_esm.models.prediction import PredictionRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HF Spaces environment detection
IS_HF_SPACE = os.getenv("SPACE_ID") is not None

# Model path (either local or downloaded from HF Hub)
MODEL_PATH = os.getenv(
    "MODEL_PATH", "experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
)

# ESM model name
MODEL_NAME = "facebook/esm1v_t33_650M_UR90S_1"

# Force CPU for HF Spaces free tier
DEVICE = "cpu"

# Load model globally (HF Spaces best practice)
logger.info(f"Loading model from {MODEL_PATH}...")
predictor = Predictor(
    model_name=MODEL_NAME, classifier_path=MODEL_PATH, device=DEVICE, config_path=None
)

# Warm up model
try:
    logger.info("Warming up model...")
    predictor.predict_single("QVQL")
    logger.info("Model ready!")
except Exception as e:
    logger.warning(f"Warmup failed (non-fatal): {e}")


def predict_sequence(sequence: str) -> tuple[str, str]:
    """
    Prediction function for Gradio interface.

    Args:
        sequence: Antibody amino acid sequence

    Returns:
        Tuple of (prediction, probability)
    """
    try:
        # Validate with Pydantic
        request = PredictionRequest(sequence=sequence)

        # Log request
        logger.info(f"Processing sequence: length={len(request.sequence)}")

        # Predict
        result = predictor.predict_single(request)

        # Format probability
        prob_percent = f"{result.probability:.1%}"

        return result.prediction, prob_percent

    except ValidationError as e:
        # User-friendly error message
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


# Example sequences
examples = [
    [
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS"
    ],  # Standard VH
    [
        "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
    ],  # Standard VL
    [
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARSWGQGTLVTVSS"
    ],  # Short VH
]

# Create Gradio interface
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
    title="🧬 Antibody Non-Specificity Predictor",
    description=(
        "Predict antibody polyreactivity (non-specificity) from Variable Heavy (VH) "
        "or Variable Light (VL) sequences using ESM-1v protein language models.\n\n"
        "**Model:** ESM-1v (650M parameters) + Logistic Regression\n"
        "**Training:** Boughter dataset (914 antibodies, ELISA polyreactivity)\n"
        "**Citation:** Sakhnini et al. (2025) - Prediction of Antibody Non-Specificity using PLMs"
    ),
    article=(
        f"**Model:** {MODEL_NAME}\n"
        f"**Device:** {DEVICE}\n"
        f"**Environment:** {'Hugging Face Spaces' if IS_HF_SPACE else 'Local'}"
    ),
    examples=examples,
    cache_examples=False,  # Don't cache on HF Spaces (saves disk)
    flagging_mode="never",
    analytics_enabled=False,
    submit_btn="🔬 Predict Non-Specificity",
    clear_btn="🗑️ Clear",
)

# Enable queue for concurrency
iface.queue(default_concurrency_limit=2, max_size=10)

# Launch app
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",  # Required for HF Spaces
        server_port=7860,
        share=False,
        show_api=False,  # No public REST API
    )
