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
from typing import Any, cast

# Add src to Python path for local imports (HF Spaces doesn't install package)
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import torch
from pydantic import ValidationError

from antibody_training_esm.core.prediction import Predictor
from antibody_training_esm.models.prediction import AssayType, PredictionRequest

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
# Note: We initialize with config_path=None assuming pickle or named config for npz
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


def predict_sequence(
    sequence: str, threshold: float, assay_type: str | None
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """
    Prediction function for Gradio interface.

    Args:
        sequence: Antibody amino acid sequence
        threshold: Decision threshold
        assay_type: Optional assay type (ELISA/PSR)

    Returns:
        Tuple of (HTML Card, Label Dict, JSON Result)
    """
    try:
        # Handle "None" string from dropdown
        validated_assay: AssayType | None = None
        if assay_type and assay_type not in ("None", ""):
            # Gradio dropdown guarantees value is "ELISA" or "PSR"
            validated_assay = cast(AssayType, assay_type)

        # Validate with Pydantic
        request = PredictionRequest(
            sequence=sequence, threshold=threshold, assay_type=validated_assay
        )

        # Log request
        logger.info(f"Processing sequence: length={len(request.sequence)}")

        # Predict
        result = predictor.predict_single(request)

        # --- Generate HTML Card ---
        is_specific = result.prediction == "specific"

        if is_specific:
            color_class = "status-safe"
            icon = "✅"
            title = "Specific (Safe)"
            msg = "Low risk of polyreactivity"
        else:
            color_class = "status-danger"
            icon = "⚠️"
            title = "Non-Specific (Risk)"
            msg = "High risk of polyreactivity"

        html_card = f"""
        <div class="status-card {color_class}">
            <span class="status-icon">{icon}</span>
            <div class="status-text">{title}</div>
            <div class="status-subtext">{msg}</div>
        </div>
        """

        # --- Generate Label ---
        # Gradio Label expects dict {label: prob}
        # We return the probability of the predicted class
        label_dict = {
            "Non-Specificity Risk": result.probability,
            "Specificity": 1.0 - result.probability,
        }

        # --- Generate JSON ---
        json_result = result.model_dump(
            exclude={"sequence"}
        )  # Exclude sequence to save space

        return html_card, label_dict, json_result

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


# --- Custom CSS ---
css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.header-text {
    text-align: center;
    margin-bottom: 20px;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-subtitle {
    font-size: 1.1rem;
    color: #6b7280;
}
.status-card {
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.status-safe {
    background-color: #ecfdf5;
    border: 2px solid #10b981;
    color: #065f46;
}
.status-danger {
    background-color: #fef2f2;
    border: 2px solid #ef4444;
    color: #991b1b;
}
.status-icon {
    font-size: 48px;
    display: block;
    margin-bottom: 15px;
}
.status-text {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin-bottom: 5px;
}
.status-subtext {
    font-size: 16px;
    opacity: 0.9;
}
.footer-links {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
    color: #9ca3af;
    font-size: 0.9rem;
}
.footer-links a {
    color: #6b7280;
    text-decoration: none;
    margin: 0 10px;
}
.footer-links a:hover {
    color: #3b82f6;
    text-decoration: underline;
}
"""

# Inline CSS injection via HTML to survive HF Spaces iframe stripping
inline_style = f"<style>{css}</style>"

# --- Example Sequences ---
examples = [
    [
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMGGIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARSTYYGGDWYFNVWGQGTLVTVSS",
        0.5,
        "ELISA",
    ],
    [
        "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK",
        0.5,
        "PSR",
    ],
    [
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARSWGQGTLVTVSS",
        0.8,
        None,
    ],
]

# --- Gradio Blocks App ---
with gr.Blocks(theme=gr.themes.Soft(), title="Antibody Predictor") as app:
    # Inject CSS early; HF Spaces strips gr.Blocks(css=...) in iframes
    gr.HTML(inline_style)

    # Header
    with gr.Column(elem_classes="header-text"):
        gr.Markdown(
            """
            <div class="header-title">🧬 Antibody Non-Specificity Predictor</div>
            <div class="header-subtitle">
                Assess polyreactivity risk using ESM-1v Protein Language Models
            </div>
            """
        )

    # Main Content
    with gr.Row(equal_height=False):
        # Left Column: Inputs
        with gr.Column(scale=1):
            with gr.Group():
                sequence_input = gr.TextArea(
                    label="Antibody Sequence (VH or VL)",
                    placeholder="Paste amino acid sequence here (e.g., QVQL...)",
                    lines=5,
                    max_lines=15,
                    show_copy_button=True,
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False), gr.Row():
                    assay_input = gr.Dropdown(
                        choices=["ELISA", "PSR", "None"],
                        value="None",
                        label="Calibrated Assay",
                        info="Use threshold calibrated for specific assay",
                    )
                    threshold_input = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Decision Threshold",
                        info="Probability cutoff for non-specificity",
                    )

            submit_btn = gr.Button(
                "🔬 Predict Non-Specificity", variant="primary", size="lg"
            )

            # Examples
            gr.Examples(
                examples=examples,
                inputs=[sequence_input, threshold_input, assay_input],
                label="Load Example Data",
            )

        # Right Column: Outputs
        with gr.Column(scale=1):
            # HTML Card
            result_html = gr.HTML(
                label="Prediction Status",
                value="""
                <div class="status-card" style="background-color: #f3f4f6; border: 2px dashed #d1d5db; color: #6b7280;">
                    <span class="status-icon">⏳</span>
                    <div class="status-text">Ready to Predict</div>
                    <div class="status-subtext">Enter a sequence to begin analysis</div>
                </div>
                """,
            )

            # Confidence Bar
            confidence_output = gr.Label(
                label="Model Confidence", num_top_classes=2, show_label=True
            )

            # Detailed JSON
            with gr.Accordion("📋 Detailed JSON Output", open=False):
                json_output = gr.JSON(label="Raw Result")

    # Footer
    gr.Markdown(
        """
        <div class="footer-links">
            Model: ESM-1v (650M) + Logistic Regression • Training: Boughter et al. (914 sequences)
            <br>
            <a href="https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1" target="_blank">ESM-1v Model</a> •
            <a href="#" target="_blank">Paper Citation (Sakhnini et al. 2025)</a>
        </div>
        """
    )

    # Logic Binding
    submit_btn.click(
        fn=predict_sequence,
        inputs=[sequence_input, threshold_input, assay_input],
        outputs=[result_html, confidence_output, json_output],
    )

# Launch
if __name__ == "__main__":
    app.queue(default_concurrency_limit=2, max_size=10)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)
