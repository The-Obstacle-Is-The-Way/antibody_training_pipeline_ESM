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

        # --- Generate HTML Card (inline styles survive HF Spaces iframe stripping) ---
        is_specific = result.prediction == "specific"

        base_style = (
            "padding:30px;border-radius:16px;text-align:center;"
            "margin-bottom:20px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);"
            "transition:all 0.3s ease;"
        )

        if is_specific:
            card_style = (
                base_style
                + "background-color:#ecfdf5;border:2px solid #10b981;color:#065f46;"
            )
            icon = "✅"
            title = "Specific (Safe)"
            msg = "Low risk of polyreactivity"
        else:
            card_style = (
                base_style
                + "background-color:#fef2f2;border:2px solid #ef4444;color:#991b1b;"
            )
            icon = "⚠️"
            title = "Non-Specific (Risk)"
            msg = "High risk of polyreactivity"

        html_card = f"""
        <div style="{card_style}">
            <span style="font-size:48px;display:block;margin-bottom:15px;">{icon}</span>
            <div style="font-size:28px;font-weight:800;letter-spacing:-0.025em;margin-bottom:5px;">
                {title}
            </div>
            <div style="font-size:16px;opacity:0.9;">{msg}</div>
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
# Force Light Theme to prevent "Dark Mode" components on White Background
# We explicitly set *_dark variables to match light variables to disable dark mode
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
).set(
    body_background_fill="#FFFFFF",
    body_background_fill_dark="#FFFFFF",
    body_text_color="#111827",
    body_text_color_dark="#111827",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    block_background_fill="#F9FAFB",
    block_background_fill_dark="#F9FAFB",
    # Label clarity improvements - lighter blue bg, crisp white text
    block_label_background_fill="#3B82F6",
    block_label_background_fill_dark="#3B82F6",
    block_label_text_color="#FFFFFF",
    block_label_text_color_dark="#FFFFFF",
    # Force white text on ALL blue/primary colored elements
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    # Align button background with labels (#3B82F6) and define standard hover (#2563EB)
    button_primary_background_fill="#3B82F6",
    button_primary_background_fill_dark="#3B82F6",
    button_primary_background_fill_hover="#2563EB",
    button_primary_background_fill_hover_dark="#2563EB",
    slider_color="#3B82F6",
    slider_color_dark="#3B82F6",
    block_title_text_color="#374151",
    block_title_text_color_dark="#374151",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    # Table-specific overrides to fix dark mode tables
    table_border_color="#E5E7EB",
    table_border_color_dark="#E5E7EB",
    table_even_background_fill="#F9FAFB",
    table_even_background_fill_dark="#F9FAFB",
    table_odd_background_fill="#FFFFFF",
    table_odd_background_fill_dark="#FFFFFF",
    table_row_focus="#DBEAFE",
    table_row_focus_dark="#DBEAFE",
)
with gr.Blocks(theme=theme, title="Antibody Predictor") as app:
    # Header (inline styles to survive HF Spaces stripping)
    gr.HTML(
        """
        <div style="text-align:center;margin-bottom:20px;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;">
            <div style="font-size:2.4rem;font-weight:700;color:#3b82f6;margin-bottom:8px;">
                🧬 Antibody Non-Specificity Predictor
            </div>
            <div style="font-size:1.1rem;color:#6b7280;">
                Assess polyreactivity risk using ESM-1v Protein Language Models
            </div>
        </div>
        """
    )

    # Main Content
    with gr.Row(equal_height=False):
        # Left Column: Inputs
        with gr.Column(scale=1):
            gr.HTML(
                '<div style="background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.875rem; display: block; margin-bottom: 4px;">Antibody Sequence (VH or VL)</div>'
            )
            sequence_input = gr.TextArea(
                placeholder="Paste amino acid sequence here (e.g., QVQL...)",
                lines=5,
                max_lines=15,
                show_copy_button=True,
                show_label=False,  # Disable built-in label
            )
            with gr.Accordion("⚙️ Advanced Settings", open=False), gr.Row():
                with gr.Column():
                    gr.HTML(
                        '<div style="background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.875rem; display: inline-block; margin-bottom: 4px;">Calibrated Assay</div>'
                    )
                    assay_input = gr.Dropdown(
                        choices=["ELISA", "PSR", "None"],
                        value="None",
                        show_label=False,
                        info="Use threshold calibrated for specific assay",
                    )
                with gr.Column():
                    gr.HTML(
                        '<div style="background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.875rem; display: inline-block; margin-bottom: 4px;">Decision Threshold</div>'
                    )
                    threshold_input = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        show_label=False,
                        info="Probability cutoff for non-specificity",
                    )

            submit_btn = gr.Button(
                "🔬 Predict Non-Specificity", variant="primary", size="lg"
            )

            # Examples
            gr.HTML(
                '<div style="background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.875rem; display: block; margin-bottom: 4px;">Load Example Data</div>'
            )
            gr.Examples(
                examples=examples,
                inputs=[sequence_input, threshold_input, assay_input],
            )

        # Right Column: Outputs
        with gr.Column(scale=1):
            # HTML Card
            result_html = gr.HTML(
                label="Prediction Status",
                value="""
                <div style="padding:30px;border-radius:16px;text-align:center;margin-bottom:20px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);background-color:#f3f4f6;border:2px dashed #d1d5db;color:#374151;">
                    <span style="font-size:48px;display:block;margin-bottom:15px;">⏳</span>
                    <div style="font-size:28px;font-weight:800;letter-spacing:-0.025em;margin-bottom:5px;">Ready to Predict</div>
                    <div style="font-size:16px;opacity:0.9;">Enter a sequence to begin analysis</div>
                </div>
                """,
            )

            # Confidence Bar
            gr.HTML(
                '<div style="background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.875rem; display: inline-block; margin-bottom: 4px;">Model Confidence</div>'
            )
            confidence_output = gr.Label(num_top_classes=2, show_label=False)

            # Detailed JSON
            with gr.Accordion("📋 Detailed JSON Output", open=False):
                gr.HTML(
                    '<div style="background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.875rem; display: inline-block; margin-bottom: 4px;">Raw Result</div>'
                )
                json_output = gr.JSON(show_label=False)

    # Footer
    gr.HTML(
        """
        <div style="text-align:center;margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;color:#6b7280;font-size:0.95rem;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;">
            Model: ESM-1v (650M) + Logistic Regression • Training: Boughter et al. (914 sequences)
            <br>
            <a style="color:#6b7280;text-decoration:none;margin:0 10px;" href="https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1" target="_blank">ESM-1v Model</a> •
            <a style="color:#6b7280;text-decoration:none;margin:0 10px;" href="#" target="_blank">Paper Citation (Sakhnini et al. 2025)</a>
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
