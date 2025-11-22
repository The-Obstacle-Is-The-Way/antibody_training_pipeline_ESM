# HF Spaces UI/UX Improvement Plan

## Context
First-time deployment of antibody non-specificity predictor to Hugging Face Spaces.
- **Model**: ESM-1v (650M) + Logistic Regression
- **Training**: Boughter dataset (914 antibodies, ELISA polyreactivity)
- **Current state**: Functional but visually basic (default Gradio theme, plain white, no visual hierarchy)

## Current Features ✅
1. Single sequence input (text area)
2. Prediction output (text)
3. Probability score (text)
4. Basic examples (collapsed)

## Missing / To Improve ❌
1. **Visual Design**: "Hella ugly", plain white, no branding, no hierarchy.
2. **Interactivity**: No real-time feedback.
3. **Advanced Options**: No ability to change assay type (ELISA/PSR) or threshold.
4. **Visualization**: No visual representation of probability (just a number).
5. **Batch Support**: No way to upload a CSV for batch prediction.

## Proposed Improvements

### Phase 1: "Gorgeous" Makeover (Immediate)
*   **Theme**: Switch to `gr.themes.Soft()` with custom colors (Blue/Indigo/Slate) to match the "scientific/medical" aesthetic.
*   **Layout**: Refactor from `gr.Interface` to `gr.Blocks` for custom layout control.
    *   **Header**: Add a proper title with emoji/icon, short description, and links to GitHub/Paper.
    *   **Input Section**: 
        *   Large text area with clear placeholder.
        *   "Paste Example" buttons (clearer than the default examples table).
    *   **Output Section**:
        *   **Status Card**: Large colored card (Green for Specific, Red for Non-Specific) with an icon.
        *   **Confidence Meter**: `gr.Label` or custom progress bar showing the probability.
    *   **Sidebar/Accordion**: Advanced settings (hidden by default).

### Phase 2: Functional Enhancements (High Value)
*   **Advanced Settings**:
    *   **Assay Type Selector**: Dropdown for `ELISA` vs `PSR` (maps to `assay_type` in `Predictor`).
    *   **Threshold Slider**: Slider (0.0 to 1.0, default 0.5) to adjust sensitivity.
*   **Real-time Validation**:
    *   Show sequence length counter.
    *   Warn if invalid characters are detected before submission.

### Phase 3: Batch Processing (Optional but Recommended)
*   **CSV Upload**: Tab for "Batch Prediction".
    *   Upload CSV -> Process -> Download CSV with predictions.
    *   Use `Predictor.predict_dataframe` backend method.

## Implementation Details

### library
Use `gradio>=4.0.0` features.

### Custom CSS
```css
.gradio-container { font-family: 'Inter', sans-serif; }
.prediction-card-safe { background-color: #ecfdf5; border: 1px solid #10b981; color: #065f46; padding: 20px; border-radius: 10px; text-align: center; }
.prediction-card-danger { background-color: #fef2f2; border: 1px solid #ef4444; color: #991b1b; padding: 20px; border-radius: 10px; text-align: center; }
```

### Code Structure
Refactor `spaces/app.py` to use `gr.Blocks`:

```python
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧬 Antibody Non-Specificity Predictor")
    
    with gr.Row():
        with gr.Column():
            sequence_input = gr.TextArea(...)
            with gr.Accordion("Advanced Settings", open=False):
                threshold = gr.Slider(...)
                assay = gr.Dropdown(...)
            submit_btn = gr.Button("Predict", variant="primary")
        
        with gr.Column():
            result_html = gr.HTML(...) # Custom card
            confidence = gr.Label(...)
            
    submit_btn.click(...)
```

## Success Criteria
1. UI looks professional and polished ("Gorgeous").
2. Users can easily interpret results (Green vs Red).
3. Advanced users can access Assay Type/Thresholds.
4. Inference speed remains acceptable on CPU.