# Gradio Integration Plan

This document outlines the plan for integrating a Gradio-based web UI into the antibody non-specificity prediction pipeline.

## 1. Goal

The primary goal is to provide a user-friendly web interface for the existing prediction pipeline, making it accessible to users without a programming background. This aligns with the "To-Be Implemented" feature mentioned in the `README.md`.

## 2. Architecture

The Gradio app will be a lightweight wrapper around the existing `Predictor` class in `src/antibody_training_esm/core/prediction.py`. This ensures that we are reusing the existing, well-tested code and not reinventing the wheel.

The app will be launched via a new CLI entry point, `antibody-app`, which will be defined in `pyproject.toml`. This new entry point will be implemented in a new file, `src/antibody_training_esm/cli/app.py`.

The Gradio app will be configured using the existing Hydra configuration system. This allows us to manage the model path and other parameters in a consistent and flexible way, without hardcoding any values.

## 3. Implementation Details

### 3.1. Dependencies

- `gradio`: This will be added as a project dependency in `pyproject.toml`.

### 3.2. CLI Entry Point

- A new CLI entry point, `antibody-app`, will be added to `pyproject.toml`.
- This will point to a `main` function in `src/antibody_training_esm/cli/app.py`.

### 3.3. Gradio App (`app.py`)

- The `app.py` file will contain the code for the Gradio interface.
- It will use the `Predictor` class to make predictions.
- The interface will take a single antibody sequence as input (a textbox).
- The output will be the prediction ("specific" or "non-specific") and the probability of non-specificity.

### 3.4. Configuration

- The Gradio app will be configured using Hydra.
- The `classifier.path` will be a required command-line argument, consistent with the `antibody-predict` CLI.

### 3.5. Testing

- A new test file will be created to test the Gradio app.
- The test will ensure that the Gradio app can be launched and that it correctly predicts the specificity of a sample antibody sequence.
- The test will use a mock `gradio.Interface` to avoid actually launching a web server during testing.

## 4. Development Workflow

The development will follow the existing TDD workflow:

1.  **Add `gradio` dependency.**
2.  **Create the CLI entry point.**
3.  **Write a failing test for the Gradio app.**
4.  **Implement the Gradio app to make the test pass.**
5.  **Refactor and clean up the code.**
6.  **Run `make all` to ensure all quality checks pass.**
7.  **Submit the change.**
