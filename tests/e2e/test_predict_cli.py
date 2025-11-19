import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def isolated_predict_test_env(tmp_path: Path) -> dict[str, Any]:
    """
    Creates a self-contained environment for the prediction CLI test.

    Note: This test will attempt to load ESM models from HuggingFace cache.
    On a fresh system without cached models, this may download ~2.5GB.
    """
    test_dir = tmp_path / "predict_test"
    test_dir.mkdir()
    input_file = test_dir / "input.csv"
    output_file = test_dir / "output.csv"
    classifier_path = test_dir / "dummy_classifier.pkl"
    conf_dir = test_dir / "conf"
    conf_dir.mkdir()

    # 1. Create test input file
    pd.DataFrame(
        {
            "sequence": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS",
                "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS",
            ]
        }
    ).to_csv(input_file, index=False)

    # 2. Create dummy classifier model
    dummy_classifier = LogisticRegression()
    dummy_classifier.fit(np.random.rand(2, 1280), [0, 1])
    joblib.dump(dummy_classifier, classifier_path)

    # 3. Create isolated Hydra config files
    # Main predict config
    with open(conf_dir / "predict.yaml", "w") as f:
        OmegaConf.save(
            config=OmegaConf.create(
                {
                    "defaults": [
                        "_self_",
                        {"override /model": "esm1v"},
                        {"override /classifier": "logreg"},
                    ],
                    "output_file": str(output_file),  # Default output
                }
            ),
            f=f,
        )
    # Model config
    (conf_dir / "model").mkdir()
    with open(conf_dir / "model" / "esm1v.yaml", "w") as f:
        OmegaConf.save(
            config=OmegaConf.create({"name": "facebook/esm1v_t33_650M_UR90S_1"}), f=f
        )
    # Classifier config
    (conf_dir / "classifier").mkdir()
    with open(conf_dir / "classifier" / "logreg.yaml", "w") as f:
        OmegaConf.save(config=OmegaConf.create({"path": None}), f=f)

    return {
        "input_file": input_file,
        "output_file": output_file,
        "conf_dir": conf_dir,
        "classifier_path": classifier_path,
    }


@pytest.mark.slow
@pytest.mark.e2e
def test_predict_cli_end_to_end(isolated_predict_test_env: dict[str, Any]) -> None:
    """
    Tests the predict CLI end-to-end in an isolated environment.

    Note: This test runs the actual CLI via subprocess and will load
    ESM models from HuggingFace cache. Marked as @slow and @e2e.
    """
    env = isolated_predict_test_env
    input_file = env["input_file"]
    output_file = env["output_file"]
    conf_dir = env["conf_dir"]
    classifier_path = env["classifier_path"]

    # Command to run using the predict entrypoint, pointing to the isolated config
    cmd = [
        "uv",
        "run",
        "antibody-predict",
        f"--config-dir={conf_dir}",
        "--config-name=predict",
        f"input_file={input_file}",
        f"output_file={output_file}",
        f"classifier.path={classifier_path}",
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Assertions
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
    assert output_file.exists(), "Output file was not created."

    # Verify the output file
    output_df = pd.read_csv(output_file)
    assert "prediction" in output_df.columns
    assert "probability" in output_df.columns
    assert len(output_df) == 2, (
        "Output file does not contain the expected number of rows."
    )
