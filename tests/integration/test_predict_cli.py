
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
from omegaconf import OmegaConf


@pytest.fixture(scope="module")
def setup_test_data():
    """Sets up necessary directories and files for testing."""
    test_dir = Path("tests/integration/test_data").resolve()
    test_dir.mkdir(exist_ok=True, parents=True)
    input_file = test_dir / "input.csv"
    pd.DataFrame(
        {
            "sequence": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS",
                "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS",
            ]
        }
    ).to_csv(input_file, index=False)

    # Create dummy config and model files
    conf_dir = (Path("src/antibody_training_esm") / "conf").resolve()
    classifier_path = test_dir / "dummy_classifier.pkl"

    # Create a dummy classifier file
    dummy_classifier = LogisticRegression()
    dummy_classifier.fit(np.random.rand(2, 1280), [0, 1])
    joblib.dump(dummy_classifier, classifier_path)

    # Create a temporary override for the logreg classifier config
    original_logreg_path = conf_dir / "classifier" / "logreg.yaml"
    original_logreg_content = original_logreg_path.read_text()
    with open(original_logreg_path, "w") as f:
        OmegaConf.save(config=OmegaConf.create({"path": str(classifier_path)}), f=f)

    yield test_dir

    # Teardown: Restore the original logreg config
    original_logreg_path.write_text(original_logreg_content)


def test_predict_cli_end_to_end(setup_test_data):
    """Tests the predict CLI end-to-end."""
    test_dir = setup_test_data
    input_file = test_dir / "input.csv"
    output_file = test_dir / "output.csv"

    # Command to run using the predict entrypoint
    cmd = [
        "uv",
        "run",
        "antibody-predict",
        f"input_file={input_file}",
        f"output_file={output_file}",
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Assertions
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
    assert output_file.exists()

    # Verify the output file
    output_df = pd.read_csv(output_file)
    assert "prediction" in output_df.columns
    assert "probability" in output_df.columns
    assert len(output_df) == 2
