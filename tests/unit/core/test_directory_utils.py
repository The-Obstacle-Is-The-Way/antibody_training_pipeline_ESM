"""
Tests for directory_utils module

Validates hierarchical directory path generation for models and test results.
"""

from pathlib import Path

import pytest

from antibody_training_esm.core.directory_utils import (
    extract_classifier_shortname,
    extract_model_shortname,
    get_hierarchical_model_dir,
    get_hierarchical_test_results_dir,
)


@pytest.mark.unit
def test_extract_model_shortname_esm1v() -> None:
    """Verify ESM-1v model name extraction"""
    assert extract_model_shortname("facebook/esm1v_t33_650M_UR90S_1") == "esm1v"


@pytest.mark.unit
def test_extract_model_shortname_esm2_650m() -> None:
    """Verify ESM2-650M model name extraction"""
    assert extract_model_shortname("facebook/esm2_t33_650M_UR50D") == "esm2_650m"


@pytest.mark.unit
def test_extract_model_shortname_esm2_3b() -> None:
    """Verify ESM2-3B model name extraction"""
    assert extract_model_shortname("facebook/esm2_t36_3B_UR50D") == "esm2_3b"


@pytest.mark.unit
def test_extract_model_shortname_antiberta() -> None:
    """Verify AntiBERTa model name extraction"""
    assert extract_model_shortname("alchemab/antiberta2") == "antiberta"


@pytest.mark.unit
def test_extract_model_shortname_protbert() -> None:
    """Verify ProtBERT model name extraction"""
    assert extract_model_shortname("Rostlab/prot_bert") == "protbert"


@pytest.mark.unit
def test_extract_model_shortname_ablang() -> None:
    """Verify AbLang model name extraction"""
    assert extract_model_shortname("qilowoq/AbLang") == "ablang"


@pytest.mark.unit
def test_extract_model_shortname_fallback() -> None:
    """Verify fallback for unknown model names"""
    result = extract_model_shortname("someorg/custom_model_v2")
    assert result == "custom_model_v2"


@pytest.mark.unit
def test_extract_classifier_shortname_logreg() -> None:
    """Verify logistic regression classifier shortname"""
    config = {"type": "logistic_regression", "C": 1.0}
    assert extract_classifier_shortname(config) == "logreg"


@pytest.mark.unit
def test_extract_classifier_shortname_xgboost() -> None:
    """Verify XGBoost classifier shortname"""
    config = {"type": "xgboost", "max_depth": 6}
    assert extract_classifier_shortname(config) == "xgboost"


@pytest.mark.unit
def test_extract_classifier_shortname_mlp() -> None:
    """Verify MLP classifier shortname"""
    config = {"type": "mlp", "hidden_layers": [128, 64]}
    assert extract_classifier_shortname(config) == "mlp"


@pytest.mark.unit
def test_extract_classifier_shortname_svm() -> None:
    """Verify SVM classifier shortname"""
    config = {"type": "svm", "kernel": "rbf"}
    assert extract_classifier_shortname(config) == "svm"


@pytest.mark.unit
def test_extract_classifier_shortname_random_forest() -> None:
    """Verify Random Forest classifier shortname"""
    config = {"type": "random_forest", "n_estimators": 100}
    assert extract_classifier_shortname(config) == "rf"


@pytest.mark.unit
def test_extract_classifier_shortname_unknown() -> None:
    """Verify fallback for unknown classifier types"""
    config = {"type": "custom_classifier"}
    assert extract_classifier_shortname(config) == "custom_classifier"


@pytest.mark.unit
def test_extract_classifier_shortname_missing_type() -> None:
    """Verify handling of missing classifier type"""
    config: dict[str, str] = {}
    assert extract_classifier_shortname(config) == "unknown"


@pytest.mark.unit
def test_get_hierarchical_model_dir_esm1v_logreg() -> None:
    """Verify hierarchical model directory for ESM-1v + LogReg"""
    path = get_hierarchical_model_dir(
        "./models",
        "facebook/esm1v_t33_650M_UR90S_1",
        {"type": "logistic_regression"},
    )
    assert path == Path("models/esm1v/logreg")


@pytest.mark.unit
def test_get_hierarchical_model_dir_esm2_xgboost() -> None:
    """Verify hierarchical model directory for ESM2-650M + XGBoost"""
    path = get_hierarchical_model_dir(
        "./models",
        "facebook/esm2_t33_650M_UR50D",
        {"type": "xgboost"},
    )
    assert path == Path("models/esm2_650m/xgboost")


@pytest.mark.unit
def test_get_hierarchical_model_dir_antiberta_mlp() -> None:
    """Verify hierarchical model directory for AntiBERTa + MLP"""
    path = get_hierarchical_model_dir(
        "./models",
        "alchemab/antiberta2",
        {"type": "mlp"},
    )
    assert path == Path("models/antiberta/mlp")


@pytest.mark.unit
def test_get_hierarchical_model_dir_absolute_path() -> None:
    """Verify hierarchical model directory with absolute base path"""
    path = get_hierarchical_model_dir(
        "/opt/models",
        "facebook/esm1v_t33_650M_UR90S_1",
        {"type": "logistic_regression"},
    )
    assert path == Path("/opt/models/esm1v/logreg")


@pytest.mark.unit
def test_get_hierarchical_test_results_dir_esm1v_logreg_jain() -> None:
    """Verify hierarchical test results directory for ESM-1v + LogReg + Jain"""
    path = get_hierarchical_test_results_dir(
        "./test_results",
        "facebook/esm1v_t33_650M_UR90S_1",
        {"type": "logistic_regression"},
        "jain",
    )
    assert path == Path("test_results/esm1v/logreg/jain")


@pytest.mark.unit
def test_get_hierarchical_test_results_dir_esm2_xgboost_harvey() -> None:
    """Verify hierarchical test results directory for ESM2 + XGBoost + Harvey"""
    path = get_hierarchical_test_results_dir(
        "./test_results",
        "facebook/esm2_t33_650M_UR50D",
        {"type": "xgboost"},
        "harvey",
    )
    assert path == Path("test_results/esm2_650m/xgboost/harvey")


@pytest.mark.unit
def test_get_hierarchical_test_results_dir_antiberta_mlp_shehata() -> None:
    """Verify hierarchical test results directory for AntiBERTa + MLP + Shehata"""
    path = get_hierarchical_test_results_dir(
        "./test_results",
        "alchemab/antiberta2",
        {"type": "mlp"},
        "shehata",
    )
    assert path == Path("test_results/antiberta/mlp/shehata")


@pytest.mark.unit
def test_get_hierarchical_test_results_dir_absolute_path() -> None:
    """Verify hierarchical test results directory with absolute base path"""
    path = get_hierarchical_test_results_dir(
        "/opt/test_results",
        "facebook/esm1v_t33_650M_UR90S_1",
        {"type": "logistic_regression"},
        "jain",
    )
    assert path == Path("/opt/test_results/esm1v/logreg/jain")
