"""
Integration Tests for Dataset Pipeline

Tests the full workflow: Dataset → Embedding → Training
Focus: Component interactions, not isolated behaviors

Testing philosophy:
- Test realistic workflows (load → embed → train → predict)
- Mock only heavyweight I/O (ESM model downloads)
- Use real pandas/sklearn operations
- Test cross-dataset compatibility (train on A, test on B)

Date: 2025-11-07
Phase: 3 (Integration Tests)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.harvey import HarveyDataset
from antibody_training_esm.datasets.jain import JainDataset
from antibody_training_esm.datasets.shehata import ShehataDataset

# ==================== Fixtures ====================


@pytest.fixture
def test_params():
    """Standard classifier params for integration tests"""
    return {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        "max_iter": 100,  # Low for fast tests
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "batch_size": 8,  # Small batches for fast tests
    }


@pytest.fixture
def mock_dataset_paths():
    """Paths to mock dataset CSV files"""
    fixtures_dir = Path(__file__).parent.parent / "fixtures/mock_datasets"
    return {
        "boughter": fixtures_dir / "boughter_sample.csv",
        "jain": fixtures_dir / "jain_sample.csv",
        "shehata": fixtures_dir / "shehata_sample.csv",
        "harvey_high": fixtures_dir / "harvey_high_sample.csv",
        "harvey_low": fixtures_dir / "harvey_low_sample.csv",
    }


# ==================== Cross-Dataset Pipeline Tests ====================


@pytest.mark.integration
def test_boughter_to_jain_pipeline(
    mock_transformers_model, test_params, mock_dataset_paths
):
    """Verify Boughter training set can predict on Jain test set (VH sequences)"""
    # Arrange: Load Boughter training data from mock
    boughter = BoughterDataset()
    df_train = boughter.load_data(
        processed_csv=str(mock_dataset_paths["boughter"]), include_mild=False
    )

    # Arrange: Load Jain test data from mock
    jain = JainDataset()
    df_test = jain.load_data(
        full_csv_path=str(mock_dataset_paths["jain"]), stage="full"
    )

    # Arrange: Extract embeddings from VH sequences
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=test_params["batch_size"],
    )

    # Use small subset for fast test
    X_train = extractor.extract_batch_embeddings(
        df_train["VH_sequence"].head(20).tolist()
    )
    y_train = df_train["label"].head(20).values

    X_test = extractor.extract_batch_embeddings(
        df_test["VH_sequence"].head(10).tolist()
    )

    # Act: Train classifier
    classifier = BinaryClassifier(params=test_params)
    classifier.fit(X_train, y_train)

    # Act: Predict on Jain
    predictions = classifier.predict(X_test)

    # Assert: Pipeline executes successfully
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)
    assert classifier.is_fitted


@pytest.mark.integration
def test_jain_to_shehata_pipeline(
    mock_transformers_model, test_params, mock_dataset_paths, tmp_path
):
    """Verify Jain training can predict on Shehata test set (different assay types)"""
    # Arrange: Load Jain training data from mock
    jain = JainDataset()
    df_train = jain.load_data(
        full_csv_path=str(mock_dataset_paths["jain"]), stage="full"
    )

    # Arrange: Load Shehata test data (prepare Excel from fixture)
    shehata = ShehataDataset()

    # Convert CSV to Excel for Shehata loader
    df_shehata = pd.read_csv(mock_dataset_paths["shehata"])
    shehata_excel = tmp_path / "shehata_test.xlsx"
    df_shehata.to_excel(shehata_excel, index=False)

    df_test = shehata.load_data(excel_path=str(shehata_excel))

    # Arrange: Extract embeddings
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=test_params["batch_size"],
    )

    X_train = extractor.extract_batch_embeddings(
        df_train["VH_sequence"].head(15).tolist()
    )
    y_train = df_train["label"].head(15).values

    X_test = extractor.extract_batch_embeddings(df_test["VH_sequence"].head(5).tolist())

    # Act: Train and predict with PSR threshold
    classifier = BinaryClassifier(params=test_params)
    classifier.fit(X_train, y_train)

    # Use PSR threshold for Shehata (different assay type)
    predictions = classifier.predict(X_test, assay_type="PSR")

    # Assert: Cross-assay prediction works
    assert len(predictions) == 5
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.integration
def test_harvey_nanobody_pipeline(
    mock_transformers_model, test_params, mock_dataset_paths
):
    """Verify Harvey nanobody dataset works in full pipeline (VHH only, no VL)"""
    # Arrange: Load Harvey nanobody data from mock
    harvey = HarveyDataset()
    df = harvey.load_data(
        high_csv_path=str(mock_dataset_paths["harvey_high"]),
        low_csv_path=str(mock_dataset_paths["harvey_low"]),
    )

    # Arrange: Extract embeddings (VH only - nanobodies have no VL)
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=test_params["batch_size"],
    )

    X = extractor.extract_batch_embeddings(df["VH_sequence"].tolist())
    y = df["label"].values

    # Act: Train classifier on nanobodies
    classifier = BinaryClassifier(params=test_params)
    classifier.fit(X[:8], y[:8])  # Train on 8 samples

    # Act: Predict on remaining
    predictions = classifier.predict(X[8:])

    # Assert: Nanobody pipeline works
    assert len(predictions) == 4  # 12 total - 8 train = 4 test
    assert all(pred in [0, 1] for pred in predictions)


# ==================== Fragment Creation Pipeline Tests ====================


@pytest.mark.integration
def test_boughter_fragment_csv_creation_pipeline(tmp_path):
    """Verify complete Boughter fragment CSV creation workflow"""
    # Arrange: Load annotated Boughter data
    annotated_csv = (
        Path(__file__).parent.parent / "fixtures/mock_datasets/boughter_annotated.csv"
    )
    df = pd.read_csv(annotated_csv)

    # Arrange: Create BoughterDataset with output directory
    dataset = BoughterDataset()
    dataset.output_dir = Path(tmp_path)

    # Act: Create fragment CSVs
    dataset.create_fragment_csvs(df, suffix="")

    # Assert: Verify fragment CSV files were created
    fragment_types = dataset.get_fragment_types()
    for ftype in fragment_types:
        fragment_csv = tmp_path / f"{ftype}_boughter.csv"
        assert fragment_csv.exists(), f"Fragment CSV for {ftype} not created"

        # Assert: Verify CSV structure (skip comment line with comment='#')
        df_fragment = pd.read_csv(fragment_csv, comment="#")
        assert "id" in df_fragment.columns
        assert "sequence" in df_fragment.columns
        assert "label" in df_fragment.columns
        assert "source" in df_fragment.columns
        assert len(df_fragment) > 0, f"Fragment CSV for {ftype} is empty"


@pytest.mark.integration
def test_jain_fragment_pipeline_with_suffix(tmp_path):
    """Verify Jain fragment creation with custom suffix"""
    # Arrange: Load annotated Jain data
    annotated_csv = (
        Path(__file__).parent.parent / "fixtures/mock_datasets/jain_annotated.csv"
    )
    df = pd.read_csv(annotated_csv)

    # Arrange: Create JainDataset with output directory
    dataset = JainDataset()
    dataset.output_dir = Path(tmp_path)

    # Act: Create fragment CSVs with custom suffix
    custom_suffix = "_full"
    dataset.create_fragment_csvs(df, suffix=custom_suffix)

    # Assert: Verify fragment CSV files with suffix were created
    fragment_types = dataset.get_fragment_types()
    for ftype in fragment_types:
        fragment_csv = tmp_path / f"{ftype}_jain{custom_suffix}.csv"
        assert fragment_csv.exists(), (
            f"Fragment CSV for {ftype}{custom_suffix} not created"
        )

        # Assert: Verify CSV structure (skip comment line with comment='#')
        df_fragment = pd.read_csv(fragment_csv, comment="#")
        assert "id" in df_fragment.columns
        assert "sequence" in df_fragment.columns
        assert "label" in df_fragment.columns
        assert "source" in df_fragment.columns
        assert len(df_fragment) > 0, f"Fragment CSV for {ftype}{custom_suffix} is empty"

    # Assert: Verify specific fragment content
    vh_only_csv = tmp_path / f"VH_only_jain{custom_suffix}.csv"
    df_vh_only = pd.read_csv(vh_only_csv, comment="#")
    assert len(df_vh_only) == len(df), (
        "VH_only fragment should have one row per input sequence"
    )


@pytest.mark.integration
def test_harvey_nanobody_fragments_pipeline(mock_dataset_paths, tmp_path):
    """Verify Harvey creates 6 nanobody fragments (not 16 full antibody fragments)"""
    # Arrange: Load Harvey nanobody data from mock
    harvey = HarveyDataset(output_dir=tmp_path)
    df = harvey.load_data(
        high_csv_path=str(mock_dataset_paths["harvey_high"]),
        low_csv_path=str(mock_dataset_paths["harvey_low"]),
    )

    # Act: Create nanobody fragments
    harvey.create_fragment_csvs(df, suffix="")

    # Assert: Only 6 nanobody fragments created
    nanobody_fragments = harvey.get_fragment_types()
    assert len(nanobody_fragments) == 6

    # Verify VHH_only fragment exists (not VH_only)
    vhh_file = tmp_path / "VHH_only_harvey.csv"
    assert vhh_file.exists()

    # Verify no light chain fragments
    vl_file = tmp_path / "VL_only_harvey.csv"
    assert not vl_file.exists(), "Light chain fragments should not exist for nanobodies"


# ==================== Multi-Stage Pipeline Tests ====================


@pytest.mark.integration
def test_jain_multi_stage_pipeline(
    mock_transformers_model, test_params, mock_dataset_paths
):
    """Verify training on different Jain stages produces different results

    NOTE: Placeholder test - Currently simulates parity stage by slicing full dataset.
    Does NOT exercise real Jain filtering logic (ELISA flags, reclassification).
    TODO: Create distinct mock CSVs for full/parity stages to properly test stage filtering.
    See TEST_SUITE_REVIEW_CHECKLIST.md Section 8 for backlog item.
    """
    # Arrange: Load mock Jain data (only full stage available in mocks)
    jain = JainDataset()
    df_full = jain.load_data(
        full_csv_path=str(mock_dataset_paths["jain"]), stage="full"
    )

    # PLACEHOLDER: Simulate parity stage by slicing (doesn't test real filtering)
    df_parity = df_full.head(len(df_full) // 2)  # Use first half as "parity"

    # Assert: Simulated stages have different sizes
    assert len(df_full) > len(df_parity)

    # Arrange: Extract embeddings for both
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=test_params["batch_size"],
    )

    X_full = extractor.extract_batch_embeddings(
        df_full["VH_sequence"].head(10).tolist()
    )
    y_full = df_full["label"].head(10).values

    X_parity = extractor.extract_batch_embeddings(
        df_parity["VH_sequence"].head(10).tolist()
    )
    y_parity = df_parity["label"].head(10).values

    # Act: Train separate classifiers
    classifier_full = BinaryClassifier(params=test_params)
    classifier_full.fit(X_full, y_full)

    classifier_parity = BinaryClassifier(params=test_params)
    classifier_parity.fit(X_parity, y_parity)

    # Assert: Both classifiers are fitted
    assert classifier_full.is_fitted
    assert classifier_parity.is_fitted


@pytest.mark.integration
def test_boughter_mild_flag_filtering_pipeline(
    mock_transformers_model, test_params, mock_dataset_paths
):
    """Verify include_mild parameter affects training data distribution"""
    # Arrange: Load Boughter with and without mild flags from mock
    boughter = BoughterDataset()
    df_no_mild = boughter.load_data(
        processed_csv=str(mock_dataset_paths["boughter"]), include_mild=False
    )
    df_with_mild = boughter.load_data(
        processed_csv=str(mock_dataset_paths["boughter"]), include_mild=True
    )

    # Assert: Different dataset sizes
    assert len(df_no_mild) < len(df_with_mild)

    # Arrange: Extract embeddings
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=test_params["batch_size"],
    )

    X_no_mild = extractor.extract_batch_embeddings(
        df_no_mild["VH_sequence"].head(10).tolist()
    )
    y_no_mild = df_no_mild["label"].head(10).values

    # Act: Train classifier on filtered data
    classifier = BinaryClassifier(params=test_params)
    classifier.fit(X_no_mild, y_no_mild)

    # Assert: Classifier trained successfully
    assert classifier.is_fitted


# ==================== End-to-End Embedding → Prediction Tests ====================


@pytest.mark.integration
def test_full_pipeline_from_sequences_to_predictions(
    mock_transformers_model, test_params
):
    """Verify complete pipeline: raw sequences → embeddings → training → predictions"""
    # Arrange: Raw sequences (simulate user input)
    sequences = [
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH",
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMH",
        "QVQLQQWGAGLLKPSETLSLTCAVYGGSFSGYYWSW",
    ]
    labels = np.array([0, 1, 0])

    # Act: Step 1 - Extract embeddings
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=test_params["batch_size"],
    )
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Act: Step 2 - Train classifier
    classifier = BinaryClassifier(params=test_params)
    classifier.fit(embeddings, labels)

    # Act: Step 3 - Predict on new sequence
    new_sequence = ["QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH"]
    new_embedding = extractor.extract_batch_embeddings(new_sequence)
    prediction = classifier.predict(new_embedding)

    # Assert: Full pipeline works
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]


@pytest.mark.integration
def test_pipeline_handles_batch_processing(mock_transformers_model, test_params):
    """Verify pipeline handles large batches efficiently"""
    # Arrange: Create 50 sequences (test batching logic)
    base_seq = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH"
    sequences = [base_seq] * 50
    labels = np.array([0, 1] * 25)

    # Act: Extract embeddings in batches
    extractor = ESMEmbeddingExtractor(
        model_name=test_params["model_name"],
        device=test_params["device"],
        batch_size=8,  # Small batch to test batching
    )
    embeddings = extractor.extract_batch_embeddings(sequences)

    # Act: Train on batched embeddings
    classifier = BinaryClassifier(params=test_params)
    classifier.fit(embeddings, labels)

    # Act: Predict in batch
    predictions = classifier.predict(embeddings)

    # Assert: Batch processing works
    assert embeddings.shape == (50, 1280)
    assert len(predictions) == 50
    assert all(pred in [0, 1] for pred in predictions)
