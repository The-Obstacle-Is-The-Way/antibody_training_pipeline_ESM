#!/usr/bin/env python3
"""
Mock ESM Models for Testing

Provides lightweight mock implementations of ESM-1v model and tokenizer
to avoid downloading 650MB model weights during testing.

Usage:
    @pytest.fixture
    def mock_transformers_model(monkeypatch):
        from tests.fixtures.mock_models import MockESMModel, MockTokenizer
        monkeypatch.setattr("transformers.AutoModel.from_pretrained", MockESMModel)
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", MockTokenizer)

Date: 2025-11-07
Philosophy: Mock I/O boundaries (model loading), test behavior everywhere else
"""

from typing import Any

import numpy as np
import torch


class MockESMModel:
    """
    Mock Hugging Face transformers ESM-1v model.

    Mimics the interface of AutoModel.from_pretrained() without requiring
    actual model download. Returns random embeddings of correct shape (1280-d).

    Usage:
        model = MockESMModel("facebook/esm1v_t33_650M_UR90S_1")
        outputs = model(input_ids, attention_mask, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # Shape: (batch, seq_len, 1280)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize mock model (accepts any args to match real API).

        Args:
            *args: Positional args (ignored)
            **kwargs: Keyword args (ignored)
        """
        self.device_type = "cpu"
        self.eval_mode = False

    def to(self, device: str):
        """Mock device assignment."""
        self.device_type = device
        return self

    def eval(self):
        """Mock eval mode (no gradient computation)."""
        self.eval_mode = True
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """
        Mock forward pass (returns random embeddings).

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            output_hidden_states: If True, return hidden states

        Returns:
            MockOutput with .hidden_states attribute
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Mock hidden states: (batch_size, seq_len, 1280)
        # ESM-1v returns 33 layers, we only need the last one
        hidden_states = torch.rand(batch_size, seq_len, 1280)

        # Mock output object (mimics transformers.modeling_outputs)
        class MockOutput:
            def __init__(self, hidden_states):
                # Return tuple of layers (we only populate the last layer)
                self.hidden_states = (None, hidden_states)

        return MockOutput(hidden_states)


class MockTokenizer:
    """
    Mock Hugging Face tokenizer for ESM-1v.

    Mimics the interface of AutoTokenizer.from_pretrained() without requiring
    actual tokenizer download. Returns mock input_ids and attention_mask.

    Usage:
        tokenizer = MockTokenizer("facebook/esm1v_t33_650M_UR90S_1")
        inputs = tokenizer(sequences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]  # Shape: (batch, seq_len)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize mock tokenizer (accepts any args to match real API).

        Args:
            *args: Positional args (ignored)
            **kwargs: Keyword args (ignored)
        """
        pass

    def __call__(
        self,
        sequences: str | list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ):
        """
        Mock tokenization (returns random token IDs).

        Args:
            sequences: Single sequence or list of sequences
            return_tensors: Return type ("pt" for PyTorch)
            padding: Whether to pad to max length
            truncation: Whether to truncate to max length
            max_length: Maximum sequence length
            **kwargs: Additional args (ignored)

        Returns:
            dict with "input_ids" and "attention_mask" tensors
        """
        # Convert single sequence to list
        if isinstance(sequences, str):
            sequences = [sequences]

        batch_size = len(sequences)

        # Calculate max sequence length (+2 for CLS and EOS tokens)
        if padding:
            max_len = max(len(s) for s in sequences) + 2
        else:
            max_len = len(sequences[0]) + 2

        if max_length is not None:
            max_len = min(max_len, max_length)

        # Mock input_ids: random token IDs in range [0, 33]
        # (ESM-1v has 33 tokens: 20 amino acids + special tokens)
        input_ids = torch.randint(0, 33, (batch_size, max_len))

        # Mock attention_mask: all 1s (all tokens are "real")
        attention_mask = torch.ones(batch_size, max_len, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MockClassifier:
    """
    Mock sklearn LogisticRegression classifier for testing.

    Provides deterministic predictions for reproducible tests.

    Usage:
        classifier = MockClassifier()
        classifier.fit(X, y)
        predictions = classifier.predict(X)
    """

    def __init__(self, **kwargs):
        """
        Initialize mock classifier.

        Args:
            **kwargs: Hyperparameters (stored but not used)
        """
        self.params = kwargs
        self.is_fitted = False
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Mock fit (just marks as fitted).

        Args:
            X: Training embeddings (n_samples, 1280)
            y: Training labels (n_samples,)
        """
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Mock predict (returns deterministic predictions).

        Args:
            X: Embeddings (n_samples, 1280)

        Returns:
            Binary predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")

        # Deterministic: even indices -> 0, odd indices -> 1
        n_samples = X.shape[0]
        return np.array([i % 2 for i in range(n_samples)])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Mock predict_proba (returns fixed probabilities).

        Args:
            X: Embeddings (n_samples, 1280)

        Returns:
            Class probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict_proba")

        n_samples = X.shape[0]
        # Return probabilities that match predict() output
        proba = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if i % 2 == 0:
                proba[i] = [0.7, 0.3]  # Class 0
            else:
                proba[i] = [0.3, 0.7]  # Class 1

        return proba

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Mock get_params (sklearn API)."""
        return dict(self.params)

    def set_params(self, **params):
        """Mock set_params (sklearn API)."""
        self.params.update(params)
        return self


def create_mock_embeddings(
    n_samples: int = 10, embedding_dim: int = 1280, seed: int = 42
) -> np.ndarray:
    """
    Create mock ESM-1v embeddings for testing.

    Args:
        n_samples: Number of samples
        embedding_dim: Embedding dimension (default: 1280 for ESM-1v)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of shape (n_samples, embedding_dim)
    """
    np.random.seed(seed)
    return np.random.rand(n_samples, embedding_dim).astype(np.float32)


def create_mock_labels(n_samples: int = 10, balanced: bool = True) -> np.ndarray:
    """
    Create mock binary labels for testing.

    Args:
        n_samples: Number of samples
        balanced: If True, creates 50/50 class distribution

    Returns:
        np.ndarray of shape (n_samples,) with values 0 or 1
    """
    if balanced:
        # Alternate 0, 1, 0, 1, ...
        return np.array([i % 2 for i in range(n_samples)])
    else:
        # All zeros (imbalanced)
        return np.zeros(n_samples, dtype=int)
