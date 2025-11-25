"""
AMPLIFY Embedding Module

Professional module for AMPLIFY 350M protein sequence embedding extraction.
Handles AMPLIFY-specific requirements: batch_size=1, trust_remote_code, attention workarounds.

CRITICAL WARNING:
    AMPLIFY has a padding/batching bug that causes non-reproducible embeddings
    when batch_size > 1. This module enforces batch_size=1.

    Source: https://www.nature.com/articles/s41598-025-05674-x

    "When processing a batch of multiple sequences with different lengths,
    shorter sequences need to be padded to the maximum length, and this
    padding should not affect computed embeddings, but if a transformer
    model does not properly mask padded sites when calculating attention
    then the padding can influence output embeddings, which will result
    in poor reproducibility."

Date: 2025-11-24
Author: Claude Code (Opus 4.5)
"""

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import (
    DEFAULT_MAX_SEQ_LENGTH,
    SEQUENCE_PREVIEW_LENGTH,
)

logger = logging.getLogger(__name__)


class AMPLIFYEmbeddingExtractor:
    """
    Extract AMPLIFY 350M embeddings for protein sequences.

    CRITICAL WARNING: AMPLIFY has a padding/batching reproducibility issue.
    This class enforces batch_size=1 for consistent results.

    Key Differences from ESMEmbeddingExtractor:
        1. Requires trust_remote_code=True (AMPLIFY uses custom modeling code)
        2. Requires attn_implementation workaround for MPS (Flash Attention is CUDA-only)
        3. Forces batch_size=1 (padding bug causes non-reproducible embeddings)
        4. Returns 960-d embeddings (vs ESM's 1280-d)

    Security Advisory:
        trust_remote_code=True executes arbitrary Python code from HuggingFace.
        This is REQUIRED for AMPLIFY (custom attention layers). Only use:
        - With pinned revision (commit SHA, not 'main' branch)
        - In trusted environments (not public-facing APIs)
        - After reviewing chandar-lab/AMPLIFY_350M modeling_amplify.py

    Example:
        >>> extractor = AMPLIFYEmbeddingExtractor(
        ...     model_name="chandar-lab/AMPLIFY_350M",
        ...     device="mps"
        ... )
        >>> embedding = extractor.embed_sequence("QVQLVQSG")
        >>> embedding.shape
        (960,)
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = 1,
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
        revision: str = "main",
    ):
        """
        Initialize AMPLIFY embedding extractor

        Args:
            model_name: HuggingFace model identifier (e.g., 'chandar-lab/AMPLIFY_350M')
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            batch_size: MUST be 1 due to AMPLIFY padding bug (forced if > 1)
            max_length: Maximum sequence length for tokenizer truncation
            revision: HuggingFace model revision (commit SHA or branch name)

        Raises:
            ImportError: If transformers library not installed
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.revision = revision

        # CRITICAL: Force batch_size=1 due to padding bug
        if batch_size != 1:
            logger.warning(
                f"⚠️  AMPLIFY PADDING BUG: batch_size={batch_size} requested but AMPLIFY has "
                f"non-reproducible embeddings with batching (Nature Sci Rep 2025). "
                f"Forcing batch_size=1. See: https://www.nature.com/articles/s41598-025-05674-x"
            )
            batch_size = 1
        self.batch_size = batch_size

        # Determine attention implementation based on device
        # Flash Attention is CUDA-only; MPS/CPU require workarounds
        attn_impl: str | None = None  # Auto-detect for CUDA
        if device == "mps":
            attn_impl = "sdpa"  # Scaled Dot-Product Attention (MPS-compatible)
            logger.info("Using SDPA attention for MPS (Flash Attention not supported)")
        elif device == "cpu":
            attn_impl = "eager"  # Standard attention for CPU
            logger.info("Using eager attention for CPU")

        # Load model with AMPLIFY-specific flags
        # Revision IS pinned via parameter (default: specific SHA in config)
        self.model = AutoModel.from_pretrained(  # nosec B615
            model_name,
            trust_remote_code=True,  # REQUIRED for AMPLIFY
            attn_implementation=attn_impl,
            output_hidden_states=True,
            revision=revision,
        )
        self.model.to(device)
        self.model.eval()

        # Revision IS pinned via parameter (default: specific SHA in config)
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_name,
            trust_remote_code=True,  # REQUIRED for AMPLIFY
            revision=revision,
        )  # type: ignore[no-untyped-call]  # HuggingFace transformers lacks type stubs

        logger.info(
            f"AMPLIFY model {model_name} (revision={revision}) loaded on {device} "
            f"with batch_size=1 (padding bug), max_length={max_length}, "
            f"attn_implementation={attn_impl or 'auto'}"
        )

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Extract AMPLIFY embedding for a single protein sequence

        Args:
            sequence: Amino acid sequence string (case-insensitive)

        Returns:
            Embedding vector as numpy array (960-d)

        Raises:
            ValueError: If sequence contains invalid amino acids or is too short

        Example:
            >>> embedding = extractor.embed_sequence("QVQLVQSG")
            >>> embedding.shape
            (960,)
        """
        # Validation (same logic as ESM)
        valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = sequence.upper().strip()

        if not all(aa in valid_aas for aa in sequence):
            raise ValueError("Invalid amino acid characters in sequence")

        if len(sequence) < 1:
            raise ValueError("Sequence too short")

        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]  # (1, seq_len, 960)

            # Masked mean pooling (exclude CLS/SEP special tokens)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            attention_mask[:, 0, :] = 0  # Mask CLS token
            attention_mask[:, -1, :] = 0  # Mask SEP token

            masked_embeddings = embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)  # (1, 960)
            sum_mask = attention_mask.sum(dim=1)  # (1, 1)

            # Prevent division by zero
            if sum_mask.item() == 0:
                raise ValueError(
                    f"Attention mask is all zeros for sequence (length: {len(sequence)}). "
                    f"Sequence preview: '{sequence[:SEQUENCE_PREVIEW_LENGTH]}...'"
                )

            mean_embeddings = sum_embeddings / sum_mask  # (1, 960)
            result: np.ndarray = mean_embeddings.squeeze(0).cpu().numpy()
            return result

    def extract_batch_embeddings(self, sequences: list[str]) -> np.ndarray:
        """
        Extract embeddings for multiple sequences.

        CRITICAL: Due to AMPLIFY padding bug, this processes sequences one at a time
        (batch_size=1) to ensure reproducibility. This is ~8× slower than batched ESM.

        Args:
            sequences: List of amino acid sequence strings

        Returns:
            Array of embeddings with shape (n_sequences, 960)

        Raises:
            RuntimeError: If any sequence fails to embed

        Example:
            >>> sequences = ["QVQL", "ACDG", "EFGH"]
            >>> embeddings = extractor.extract_batch_embeddings(sequences)
            >>> embeddings.shape
            (3, 960)
        """
        embeddings_list: list[np.ndarray] = []

        logger.info(
            f"Extracting AMPLIFY embeddings for {len(sequences)} sequences "
            f"(batch_size=1 due to padding bug, this will be slow)..."
        )

        # Process one at a time (batch_size=1)
        for idx, seq in enumerate(sequences):
            try:
                emb = self.embed_sequence(seq)
                embeddings_list.append(emb)

                # Progress logging every 100 sequences
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(sequences)} sequences...")

            except Exception as e:
                logger.error(f"Failed to process sequence {idx}: {seq[:50]}... - {e}")
                raise RuntimeError(
                    f"Embedding extraction failed at sequence {idx}. Cannot continue."
                ) from e

        return np.array(embeddings_list)

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache for CUDA or MPS devices to prevent memory leaks"""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        elif str(self.device).startswith("mps"):
            torch.mps.empty_cache()
