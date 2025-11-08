"""
ESM Embedding Module

Professional module for ESM-1V protein sequence embedding extraction.
Handles batch processing, GPU memory management, and validation.
"""

import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_BATCH_SIZE, DEFAULT_MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)


class ESMEmbeddingExtractor:
    """Extract ESM-1V embeddings for protein sequences with proper batching and GPU management"""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ):
        """
        Initialize ESM embedding extractor

        Args:
            model_name: HuggingFace model identifier (e.g., 'facebook/esm1v_t33_650M_UR90S_1')
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            batch_size: Number of sequences to process per batch
            max_length: Maximum sequence length for tokenizer truncation/padding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Load model with output_hidden_states enabled
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(
            f"ESM model {model_name} loaded on {device} with batch_size={batch_size} "
            f"and max_length={max_length}"
        )

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Extract ESM-1V embedding for a single protein sequence

        Args:
            sequence: Amino acid sequence string

        Returns:
            Embedding vector as numpy array

        Raises:
            ValueError: If sequence contains invalid amino acids or is too short
        """
        try:
            # Validate sequence
            valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
            sequence = sequence.upper().strip()

            if not all(aa in valid_aas for aa in sequence):
                raise ValueError("Invalid amino acid characters in sequence")

            if len(sequence) < 1:
                raise ValueError("Sequence too short")

            # Tokenize the sequence
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
                embeddings = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

                # Use attention mask to properly exclude padding and special tokens
                attention_mask = inputs["attention_mask"].unsqueeze(
                    -1
                )  # (batch, seq_len, 1)

                # Mask out special tokens (first and last)
                attention_mask[:, 0, :] = 0  # CLS token
                attention_mask[:, -1, :] = 0  # EOS token

                # Masked mean pooling
                masked_embeddings = embeddings * attention_mask
                sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over sequence
                sum_mask = attention_mask.sum(dim=1)  # Count valid tokens
                mean_embeddings = sum_embeddings / sum_mask  # Average

                result: np.ndarray = mean_embeddings.squeeze(0).cpu().numpy()
                return result

        except Exception as e:
            logger.error(f"Error getting embeddings for sequence: {e}")
            raise

    def extract_batch_embeddings(self, sequences: list[str]) -> np.ndarray:
        """
        Extract embeddings for multiple sequences using efficient batching

        Args:
            sequences: List of amino acid sequence strings

        Returns:
            Array of embeddings with shape (n_sequences, embedding_dim)
        """
        embeddings_list = []

        logger.info(
            f"Extracting embeddings for {len(sequences)} sequences with batch_size={self.batch_size}..."
        )

        # Process sequences in batches
        num_batches = (len(sequences) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]

            try:
                # Validate and clean sequences
                valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
                cleaned_sequences: list[str] = []
                for seq in batch_sequences:
                    seq = seq.upper().strip()
                    if not all(aa in valid_aas for aa in seq) or len(seq) < 1:
                        logger.warning(
                            f"Invalid sequence at index {start_idx + len(cleaned_sequences)}, using zeros"
                        )
                        cleaned_sequences.append(
                            "M"
                        )  # Placeholder for invalid sequences
                    else:
                        cleaned_sequences.append(seq)

                # Tokenize the batch with padding
                inputs = self.tokenizer(
                    cleaned_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings for the batch
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    embeddings = outputs.hidden_states[
                        -1
                    ]  # (batch, seq_len, hidden_dim)

                    # Use attention mask to properly exclude padding and special tokens
                    attention_mask = inputs["attention_mask"].unsqueeze(
                        -1
                    )  # (batch, seq_len, 1)

                    # Mask out special tokens (first and last)
                    attention_mask[:, 0, :] = 0  # CLS token
                    attention_mask[:, -1, :] = 0  # EOS token

                    # Masked mean pooling
                    masked_embeddings = embeddings * attention_mask
                    sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over sequence
                    sum_mask = attention_mask.sum(dim=1)  # Count valid tokens
                    mean_embeddings = sum_embeddings / sum_mask  # Average

                    # Convert to numpy and add to list
                    batch_embeddings = mean_embeddings.cpu().numpy()
                    for emb in batch_embeddings:
                        embeddings_list.append(emb)

                # Clear GPU cache periodically to prevent OOM
                if (batch_idx + 1) % 10 == 0:
                    self._clear_gpu_cache()

            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                # Add zero embeddings for failed batch
                for _ in range(len(batch_sequences)):
                    embeddings_list.append(np.zeros(1280))  # ESM-1V embedding dimension

        # Final cache clear
        self._clear_gpu_cache()
        return np.array(embeddings_list)

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache for CUDA or MPS devices to prevent memory leaks"""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        elif str(self.device).startswith("mps"):
            torch.mps.empty_cache()
