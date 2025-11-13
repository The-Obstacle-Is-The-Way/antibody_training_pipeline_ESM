"""Tessier → GDPa1 Transfer Learning (Multi-Task Learning)

Strategy:
1. Train a SINGLE PyTorch model on BOTH datasets simultaneously
2. Shared representation layer learns from 298k Tessier sequences
3. Task-specific heads for binary (Tessier) and continuous (GDPa1) labels
4. Use optimal embeddings: 62.8% ESM-1v + 36.2% p-IgGen

Expected: Multi-task learning leverages Tessier's massive dataset to improve
GDPa1 predictions, beating 0.507 baseline (target: 0.52+).

Research basis: Scientific Reports (2022) shows multi-task achieves same
performance as single-task with 1/8th the data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ===== MULTI-TASK MODEL DEFINITION =====
class MultiTaskHead(nn.Module):
    """Multi-task model with shared representation and task-specific heads.

    Architecture:
    - Shared layer: 3328D embeddings → 512D representation
    - Tessier head: 512D → 1D (binary classification, sigmoid)
    - GDPa1 head: 512D → 1D (continuous regression, linear)

    Training:
    - Loss = α * BCE(Tessier) + (1-α) * MSE(GDPa1)
    - Both gradients flow through shared layer
    - Tessier's 298k samples teach general polyreactivity patterns
    - GDPa1's 197 samples specialize the regression head
    """

    def __init__(
        self, input_dim: int = 3328, hidden_dim: int = 512, dropout: float = 0.2
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tessier_head = nn.Linear(hidden_dim, 1)  # Binary classification
        self.gdpa1_head = nn.Linear(hidden_dim, 1)  # Continuous regression

    def forward(self, x: torch.Tensor, task: str = "gdpa1") -> torch.Tensor:
        """Forward pass through shared layer and task-specific head."""
        shared_repr = self.shared(x)

        if task == "tessier":
            return torch.sigmoid(self.tessier_head(shared_repr))
        elif task == "gdpa1":
            return self.gdpa1_head(shared_repr)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'tessier' or 'gdpa1'.")


print("=" * 80)
print("TESSIER → GDPa1 MULTI-TASK TRANSFER LEARNING")
print("=" * 80)

# ===== HYPERPARAMETERS =====
W_ESM1V = 0.6277277239740922
W_PIGGEN = 0.3616072171426821

# Multi-task hyperparameters
HIDDEN_DIM = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
DROPOUT = 0.2
TASK_WEIGHT_ALPHA = 0.5  # Balance between Tessier (α) and GDPa1 (1-α)
NUM_EPOCHS = 100
BATCH_SIZE_TESSIER = 32
BATCH_SIZE_GDPA1 = 16

logger.info("=" * 80)
logger.info("MULTI-TASK HYPERPARAMETERS")
logger.info("=" * 80)
logger.info(f"Embedding weights: {W_ESM1V:.3f}*ESM-1v + {W_PIGGEN:.3f}*p-IgGen")
logger.info(f"Hidden dim: {HIDDEN_DIM}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info(f"Weight decay (L2): {WEIGHT_DECAY}")
logger.info(f"Dropout: {DROPOUT}")
logger.info(f"Task weight α: {TASK_WEIGHT_ALPHA:.2f} (Tessier vs GDPa1)")
logger.info(f"Epochs: {NUM_EPOCHS}")
logger.info(f"Batch sizes: Tessier={BATCH_SIZE_TESSIER}, GDPa1={BATCH_SIZE_GDPA1}")

# ===== STEP 1: LOAD TESSIER DATA (SEPARATE VH/VL) =====
logger.info("=" * 80)
logger.info("STEP 1: LOADING TESSIER DATA (298K SEQUENCES - VH+VL SEPARATE)")
logger.info("=" * 80)

# Load VH and VL separately (like GDPa1) to ensure dimension compatibility
tessier_vh_file = Path("train_datasets/tessier/annotated/VH_only_tessier.csv")
tessier_vl_file = Path("train_datasets/tessier/annotated/VL_only_tessier.csv")

tessier_vh_df = pd.read_csv(tessier_vh_file)
tessier_vl_df = pd.read_csv(tessier_vl_file)

# Filter to training set only
tessier_train_vh = tessier_vh_df[tessier_vh_df["split"] == "train"].copy()
tessier_train_vl = tessier_vl_df[tessier_vl_df["split"] == "train"].copy()

logger.info(f"Tessier training set: {len(tessier_train_vh)} sequences")
logger.info(
    f"  Polyreactive: {(tessier_train_vh['label'] == 1).sum()}, "
    f"Specific: {(tessier_train_vh['label'] == 0).sum()}"
)

# Extract VH and VL sequences separately
vh_sequences_tessier = tessier_train_vh["sequence"].tolist()
vl_sequences_tessier = tessier_train_vl["sequence"].tolist()
tessier_labels = tessier_train_vh["label"].values  # Labels are identical in both files

# ===== STEP 2: EXTRACT TESSIER EMBEDDINGS (SEPARATE VH/VL) =====
logger.info("=" * 80)
logger.info("STEP 2: EXTRACTING TESSIER EMBEDDINGS (VH+VL SEPARATE)")
logger.info("=" * 80)

esm1v_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1", device="mps", batch_size=16
)

# Extract VH and VL embeddings separately (like GDPa1) to get 3328D total
logger.info(
    "Extracting ESM-1v embeddings for Tessier VH (this will take ~30-45 min)..."
)
tessier_esm1v_vh = get_or_create_embeddings(
    vh_sequences_tessier,
    esm1v_extractor,
    "./embeddings_cache",
    "tessier_train_vh",
    logger,
)

logger.info(
    "Extracting ESM-1v embeddings for Tessier VL (this will take ~30-45 min)..."
)
tessier_esm1v_vl = get_or_create_embeddings(
    vl_sequences_tessier,
    esm1v_extractor,
    "./embeddings_cache",
    "tessier_train_vl",
    logger,
)

# Concatenate VH and VL embeddings: 1280D + 1280D = 2560D (matching GDPa1)
tessier_esm1v = np.concatenate([tessier_esm1v_vh, tessier_esm1v_vl], axis=1)
logger.info(f"Concatenated Tessier ESM-1v embeddings: {tessier_esm1v.shape}")

# Load p-IgGen model for embedding extraction
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
piggen_model_name = "ollieturnbull/p-IgGen"
logger.info(f"Loading p-IgGen model: {piggen_model_name}")
piggen_tokenizer = AutoTokenizer.from_pretrained(piggen_model_name)
piggen_model = AutoModelForCausalLM.from_pretrained(piggen_model_name).to(device)
piggen_model.eval()


def encode_piggen_embeddings_paired(
    vh_sequences: list[str], vl_sequences: list[str], batch_size: int = 8
) -> np.ndarray:
    """Encode VH+VL sequence pairs with p-IgGen model.

    Format matches GDPa1: "1 V H S E Q V L S E Q 2" (space-separated amino acids)
    """
    # Format: "1 {VH} {VL} 2" with space-separated amino acids
    formatted_seqs = [
        f"1 {' '.join(vh)} {' '.join(vl)} 2"
        for vh, vl in zip(vh_sequences, vl_sequences, strict=True)
    ]

    embeddings: list[np.ndarray] = []
    for i in range(0, len(formatted_seqs), batch_size):
        batch = formatted_seqs[i : i + batch_size]
        inputs = piggen_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = piggen_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled)

        if (i // batch_size + 1) % 50 == 0:
            logger.info(
                f"  p-IgGen progress: {min(i + batch_size, len(formatted_seqs))}/{len(formatted_seqs)}"
            )

    return np.vstack(embeddings)


logger.info("Extracting p-IgGen embeddings for Tessier (VH+VL pairs)...")
tessier_piggen_cache = Path("./embeddings_cache/tessier_piggen_embeddings.npy")
if tessier_piggen_cache.exists():
    tessier_piggen = np.load(tessier_piggen_cache)
    logger.info(f"Loaded cached p-IgGen embeddings: {tessier_piggen.shape}")
else:
    tessier_piggen = encode_piggen_embeddings_paired(
        vh_sequences_tessier, vl_sequences_tessier
    )
    np.save(tessier_piggen_cache, tessier_piggen)
    logger.info(f"Cached p-IgGen embeddings: {tessier_piggen.shape}")

# Combine embeddings with optimal weights
tessier_combined = np.concatenate(
    [W_ESM1V * tessier_esm1v, W_PIGGEN * tessier_piggen], axis=1
)
logger.info(f"Combined Tessier embeddings: {tessier_combined.shape}")

# ===== STEP 3: LOAD GDPa1 DATA =====
logger.info("=" * 80)
logger.info("STEP 3: LOADING GDPa1 DATA (197 SEQUENCES)")
logger.info("=" * 80)

train_assay_file = Path("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")
train_fold_file = Path("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")

assay_df = pd.read_csv(train_assay_file)
fold_df = pd.read_csv(train_fold_file)
full_df = fold_df.merge(
    assay_df[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
)

labeled_df = full_df.dropna(subset=["PR_CHO"])

vh_sequences = labeled_df["vh_protein_sequence"].tolist()
vl_sequences = labeled_df["vl_protein_sequence"].tolist()
gdpa1_labels = labeled_df["PR_CHO"].values
folds = labeled_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

logger.info(f"GDPa1 labeled: {len(vh_sequences)} sequences")

# ===== STEP 4: EXTRACT GDPa1 EMBEDDINGS =====
logger.info("=" * 80)
logger.info("STEP 4: EXTRACTING GDPa1 EMBEDDINGS")
logger.info("=" * 80)

# Extract ESM-1v embeddings for GDPa1
logger.info("Extracting ESM-1v embeddings for GDPa1...")
gdpa1_esm1v_vh = get_or_create_embeddings(
    vh_sequences, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vh", logger
)
gdpa1_esm1v_vl = get_or_create_embeddings(
    vl_sequences, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vl", logger
)
gdpa1_esm1v = np.concatenate([gdpa1_esm1v_vh, gdpa1_esm1v_vl], axis=1)

# Extract p-IgGen embeddings for GDPa1
logger.info("Extracting p-IgGen embeddings for GDPa1...")
gdpa1_piggen_cache = Path("./embeddings_cache/ginkgo_piggen_embeddings.npy")
gdpa1_piggen = np.load(gdpa1_piggen_cache)

# Combine with optimal weights
gdpa1_combined = np.concatenate(
    [W_ESM1V * gdpa1_esm1v, W_PIGGEN * gdpa1_piggen], axis=1
)
logger.info(f"Combined GDPa1 embeddings: {gdpa1_combined.shape}")

# ===== STEP 5: MULTI-TASK TRANSFER LEARNING =====
logger.info("=" * 80)
logger.info("STEP 5: MULTI-TASK TRANSFER LEARNING")
logger.info("=" * 80)

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Convert numpy arrays to PyTorch tensors
tessier_X = torch.from_numpy(tessier_combined).float()
tessier_y = torch.from_numpy(tessier_labels.astype(np.float32)).float().unsqueeze(1)
gdpa1_X = torch.from_numpy(gdpa1_combined).float()
gdpa1_y = torch.from_numpy(gdpa1_labels.astype(np.float32)).float().unsqueeze(1)

logger.info(f"Tessier: {tessier_X.shape[0]} samples")
logger.info(f"GDPa1: {gdpa1_X.shape[0]} samples")

# Run 5-fold cross-validation
logger.info("Running 5-fold cross-validation with multi-task learning...")
fold_spearmans_multitask = []
oof_multitask = np.zeros(len(gdpa1_labels))

for fold_idx in sorted(set(folds)):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FOLD {fold_idx}")
    logger.info(f"{'=' * 60}")

    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    # GDPa1 train/val split
    X_train_gdpa1 = gdpa1_X[train_mask].to(device)
    y_train_gdpa1 = gdpa1_y[train_mask].to(device)
    X_val_gdpa1 = gdpa1_X[val_mask].to(device)
    y_val_gdpa1 = gdpa1_y[val_mask].to(device)

    # Full Tessier data (always use all of it)
    X_tessier = tessier_X.to(device)
    y_tessier = tessier_y.to(device)

    logger.info(f"  GDPa1 train: {X_train_gdpa1.shape[0]} samples")
    logger.info(f"  GDPa1 val: {X_val_gdpa1.shape[0]} samples")
    logger.info(f"  Tessier: {X_tessier.shape[0]} samples")

    # Initialize model
    model = MultiTaskHead(
        input_dim=gdpa1_combined.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(device)

    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # Training loop
    model.train()
    best_val_spearman = -1.0
    patience = 0
    patience_limit = 10

    for epoch in range(NUM_EPOCHS):
        # Sample random batches from Tessier
        tessier_indices = torch.randperm(len(X_tessier))[:BATCH_SIZE_TESSIER]
        X_batch_tessier = X_tessier[tessier_indices]
        y_batch_tessier = y_tessier[tessier_indices]

        # Sample random batches from GDPa1 training set
        gdpa1_indices = torch.randperm(len(X_train_gdpa1))[:BATCH_SIZE_GDPA1]
        X_batch_gdpa1 = X_train_gdpa1[gdpa1_indices]
        y_batch_gdpa1 = y_train_gdpa1[gdpa1_indices]

        # Forward pass on both tasks
        pred_tessier = model(X_batch_tessier, task="tessier")
        pred_gdpa1 = model(X_batch_gdpa1, task="gdpa1")

        # Compute losses
        loss_tessier = bce_loss(pred_tessier, y_batch_tessier)
        loss_gdpa1 = mse_loss(pred_gdpa1, y_batch_gdpa1)

        # Weighted combination
        total_loss = (
            TASK_WEIGHT_ALPHA * loss_tessier + (1 - TASK_WEIGHT_ALPHA) * loss_gdpa1
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_gdpa1, task="gdpa1")
                val_spearman, _ = spearmanr(
                    y_val_gdpa1.cpu().numpy(), val_pred.cpu().numpy()
                )

                if val_spearman > best_val_spearman:
                    best_val_spearman = val_spearman
                    patience = 0
                else:
                    patience += 1

                logger.info(
                    f"  Epoch {epoch + 1:3d}: "
                    f"Loss={total_loss.item():.4f} "
                    f"(Tessier={loss_tessier.item():.4f}, GDPa1={loss_gdpa1.item():.4f}), "
                    f"Val Spearman={val_spearman:.4f}"
                )

            model.train()

            # Early stopping
            if patience >= patience_limit:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    # Final validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_gdpa1, task="gdpa1").cpu().numpy()
        oof_multitask[val_mask] = val_pred.flatten()

        fold_spearman, _ = spearmanr(y_val_gdpa1.cpu().numpy(), val_pred)
        fold_spearmans_multitask.append(fold_spearman)

        logger.info(f"\n  ✅ Fold {fold_idx} Final Spearman: {fold_spearman:.5f}")

mean_multitask = np.mean(fold_spearmans_multitask)

logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS")
logger.info("=" * 80)
logger.info(f"🎯 Multi-Task Learning: {mean_multitask:.5f} Spearman")
logger.info(
    f"📊 vs Baseline (0.507): {mean_multitask - 0.507:+.5f} ({(mean_multitask / 0.507 - 1) * 100:+.2f}%)"
)
logger.info(f"🏆 vs Leader (0.89): {mean_multitask - 0.89:+.3f}")
logger.info(f"\nPer-fold results: {[f'{s:.4f}' for s in fold_spearmans_multitask]}")

if mean_multitask > 0.507:
    logger.info("\n✅ MULTI-TASK TRANSFER LEARNING WINS!")
    logger.info("   Transfer learning successfully improved over baseline!")
else:
    logger.info("\n⚠️  Multi-task did not beat baseline (0.507)")
    logger.info("   Consider tuning: α, learning rate, hidden_dim, dropout")

logger.info("=" * 80)
