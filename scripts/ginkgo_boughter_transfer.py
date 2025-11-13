"""Transfer Learning Experiment: Boughter Pre-training → GDPa1 Fine-tuning.

Strategy:
1. Pre-train Ridge on Boughter (914 VH+VL pairs, binary ELISA polyreactivity)
2. Fine-tune on GDPa1 (197 labeled pairs, continuous PR_CHO)
3. Compare to baseline (0.500 Spearman)

Competition Compliance:
- Using external data (Boughter) is allowed
- Must report CV on GDPa1 using their folds
- This is the NUCLEAR OPTION to beat current leader (0.504)

Expected gain: +2-7% (5.6x more training data!)
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_cache_key(
    model_name: str, sequences: list[str], revision: str = "main"
) -> str:
    """Compute SHA-256 hash for embedding cache."""
    content = f"{model_name}_{revision}_" + "_".join(sequences[:10])
    return hashlib.sha256(content.encode()).hexdigest()


def extract_esm_embeddings(
    sequences: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 8,
    cache_dir: Path = Path("embeddings_cache"),
) -> np.ndarray:
    """Extract ESM embeddings with caching."""
    cache_dir.mkdir(exist_ok=True)

    # Compute cache key
    cache_key = compute_cache_key(model_name, sequences)
    cache_file = cache_dir / f"{cache_key}.npy"

    if cache_file.exists():
        logger.info(f"Loading cached embeddings: {cache_file}")
        return np.load(cache_file)

    logger.info(f"Extracting {model_name} embeddings for {len(sequences)} sequences...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch)}/{len(sequences)} sequences")

    embeddings = np.vstack(embeddings)
    np.save(cache_file, embeddings)
    logger.info(f"Cached embeddings: {cache_file}")

    return embeddings


def extract_piggen_embeddings(
    vh_seqs: list[str],
    vl_seqs: list[str],
    batch_size: int = 8,
    cache_dir: Path = Path("embeddings_cache"),
) -> np.ndarray:
    """Extract p-IgGen embeddings with caching."""
    cache_dir.mkdir(exist_ok=True)

    # Compute cache key
    combined_seqs = [f"{vh}|{vl}" for vh, vl in zip(vh_seqs, vl_seqs, strict=True)]
    cache_key = compute_cache_key("Exscientia/IgBert", combined_seqs)
    cache_file = cache_dir / f"{cache_key}_piggen.npy"

    if cache_file.exists():
        logger.info(f"Loading cached p-IgGen embeddings: {cache_file}")
        return np.load(cache_file)

    logger.info(f"Extracting p-IgGen embeddings for {len(vh_seqs)} antibodies...")

    tokenizer = AutoTokenizer.from_pretrained("Exscientia/IgBert")
    model = AutoModel.from_pretrained("Exscientia/IgBert")
    model.eval()

    # Format: "1" + space-separated VH + space-separated VL + "2"
    sequences = [
        f"1 {' '.join(vh)} {' '.join(vl)} 2"
        for vh, vl in zip(vh_seqs, vl_seqs, strict=True)
    ]

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch)}/{len(sequences)} antibodies")

    embeddings = np.vstack(embeddings)
    np.save(cache_file, embeddings)
    logger.info(f"Cached p-IgGen embeddings: {cache_file}")

    return embeddings


print("=" * 80)
print("TRANSFER LEARNING: Boughter → GDPa1")
print("=" * 80)

# ===== LOAD DATA =====
logger.info("Loading combined dataset...")
combined = pd.read_csv("combined_datasets/boughter_ginkgo_combined.csv")

boughter_df = combined[combined["dataset"] == "boughter"].copy()
ginkgo_df = combined[combined["dataset"] == "ginkgo"].copy()

logger.info(f"Boughter: {len(boughter_df)} antibodies")
logger.info(f"GDPa1:    {len(ginkgo_df)} antibodies")

# ===== EXTRACT EMBEDDINGS =====
logger.info("\nExtracting ESM-1v embeddings (VH+VL concatenation)...")

# Boughter VH+VL
boughter_vh_vl = [
    vh + vl
    for vh, vl in zip(
        boughter_df["vh_protein_sequence"],
        boughter_df["vl_protein_sequence"],
        strict=True,
    )
]
boughter_esm1v = extract_esm_embeddings(
    boughter_vh_vl, model_name="facebook/esm1v_t33_650M_UR90S_1"
)

# GDPa1 VH+VL
ginkgo_vh_vl = [
    vh + vl
    for vh, vl in zip(
        ginkgo_df["vh_protein_sequence"], ginkgo_df["vl_protein_sequence"], strict=True
    )
]
ginkgo_esm1v = extract_esm_embeddings(
    ginkgo_vh_vl, model_name="facebook/esm1v_t33_650M_UR90S_1"
)

logger.info("\nExtracting p-IgGen embeddings...")
boughter_piggen = extract_piggen_embeddings(
    list(boughter_df["vh_protein_sequence"]), list(boughter_df["vl_protein_sequence"])
)
ginkgo_piggen = extract_piggen_embeddings(
    list(ginkgo_df["vh_protein_sequence"]), list(ginkgo_df["vl_protein_sequence"])
)

logger.info(f"Boughter ESM-1v: {boughter_esm1v.shape}")
logger.info(f"Boughter p-IgGen: {boughter_piggen.shape}")
logger.info(f"GDPa1 ESM-1v: {ginkgo_esm1v.shape}")
logger.info(f"GDPa1 p-IgGen: {ginkgo_piggen.shape}")

# ===== STRATEGY 1: PRE-TRAIN ON BOUGHTER, FINE-TUNE ON GINKGO =====
logger.info("\n" + "=" * 80)
logger.info("STRATEGY 1: Pre-train on Boughter → Fine-tune on GDPa1")
logger.info("=" * 80)

# Optimal prediction weights from previous experiment
w_esm1v = 0.6277277239740922
w_piggen = 0.3616072171426821
best_alpha = 5.5

logger.info(
    f"Using optimal prediction weights: {w_esm1v:.3f}*ESM-1v + {w_piggen:.3f}*p-IgGen"
)
logger.info(f"Ridge alpha: {best_alpha}")

# Get labels and folds
boughter_labels = boughter_df["label_binary"].values
ginkgo_labels = ginkgo_df["PR_CHO"].values
ginkgo_folds = ginkgo_df["fold"].values

# Step 1: Pre-train TWO separate models on Boughter
logger.info("\nStep 1: Pre-training on Boughter (914 samples)...")

# Pre-train ESM-1v model
boughter_model_esm1v = Ridge(alpha=best_alpha, random_state=42)
boughter_model_esm1v.fit(boughter_esm1v, boughter_labels)

# Pre-train p-IgGen model
boughter_model_piggen = Ridge(alpha=best_alpha, random_state=42)
boughter_model_piggen.fit(boughter_piggen, boughter_labels)

# Evaluate on Boughter
boughter_pred_esm1v = boughter_model_esm1v.predict(boughter_esm1v)
boughter_pred_piggen = boughter_model_piggen.predict(boughter_piggen)
boughter_ensemble_pred = w_esm1v * boughter_pred_esm1v + w_piggen * boughter_pred_piggen
boughter_acc = ((boughter_ensemble_pred > 0.5) == boughter_labels).mean()
logger.info(f"Boughter training accuracy (ensemble): {boughter_acc:.4f}")

# Step 2: Fine-tune on GDPa1 with CV
logger.info("\nStep 2: Fine-tuning on GDPa1 with CV (197 samples)...")

oof_predictions = np.zeros(len(ginkgo_df))
fold_spearmans = []

for fold_idx in sorted(set(ginkgo_folds)):
    train_mask = ginkgo_folds != fold_idx
    val_mask = ginkgo_folds == fold_idx

    # Fine-tune ESM-1v model
    model_esm1v = Ridge(alpha=best_alpha, random_state=42)
    model_esm1v.coef_ = boughter_model_esm1v.coef_.copy()  # Transfer weights!
    model_esm1v.intercept_ = boughter_model_esm1v.intercept_
    model_esm1v.fit(ginkgo_esm1v[train_mask], ginkgo_labels[train_mask])

    # Fine-tune p-IgGen model
    model_piggen = Ridge(alpha=best_alpha, random_state=42)
    model_piggen.coef_ = boughter_model_piggen.coef_.copy()  # Transfer weights!
    model_piggen.intercept_ = boughter_model_piggen.intercept_
    model_piggen.fit(ginkgo_piggen[train_mask], ginkgo_labels[train_mask])

    # Predict on validation fold with BOTH models
    val_pred_esm1v = model_esm1v.predict(ginkgo_esm1v[val_mask])
    val_pred_piggen = model_piggen.predict(ginkgo_piggen[val_mask])

    # Ensemble predictions
    val_preds = w_esm1v * val_pred_esm1v + w_piggen * val_pred_piggen
    oof_predictions[val_mask] = val_preds

    # Compute fold Spearman
    fold_spearman, _ = spearmanr(ginkgo_labels[val_mask], val_preds)
    fold_spearmans.append(fold_spearman)

    logger.info(
        f"  Fold {fold_idx}: Spearman = {fold_spearman:.4f} ({val_mask.sum()} samples)"
    )

# Overall results
overall_spearman, _ = spearmanr(ginkgo_labels, oof_predictions)
mean_fold_spearman = np.mean(fold_spearmans)

logger.info("\nTransfer Learning Results:")
logger.info(f"  Overall OOF Spearman: {overall_spearman:.4f}")
logger.info(f"  Mean per-fold Spearman: {mean_fold_spearman:.4f}")
logger.info(f"  Per-fold: {[f'{s:.4f}' for s in fold_spearmans]}")

# ===== STRATEGY 2: COMBINED TRAINING =====
logger.info("\n" + "=" * 80)
logger.info("STRATEGY 2: Combined Training (Boughter + GDPa1)")
logger.info("=" * 80)

# Normalize Boughter binary labels to GDPa1 PR_CHO scale
# Map 0 → GDPa1_low, 1 → GDPa1_high
ginkgo_low = np.percentile(ginkgo_labels, 25)  # Low polyreactivity
ginkgo_high = np.percentile(ginkgo_labels, 75)  # High polyreactivity

boughter_scaled = np.where(boughter_labels == 0, ginkgo_low, ginkgo_high)

logger.info(f"Scaled Boughter labels: 0 → {ginkgo_low:.3f}, 1 → {ginkgo_high:.3f}")

# Train on combined data, predict on GDPa1 folds
oof_predictions_combined = np.zeros(len(ginkgo_df))
fold_spearmans_combined = []

for fold_idx in sorted(set(ginkgo_folds)):
    # Training set: ALL Boughter + GDPa1 training folds
    ginkgo_train_mask = ginkgo_folds != fold_idx
    ginkgo_val_mask = ginkgo_folds == fold_idx

    # Combine Boughter (all) + GDPa1 (training folds) for BOTH embedding types
    # ESM-1v
    train_esm1v = np.vstack([boughter_esm1v, ginkgo_esm1v[ginkgo_train_mask]])
    train_labels = np.concatenate([boughter_scaled, ginkgo_labels[ginkgo_train_mask]])

    # p-IgGen
    train_piggen = np.vstack([boughter_piggen, ginkgo_piggen[ginkgo_train_mask]])

    # Train TWO models on combined data
    model_esm1v = Ridge(alpha=best_alpha, random_state=42)
    model_esm1v.fit(train_esm1v, train_labels)

    model_piggen = Ridge(alpha=best_alpha, random_state=42)
    model_piggen.fit(train_piggen, train_labels)

    # Predict on validation fold with BOTH models
    val_pred_esm1v = model_esm1v.predict(ginkgo_esm1v[ginkgo_val_mask])
    val_pred_piggen = model_piggen.predict(ginkgo_piggen[ginkgo_val_mask])

    # Ensemble predictions
    val_preds = w_esm1v * val_pred_esm1v + w_piggen * val_pred_piggen
    oof_predictions_combined[ginkgo_val_mask] = val_preds

    # Compute fold Spearman
    val_labels = ginkgo_labels[ginkgo_val_mask]
    fold_spearman, _ = spearmanr(val_labels, val_preds)
    fold_spearmans_combined.append(fold_spearman)

    logger.info(
        f"  Fold {fold_idx}: Spearman = {fold_spearman:.4f} "
        f"({len(train_labels)} train, {len(val_labels)} val)"
    )

# Overall results
overall_spearman_combined, _ = spearmanr(ginkgo_labels, oof_predictions_combined)
mean_fold_spearman_combined = np.mean(fold_spearmans_combined)

logger.info("\nCombined Training Results:")
logger.info(f"  Overall OOF Spearman: {overall_spearman_combined:.4f}")
logger.info(f"  Mean per-fold Spearman: {mean_fold_spearman_combined:.4f}")
logger.info(f"  Per-fold: {[f'{s:.4f}' for s in fold_spearmans_combined]}")

# ===== COMPARISON TO BASELINE =====
logger.info("\n" + "=" * 80)
logger.info("COMPARISON TO BASELINE")
logger.info("=" * 80)

baseline_score = 0.50043  # From previous experiment

logger.info(f"Baseline (ESM-1v + p-IgGen, no Boughter): {baseline_score:.5f}")
logger.info(f"Strategy 1 (Transfer Learning):            {mean_fold_spearman:.5f}")
logger.info(
    f"Strategy 2 (Combined Training):            {mean_fold_spearman_combined:.5f}"
)

# Calculate improvement
improvement_s1 = ((mean_fold_spearman - baseline_score) / baseline_score) * 100
improvement_s2 = ((mean_fold_spearman_combined - baseline_score) / baseline_score) * 100

logger.info("\nImprovement vs baseline:")
logger.info(f"  Strategy 1: {improvement_s1:+.2f}%")
logger.info(f"  Strategy 2: {improvement_s2:+.2f}%")

# Determine best strategy
if mean_fold_spearman > mean_fold_spearman_combined:
    best_strategy = "Transfer Learning"
    best_score = mean_fold_spearman
    best_preds = oof_predictions
else:
    best_strategy = "Combined Training"
    best_score = mean_fold_spearman_combined
    best_preds = oof_predictions_combined

logger.info(f"\n🏆 WINNER: {best_strategy} ({best_score:.5f})")

# ===== SAVE RESULTS =====
output_dir = Path("experiment_results")
output_dir.mkdir(exist_ok=True)

results_df = pd.DataFrame(
    {
        "experiment": ["Baseline", "Transfer Learning", "Combined Training"],
        "mean_spearman": [
            baseline_score,
            mean_fold_spearman,
            mean_fold_spearman_combined,
        ],
        "improvement_pct": [0.0, improvement_s1, improvement_s2],
    }
)

results_file = output_dir / "boughter_transfer_results.csv"
results_df.to_csv(results_file, index=False)
logger.info(f"\n✅ Results saved: {results_file}")

# Save OOF predictions
oof_file = output_dir / "boughter_transfer_oof_predictions.csv"
oof_df = pd.DataFrame(
    {
        "antibody_id": ginkgo_df["antibody_id"],
        "antibody_name": ginkgo_df["antibody_name"],
        "true_PR_CHO": ginkgo_labels,
        "pred_transfer": oof_predictions,
        "pred_combined": oof_predictions_combined,
        "fold": ginkgo_folds,
    }
)
oof_df.to_csv(oof_file, index=False)
logger.info(f"✅ OOF predictions saved: {oof_file}")

logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT COMPLETE!")
logger.info("=" * 80)
logger.info(f"Best strategy: {best_strategy}")
logger.info(f"Best score: {best_score:.5f}")
logger.info(f"Baseline: {baseline_score:.5f}")
logger.info("Leader: 0.504")
logger.info(f"Gap to leader: {(0.504 - best_score):.5f}")

if best_score > baseline_score:
    logger.info("\n🚀 IMPROVEMENT FOUND! Ready to generate submission.")
else:
    logger.info("\n⚠️  No improvement. Consider other strategies.")
