"""Transfer Learning Experiment: Boughter Pre-training + GDPa1 Fine-tuning.

Strategy:
1. Pre-train on Boughter (914 samples, ELISA polyreactivity)
2. Fine-tune on GDPa1 (197 samples, PR_CHO polyreactivity)
3. Compare against baseline (GDPa1 only)

Hypothesis: More training data = better protein representations = better generalization

Competition Compliance:
- Train on external data (allowed by rules)
- Report CV on GDPa1 folds (required)
- Use identical fold splits for fair comparison
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 80)
print("TRANSFER LEARNING: Boughter → GDPa1")
print("=" * 80)

# ===== LOAD COMBINED DATASET =====
logger.info("Loading combined dataset...")

boughter_df = pd.read_csv("combined_datasets/boughter_training.csv")
ginkgo_df = pd.read_csv("combined_datasets/ginkgo_labeled.csv")

logger.info(f"Boughter: {len(boughter_df)} antibodies")
logger.info(f"GDPa1:    {len(ginkgo_df)} antibodies")

# ===== EXTRACT EMBEDDINGS =====
logger.info("\n" + "=" * 80)
logger.info("EXTRACTING EMBEDDINGS (ESM-1v + p-IgGen)")
logger.info("=" * 80)

# ESM-1v embeddings
esm1v_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
)

# Boughter embeddings
logger.info("Extracting Boughter embeddings...")
boughter_vh_seqs = boughter_df["vh_protein_sequence"].tolist()
boughter_vl_seqs = boughter_df["vl_protein_sequence"].tolist()
boughter_labels = boughter_df["label_binary"].values

boughter_vh_emb = get_or_create_embeddings(
    boughter_vh_seqs, esm1v_extractor, "./embeddings_cache", "boughter_vh", logger
)
boughter_vl_emb = get_or_create_embeddings(
    boughter_vl_seqs, esm1v_extractor, "./embeddings_cache", "boughter_vl", logger
)
boughter_esm1v = np.concatenate([boughter_vh_emb, boughter_vl_emb], axis=1)

logger.info(f"Boughter ESM-1v shape: {boughter_esm1v.shape}")

# GDPa1 embeddings
logger.info("Extracting GDPa1 embeddings...")
ginkgo_vh_seqs = ginkgo_df["vh_protein_sequence"].tolist()
ginkgo_vl_seqs = ginkgo_df["vl_protein_sequence"].tolist()
ginkgo_labels = ginkgo_df["PR_CHO"].values
ginkgo_folds = ginkgo_df["fold"].values

ginkgo_vh_emb = get_or_create_embeddings(
    ginkgo_vh_seqs, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vh", logger
)
ginkgo_vl_emb = get_or_create_embeddings(
    ginkgo_vl_seqs, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vl", logger
)
ginkgo_esm1v = np.concatenate([ginkgo_vh_emb, ginkgo_vl_emb], axis=1)

logger.info(f"GDPa1 ESM-1v shape: {ginkgo_esm1v.shape}")

# p-IgGen embeddings
logger.info("Extracting p-IgGen embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
piggen_model_name = "ollieturnbull/p-IgGen"
piggen_tokenizer = AutoTokenizer.from_pretrained(piggen_model_name)
piggen_model = AutoModelForCausalLM.from_pretrained(piggen_model_name).to(device)


def embed_piggen(vh_seqs, vl_seqs, batch_size=8):
    """Extract p-IgGen embeddings with format: 1 + VH + VL + 2."""
    sequences = [
        "1" + " ".join(vh) + " " + " ".join(vl) + "2"
        for vh, vl in zip(vh_seqs, vl_seqs, strict=True)
    ]

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = piggen_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = piggen_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled)

    return np.vstack(embeddings)


boughter_piggen_cache = Path("./embeddings_cache/boughter_piggen_embeddings.npy")
if boughter_piggen_cache.exists():
    boughter_piggen = np.load(boughter_piggen_cache)
    logger.info(f"Boughter p-IgGen loaded from cache: {boughter_piggen.shape}")
else:
    boughter_piggen = embed_piggen(boughter_vh_seqs, boughter_vl_seqs)
    np.save(boughter_piggen_cache, boughter_piggen)
    logger.info(f"Boughter p-IgGen extracted: {boughter_piggen.shape}")

ginkgo_piggen_cache = Path("./embeddings_cache/ginkgo_piggen_embeddings.npy")
if ginkgo_piggen_cache.exists():
    ginkgo_piggen = np.load(ginkgo_piggen_cache)
    logger.info(f"GDPa1 p-IgGen loaded from cache: {ginkgo_piggen.shape}")
else:
    ginkgo_piggen = embed_piggen(ginkgo_vh_seqs, ginkgo_vl_seqs)
    np.save(ginkgo_piggen_cache, ginkgo_piggen)
    logger.info(f"GDPa1 p-IgGen extracted: {ginkgo_piggen.shape}")

# ===== EXPERIMENT 1: BASELINE (GDPa1 only) =====
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 1: BASELINE (GDPa1 only)")
logger.info("=" * 80)

best_alpha = 5.5
w_esm1v = 0.6277277239740922
w_piggen = 0.3616072171426821

baseline_oof = np.zeros(len(ginkgo_labels))
baseline_fold_spearmans = []

for fold_idx in sorted(set(ginkgo_folds)):
    train_mask = ginkgo_folds != fold_idx
    val_mask = ginkgo_folds == fold_idx

    # ESM-1v model
    model1 = Ridge(alpha=best_alpha)
    model1.fit(ginkgo_esm1v[train_mask], ginkgo_labels[train_mask])
    pred1 = model1.predict(ginkgo_esm1v[val_mask])

    # p-IgGen model
    model2 = Ridge(alpha=best_alpha)
    model2.fit(ginkgo_piggen[train_mask], ginkgo_labels[train_mask])
    pred2 = model2.predict(ginkgo_piggen[val_mask])

    # Ensemble
    ensemble_pred = w_esm1v * pred1 + w_piggen * pred2
    baseline_oof[val_mask] = ensemble_pred

    fold_spearman, _ = spearmanr(ginkgo_labels[val_mask], ensemble_pred)
    baseline_fold_spearmans.append(fold_spearman)
    logger.info(f"Fold {fold_idx}: Spearman = {fold_spearman:.4f}")

baseline_score = np.mean(baseline_fold_spearmans)
logger.info(f"\n🎯 BASELINE Score: {baseline_score:.5f}")

# ===== EXPERIMENT 2: TRANSFER LEARNING =====
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 2: TRANSFER LEARNING (Boughter → GDPa1)")
logger.info("=" * 80)

# Normalize Boughter labels to GDPa1 scale
boughter_labels_norm = boughter_labels * ginkgo_labels.std() + ginkgo_labels.mean()

logger.info("Boughter label normalization:")
logger.info("  - Original: binary 0/1")
logger.info(f"  - Normalized mean: {boughter_labels_norm.mean():.3f}")
logger.info(f"  - Normalized std: {boughter_labels_norm.std():.3f}")
logger.info(f"  - GDPa1 mean: {ginkgo_labels.mean():.3f}")
logger.info(f"  - GDPa1 std: {ginkgo_labels.std():.3f}")

transfer_oof = np.zeros(len(ginkgo_labels))
transfer_fold_spearmans = []

for fold_idx in sorted(set(ginkgo_folds)):
    train_mask = ginkgo_folds != fold_idx
    val_mask = ginkgo_folds == fold_idx

    # ===== ESM-1v Transfer =====
    # Step 1: Pre-train on Boughter
    esm1v_pretrain = Ridge(alpha=best_alpha)
    esm1v_pretrain.fit(boughter_esm1v, boughter_labels_norm)

    # Step 2: Fine-tune on GDPa1 training fold
    esm1v_finetune = Ridge(alpha=best_alpha)
    esm1v_finetune.coef_ = esm1v_pretrain.coef_.copy()  # Transfer weights
    esm1v_finetune.intercept_ = esm1v_pretrain.intercept_
    esm1v_finetune.fit(ginkgo_esm1v[train_mask], ginkgo_labels[train_mask])
    pred1 = esm1v_finetune.predict(ginkgo_esm1v[val_mask])

    # ===== p-IgGen Transfer =====
    # Step 1: Pre-train on Boughter
    piggen_pretrain = Ridge(alpha=best_alpha)
    piggen_pretrain.fit(boughter_piggen, boughter_labels_norm)

    # Step 2: Fine-tune on GDPa1 training fold
    piggen_finetune = Ridge(alpha=best_alpha)
    piggen_finetune.coef_ = piggen_pretrain.coef_.copy()  # Transfer weights
    piggen_finetune.intercept_ = piggen_pretrain.intercept_
    piggen_finetune.fit(ginkgo_piggen[train_mask], ginkgo_labels[train_mask])
    pred2 = piggen_finetune.predict(ginkgo_piggen[val_mask])

    # Ensemble
    ensemble_pred = w_esm1v * pred1 + w_piggen * pred2
    transfer_oof[val_mask] = ensemble_pred

    fold_spearman, _ = spearmanr(ginkgo_labels[val_mask], ensemble_pred)
    transfer_fold_spearmans.append(fold_spearman)
    logger.info(f"Fold {fold_idx}: Spearman = {fold_spearman:.4f}")

transfer_score = np.mean(transfer_fold_spearmans)
logger.info(f"\n🎯 TRANSFER LEARNING Score: {transfer_score:.5f}")

# ===== EXPERIMENT 3: COMBINED TRAINING =====
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 3: COMBINED TRAINING (Boughter + GDPa1 merged)")
logger.info("=" * 80)

combined_oof = np.zeros(len(ginkgo_labels))
combined_fold_spearmans = []

for fold_idx in sorted(set(ginkgo_folds)):
    train_mask = ginkgo_folds != fold_idx
    val_mask = ginkgo_folds == fold_idx

    # Combine Boughter + GDPa1 training fold
    combined_esm1v = np.vstack([boughter_esm1v, ginkgo_esm1v[train_mask]])
    combined_piggen = np.vstack([boughter_piggen, ginkgo_piggen[train_mask]])
    combined_labels = np.concatenate([boughter_labels_norm, ginkgo_labels[train_mask]])

    # Train on ALL data
    model1 = Ridge(alpha=best_alpha)
    model1.fit(combined_esm1v, combined_labels)
    pred1 = model1.predict(ginkgo_esm1v[val_mask])

    model2 = Ridge(alpha=best_alpha)
    model2.fit(combined_piggen, combined_labels)
    pred2 = model2.predict(ginkgo_piggen[val_mask])

    # Ensemble
    ensemble_pred = w_esm1v * pred1 + w_piggen * pred2
    combined_oof[val_mask] = ensemble_pred

    fold_spearman, _ = spearmanr(ginkgo_labels[val_mask], ensemble_pred)
    combined_fold_spearmans.append(fold_spearman)
    logger.info(f"Fold {fold_idx}: Spearman = {fold_spearman:.4f}")

combined_score = np.mean(combined_fold_spearmans)
logger.info(f"\n🎯 COMBINED TRAINING Score: {combined_score:.5f}")

# ===== FINAL COMPARISON =====
logger.info("\n" + "=" * 80)
logger.info("FINAL COMPARISON")
logger.info("=" * 80)
logger.info("Current leader:           0.504")
logger.info(f"1. Baseline (GDPa1 only): {baseline_score:.5f}")
logger.info(
    f"2. Transfer Learning:     {transfer_score:.5f} (gain: {transfer_score - baseline_score:+.5f})"
)
logger.info(
    f"3. Combined Training:     {combined_score:.5f} (gain: {combined_score - baseline_score:+.5f})"
)
logger.info("=" * 80)

# Save results
results_df = pd.DataFrame(
    [
        {"experiment": "baseline_ginkgo_only", "score": baseline_score},
        {"experiment": "transfer_learning", "score": transfer_score},
        {"experiment": "combined_training", "score": combined_score},
    ]
)
results_df["gain_vs_baseline"] = results_df["score"] - baseline_score
results_df.to_csv("boughter_transfer_results.csv", index=False)
logger.info("✅ Results saved to: boughter_transfer_results.csv")

# Determine best approach
best_experiment = results_df.loc[results_df["score"].idxmax()]
logger.info(f"\n🏆 BEST APPROACH: {best_experiment['experiment']}")
logger.info(f"   Score: {best_experiment['score']:.5f}")
logger.info(f"   Gain: {best_experiment['gain_vs_baseline']:+.5f}")

if best_experiment["score"] > 0.504:
    logger.info("\n🎉 SUCCESS! Beat leader (0.504)")
else:
    gap = 0.504 - best_experiment["score"]
    logger.info(f"\n⚠️  Still {gap:.5f} behind leader (0.504)")
