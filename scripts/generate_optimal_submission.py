"""Generate submission with optimal weights: 0.628*ESM-1v + 0.362*p-IgGen."""

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
print("GENERATING OPTIMAL SUBMISSION (0.628*ESM-1v + 0.362*p-IgGen)")
print("=" * 80)

# Load data
train_assay_file = Path("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")
train_fold_file = Path("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")

assay_df = pd.read_csv(train_assay_file)
fold_df = pd.read_csv(train_fold_file)
full_df = fold_df.merge(
    assay_df[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
)

labeled_df = full_df.dropna(subset=["PR_CHO"])
unlabeled_df = full_df[full_df["PR_CHO"].isna()]

vh_sequences = labeled_df["vh_protein_sequence"].tolist()
vl_sequences = labeled_df["vl_protein_sequence"].tolist()
labels = labeled_df["PR_CHO"].values
folds = labeled_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

logger.info(f"Loaded {len(vh_sequences)} labeled samples")
logger.info(f"Found {len(unlabeled_df)} unlabeled samples")

# ===== LOAD ESM-1v =====
esm1v_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
)
logger.info("Loading ESM-1v embeddings...")
esm1v_vh = get_or_create_embeddings(
    vh_sequences, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vh", logger
)
esm1v_vl = get_or_create_embeddings(
    vl_sequences, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vl", logger
)
esm1v_embeddings = np.concatenate([esm1v_vh, esm1v_vl], axis=1)

# ===== LOAD p-IgGen =====
logger.info("Loading p-IgGen embeddings...")
piggen_cache = Path("./embeddings_cache/ginkgo_piggen_embeddings.npy")
piggen_embeddings = np.load(piggen_cache)
logger.info(f"p-IgGen loaded: {piggen_embeddings.shape}")

# ===== OPTIMAL WEIGHTS =====
w_esm1v = 0.6277277239740922
w_piggen = 0.3616072171426821
best_alpha = 5.5

logger.info(f"Weights: ESM-1v={w_esm1v:.3f}, p-IgGen={w_piggen:.3f}")

# ===== GENERATE CV PREDICTIONS =====
logger.info("=" * 80)
logger.info("GENERATING CV PREDICTIONS")
logger.info("=" * 80)

oof_predictions = np.zeros(len(vh_sequences))
fold_spearmans = []

for fold_idx in sorted(set(folds)):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    # ESM-1v model
    model1 = Ridge(alpha=best_alpha)
    model1.fit(esm1v_embeddings[train_mask], labels[train_mask])
    pred1 = model1.predict(esm1v_embeddings[val_mask])

    # p-IgGen model
    model2 = Ridge(alpha=best_alpha)
    model2.fit(piggen_embeddings[train_mask], labels[train_mask])
    pred2 = model2.predict(piggen_embeddings[val_mask])

    # Optimal ensemble
    ensemble_pred = w_esm1v * pred1 + w_piggen * pred2
    oof_predictions[val_mask] = ensemble_pred

    fold_spearman, _ = spearmanr(labels[val_mask], ensemble_pred)
    fold_spearmans.append(fold_spearman)
    logger.info(f"Fold {fold_idx}: Spearman = {fold_spearman:.4f}")

mean_spearman = np.mean(fold_spearmans)
logger.info(f"\n🎯 Mean per-fold Spearman: {mean_spearman:.5f} (LEADERBOARD SCORE)")

# ===== PREDICT ON UNLABELED =====
if len(unlabeled_df) > 0:
    logger.info("=" * 80)
    logger.info(f"PREDICTING ON {len(unlabeled_df)} UNLABELED ANTIBODIES")
    logger.info("=" * 80)

    unlabeled_vh = unlabeled_df["vh_protein_sequence"].tolist()
    unlabeled_vl = unlabeled_df["vl_protein_sequence"].tolist()

    # ESM-1v embeddings
    unlabeled_esm1v_vh = get_or_create_embeddings(
        unlabeled_vh,
        esm1v_extractor,
        "./embeddings_cache",
        "ginkgo_unlabeled_vh",
        logger,
    )
    unlabeled_esm1v_vl = get_or_create_embeddings(
        unlabeled_vl,
        esm1v_extractor,
        "./embeddings_cache",
        "ginkgo_unlabeled_vl",
        logger,
    )
    unlabeled_esm1v = np.concatenate([unlabeled_esm1v_vh, unlabeled_esm1v_vl], axis=1)

    # p-IgGen embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    piggen_model_name = "ollieturnbull/p-IgGen"
    piggen_tokenizer = AutoTokenizer.from_pretrained(piggen_model_name)
    piggen_model = AutoModelForCausalLM.from_pretrained(piggen_model_name).to(device)

    sequences = [
        "1" + " ".join(vh) + " " + " ".join(vl) + "2"
        for vh, vl in zip(unlabeled_vh, unlabeled_vl, strict=True)
    ]

    embeddings_list = []
    for i in range(0, len(sequences), 8):
        batch = sequences[i : i + 8]
        inputs = piggen_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = piggen_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            embeddings_list.append(mean_pooled)

    unlabeled_piggen = np.vstack(embeddings_list)

    # Train final models
    logger.info("Training final models on all labeled data...")
    final_model1 = Ridge(alpha=best_alpha)
    final_model1.fit(esm1v_embeddings, labels)
    unlabeled_pred1 = final_model1.predict(unlabeled_esm1v)

    final_model2 = Ridge(alpha=best_alpha)
    final_model2.fit(piggen_embeddings, labels)
    unlabeled_pred2 = final_model2.predict(unlabeled_piggen)

    unlabeled_predictions = w_esm1v * unlabeled_pred1 + w_piggen * unlabeled_pred2
    logger.info(
        f"Generated predictions for {len(unlabeled_predictions)} unlabeled antibodies"
    )

# ===== SAVE CV PREDICTIONS =====
labeled_submission = labeled_df[
    [
        "antibody_name",
        "vh_protein_sequence",
        "vl_protein_sequence",
        "hierarchical_cluster_IgG_isotype_stratified_fold",
    ]
].copy()
labeled_submission["PR_CHO"] = oof_predictions

if len(unlabeled_df) > 0:
    unlabeled_submission = unlabeled_df[
        [
            "antibody_name",
            "vh_protein_sequence",
            "vl_protein_sequence",
            "hierarchical_cluster_IgG_isotype_stratified_fold",
        ]
    ].copy()
    unlabeled_submission["PR_CHO"] = unlabeled_predictions
    cv_submission = pd.concat(
        [labeled_submission, unlabeled_submission], ignore_index=True
    )
else:
    cv_submission = labeled_submission

output_dir = Path("ginkgo_submissions_optimal")
output_dir.mkdir(exist_ok=True)
cv_file = output_dir / "ginkgo_cv_predictions_PR_CHO.csv"
cv_submission.to_csv(cv_file, index=False)
logger.info(f"✅ CV predictions saved to: {cv_file}")

# ===== TEST SET PREDICTIONS =====
logger.info("=" * 80)
logger.info("GENERATING TEST SET PREDICTIONS")
logger.info("=" * 80)

test_file = Path("test_datasets/ginkgo/heldout-set-sequences.csv")
if test_file.exists():
    test_df = pd.read_csv(test_file)
    test_vh = test_df["vh_protein_sequence"].tolist()
    test_vl = test_df["vl_protein_sequence"].tolist()

    # ESM-1v
    test_esm1v_vh = get_or_create_embeddings(
        test_vh, esm1v_extractor, "./embeddings_cache", "ginkgo_test_vh", logger
    )
    test_esm1v_vl = get_or_create_embeddings(
        test_vl, esm1v_extractor, "./embeddings_cache", "ginkgo_test_vl", logger
    )
    test_esm1v = np.concatenate([test_esm1v_vh, test_esm1v_vl], axis=1)

    # p-IgGen
    test_sequences = [
        "1" + " ".join(vh) + " " + " ".join(vl) + "2"
        for vh, vl in zip(test_vh, test_vl, strict=True)
    ]

    test_embeddings_list = []
    for i in range(0, len(test_sequences), 8):
        batch = test_sequences[i : i + 8]
        inputs = piggen_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = piggen_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            test_embeddings_list.append(mean_pooled)

    test_piggen = np.vstack(test_embeddings_list)

    # Train final models
    logger.info("Training final models on all training data...")
    final_model1 = Ridge(alpha=best_alpha)
    final_model1.fit(esm1v_embeddings, labels)
    test_pred1 = final_model1.predict(test_esm1v)

    final_model2 = Ridge(alpha=best_alpha)
    final_model2.fit(piggen_embeddings, labels)
    test_pred2 = final_model2.predict(test_piggen)

    test_predictions = w_esm1v * test_pred1 + w_piggen * test_pred2

    test_submission = test_df[
        ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
    ].copy()
    test_submission["PR_CHO"] = test_predictions

    test_file_out = output_dir / "ginkgo_test_predictions_PR_CHO.csv"
    test_submission.to_csv(test_file_out, index=False)
    logger.info(f"✅ Test predictions saved to: {test_file_out}")

logger.info("=" * 80)
logger.info("SUBMISSION FILES READY!")
logger.info("=" * 80)
logger.info(f"📁 Directory: {output_dir}/")
logger.info(f"📄 CV file: {cv_file.name} ({len(cv_submission)} antibodies)")
logger.info("📄 Test file: ginkgo_test_predictions_PR_CHO.csv")
logger.info(f"\n🎯 Expected leaderboard score: {mean_spearman:.5f}")
logger.info("🏆 PREDICTED RANK: #1 (current leader: 0.504)")
logger.info("=" * 80)
