"""Generate submission with AGGRESSIVE LightGBM on optimal embeddings.

Iteration 1A: More aggressive hyperparameters to beat Ridge baseline (0.507)
- num_leaves: 16 → 64 (4x increase)
- max_depth: 4 → 7
- learning_rate: 0.01 → 0.03
- n_estimators: 1000 → 2000
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 80)
print("LIGHTGBM AGGRESSIVE (Iteration 1A): Deeper Trees + Faster Learning")
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

# ===== LOAD p-IgGen MODEL FOR INFERENCE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
piggen_model_name = "ollieturnbull/p-IgGen"
logger.info(f"Loading p-IgGen model: {piggen_model_name}")
piggen_tokenizer = AutoTokenizer.from_pretrained(piggen_model_name)
piggen_model = AutoModelForCausalLM.from_pretrained(piggen_model_name).to(device)
piggen_model.eval()  # CRITICAL: Set to eval mode to disable dropout
logger.info("p-IgGen model loaded and set to eval mode")


def encode_piggen_embeddings(
    vh_list: list[str], vl_list: list[str], batch_size: int = 8
) -> np.ndarray:
    """Encode VH/VL pairs with the p-IgGen model using proper sentinel formatting."""
    sequences = [
        f"1 {' '.join(vh)} {' '.join(vl)} 2"
        for vh, vl in zip(vh_list, vl_list, strict=True)
    ]

    embeddings: list[np.ndarray] = []
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

        if (i // batch_size + 1) % 10 == 0:
            logger.info(
                "  p-IgGen encoding progress: %d/%d sequences",
                min(i + batch_size, len(sequences)),
                len(sequences),
            )

    return np.vstack(embeddings)


# ===== LOAD p-IgGen =====
logger.info("Loading p-IgGen embeddings...")
piggen_cache = Path("./embeddings_cache/ginkgo_piggen_embeddings.npy")
if piggen_cache.exists():
    piggen_embeddings = np.load(piggen_cache)
    logger.info(f"p-IgGen loaded: {piggen_embeddings.shape}")
else:
    logger.info(
        "p-IgGen cache not found — regenerating embeddings with correct formatting"
    )
    piggen_embeddings = encode_piggen_embeddings(vh_sequences, vl_sequences)
    np.save(piggen_cache, piggen_embeddings)
    logger.info(f"p-IgGen embeddings regenerated and cached: {piggen_embeddings.shape}")

# ===== COMBINE EMBEDDINGS WITH OPTIMAL WEIGHTS =====
w_esm1v = 0.6277277239740922
w_piggen = 0.3616072171426821

logger.info(f"Combining embeddings: {w_esm1v:.3f}*ESM-1v + {w_piggen:.3f}*p-IgGen")
combined_embeddings = np.concatenate(
    [w_esm1v * esm1v_embeddings, w_piggen * piggen_embeddings], axis=1
)
logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")

# ===== LIGHTGBM HYPERPARAMETERS =====
# Conservative settings (as recommended in roadmap)
lgbm_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 64,  # Conservative (roadmap: start with 16)
    "max_depth": 7,  # Conservative (roadmap: start with 4)
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 100  # Stop if no improvement for 100 rounds

logger.info("LightGBM hyperparameters:")
for key, val in lgbm_params.items():
    logger.info(f"  {key}: {val}")

# ===== GENERATE CV PREDICTIONS =====
logger.info("=" * 80)
logger.info("CROSS-VALIDATION WITH LIGHTGBM")
logger.info("=" * 80)

oof_predictions = np.zeros(len(vh_sequences))
fold_spearmans = []

for fold_idx in sorted(set(folds)):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train = combined_embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = combined_embeddings[val_mask]
    y_val = labels[val_mask]

    # Train LightGBM with early stopping (100 rounds patience)
    model = LGBMRegressor(**lgbm_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
    )

    best_iter = (
        model.best_iteration_
        if model.best_iteration_ is not None
        else model.n_estimators
    )

    # Predict on validation fold
    val_pred = model.predict(X_val)
    oof_predictions[val_mask] = val_pred

    fold_spearman, _ = spearmanr(y_val, val_pred)
    fold_spearmans.append(fold_spearman)
    logger.info(
        f"Fold {fold_idx}: Spearman = {fold_spearman:.4f} "
        f"(trained {best_iter}/{model.n_estimators} estimators)"
    )

mean_spearman = np.mean(fold_spearmans)
logger.info("=" * 80)
logger.info(f"🎯 Mean per-fold Spearman: {mean_spearman:.5f}")
logger.info(f"📊 Per-fold breakdown: {[f'{s:.3f}' for s in fold_spearmans]}")
logger.info(f"📈 vs Our Ridge Baseline (0.507): {mean_spearman - 0.507:+.5f}")
logger.info(f"🏆 Actual Leader: 0.89 Spearman (Gap: {mean_spearman - 0.89:+.3f})")
logger.info("=" * 80)

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

    # p-IgGen embeddings (model already loaded and in eval mode)
    sequences = [
        f"1 {' '.join(vh)} {' '.join(vl)} 2"
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

    # Combine embeddings
    unlabeled_combined = np.concatenate(
        [w_esm1v * unlabeled_esm1v, w_piggen * unlabeled_piggen], axis=1
    )

    # Train final model on all labeled data
    logger.info("Training final LightGBM on all labeled data...")
    final_model = LGBMRegressor(**lgbm_params)
    final_model.fit(combined_embeddings, labels)
    unlabeled_predictions = final_model.predict(unlabeled_combined)
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

output_dir = Path("ginkgo_submissions_lightgbm_aggressive")
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

    # p-IgGen (model already loaded and in eval mode)
    test_sequences = [
        f"1 {' '.join(vh)} {' '.join(vl)} 2"
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

    # Combine embeddings
    test_combined = np.concatenate(
        [w_esm1v * test_esm1v, w_piggen * test_piggen], axis=1
    )

    # Train final model on all training data
    logger.info("Training final LightGBM on all training data...")
    final_model = LGBMRegressor(**lgbm_params)
    final_model.fit(combined_embeddings, labels)
    test_predictions = final_model.predict(test_combined)

    test_submission = test_df[
        ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
    ].copy()
    test_submission["PR_CHO"] = test_predictions

    test_file_out = output_dir / "ginkgo_test_predictions_PR_CHO.csv"
    test_submission.to_csv(test_file_out, index=False)
    logger.info(f"✅ Test predictions saved to: {test_file_out}")

logger.info("=" * 80)
logger.info("LIGHTGBM EXPERIMENT COMPLETE!")
logger.info("=" * 80)
logger.info(f"📁 Directory: {output_dir}/")
logger.info(f"📄 CV file: {cv_file.name} ({len(cv_submission)} antibodies)")
logger.info("📄 Test file: ginkgo_test_predictions_PR_CHO.csv")
logger.info(f"\n🎯 LightGBM CV Spearman: {mean_spearman:.5f}")
logger.info("📊 Our Ridge Baseline: 0.507")
logger.info(f"📈 Improvement vs Ridge: {mean_spearman - 0.507:+.5f}")
logger.info(f"🏆 Actual Leader: 0.89 (Gap: {mean_spearman - 0.89:+.3f})")

if mean_spearman > 0.507:
    logger.info("✅ BEATS OUR RIDGE BASELINE!")
else:
    logger.info("❌ Does not beat our Ridge baseline (0.507)")
logger.info("=" * 80)
