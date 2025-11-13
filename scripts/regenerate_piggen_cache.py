"""Regenerate p-IgGen embeddings cache with correct format."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("REGENERATING p-IgGen CACHE WITH CORRECT FORMAT")
logger.info("=" * 80)

# Load data
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

logger.info(f"Loaded {len(vh_sequences)} labeled samples")

# Load p-IgGen model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
piggen_model_name = "ollieturnbull/p-IgGen"
logger.info(f"Loading p-IgGen model: {piggen_model_name}")
piggen_tokenizer = AutoTokenizer.from_pretrained(piggen_model_name)
piggen_model = AutoModelForCausalLM.from_pretrained(piggen_model_name).to(device)
piggen_model.eval()  # CRITICAL: Set to eval mode to disable dropout
logger.info("p-IgGen model loaded and set to eval mode")

# Generate embeddings with CORRECT format
logger.info(
    "Generating p-IgGen embeddings with correct format (spaces around sentinels)..."
)
sequences = [
    f"1 {' '.join(vh)} {' '.join(vl)} 2"
    for vh, vl in zip(vh_sequences, vl_sequences, strict=True)
]

logger.info(f"Example sequence format: {sequences[0][:100]}...")

embeddings_list = []
batch_size = 8
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
        embeddings_list.append(mean_pooled)

    if (i // batch_size) % 10 == 0:
        logger.info(f"Processed {i + len(batch)}/{len(sequences)} sequences")

piggen_embeddings = np.vstack(embeddings_list)
logger.info(f"Generated embeddings shape: {piggen_embeddings.shape}")

# Save to cache
cache_file = Path("embeddings_cache/ginkgo_piggen_embeddings.npy")
np.save(cache_file, piggen_embeddings)
logger.info(f"✅ Saved new p-IgGen embeddings to: {cache_file}")
logger.info(f"   File size: {cache_file.stat().st_size / 1024:.1f} KB")

logger.info("=" * 80)
logger.info("CACHE REGENERATION COMPLETE!")
logger.info("=" * 80)
logger.info("Next step: Re-run LightGBM/ElasticNet/Optimal submission scripts")
