#!/usr/bin/env python3
"""
AMPLIFY Reproducibility Validation Script

Compares CPU float32 embeddings (gold standard) vs MPS embeddings to verify
that AMPLIFY's padding bug workaround (batch_size=1) produces consistent results.

Usage:
    uv run python scripts/validate_amplify_reproducibility.py

Expected Output:
    ✅ Mean absolute difference: < 1e-6 (excellent)
    ⚠️  Mean absolute difference: 1e-6 to 1e-4 (acceptable)
    ❌ Mean absolute difference: > 1e-4 (problematic)

Source: https://www.nature.com/articles/s41598-025-05674-x
Date: 2025-11-23
"""

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_embeddings(cache_path: Path) -> np.ndarray[Any, Any]:
    """Load embeddings from cache file"""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    if "embeddings" not in cache:
        raise ValueError(f"Cache file missing 'embeddings' key: {cache_path}")

    return cache["embeddings"]  # type: ignore[no-any-return]


def find_cache_file(cache_dir: Path, pattern: str) -> Path:
    """Find cache file matching pattern (prefers newest by mtime)"""
    matches = sorted(
        cache_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )

    if not matches:
        raise FileNotFoundError(f"No cache files matching pattern: {pattern}")
    elif len(matches) > 1:
        # Log still refers to matches[-1], now guaranteed to be newest by mtime
        print(f"⚠️  Multiple cache files found, using most recent: {matches[-1]}")

    return matches[-1]


def main() -> None:
    # Find cache files
    cache_dir = Path("experiments/cache")

    try:
        cpu_cache = find_cache_file(cache_dir, "*amplify*cpu*.pkl")
        mps_cache = find_cache_file(cache_dir, "*amplify*mps*.pkl")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nRun these first:")
        print("  uv run antibody-train model=amplify_350m hardware.device=cpu")
        print("  uv run antibody-train model=amplify_350m hardware.device=mps")
        sys.exit(1)

    print(f"CPU cache (gold standard): {cpu_cache.name}")
    print(f"MPS cache (validation):    {mps_cache.name}")
    print()

    # Load embeddings
    cpu_emb = load_embeddings(cpu_cache)
    mps_emb = load_embeddings(mps_cache)

    # Compare shapes
    if cpu_emb.shape != mps_emb.shape:
        print("❌ ERROR: Shape mismatch!")
        print(f"   CPU: {cpu_emb.shape}")
        print(f"   MPS: {mps_emb.shape}")
        sys.exit(1)

    # Calculate metrics
    mae = np.mean(np.abs(cpu_emb - mps_emb))
    max_diff = np.max(np.abs(cpu_emb - mps_emb))
    mse = np.mean((cpu_emb - mps_emb) ** 2)

    # Report results
    print(f"{'=' * 70}")
    print("AMPLIFY Reproducibility Validation")
    print(f"{'=' * 70}")
    print(f"Embeddings shape:         {cpu_emb.shape}")
    print(f"Mean Absolute Error:      {mae:.2e}")
    print(f"Max Absolute Difference:  {max_diff:.2e}")
    print(f"Mean Squared Error:       {mse:.2e}")
    print()

    # Thresholds from Nature Sci Rep recommendations
    if mae < 1e-6:
        print("✅ EXCELLENT: Embeddings are nearly identical (MAE < 1e-6)")
        print("   MPS is safe to use for AMPLIFY.")
        print("   Recommendation: Use MPS for faster inference.")
        exit_code = 0
    elif mae < 1e-4:
        print("⚠️  ACCEPTABLE: Small differences detected (1e-6 < MAE < 1e-4)")
        print("   MPS may be used but prefer CPU for critical work.")
        print("   Recommendation: Use CPU for final benchmarks, MPS for development.")
        exit_code = 0
    else:
        print("❌ PROBLEMATIC: Large differences detected (MAE > 1e-4)")
        print("   MPS is NOT reliable for AMPLIFY. Use CPU only.")
        print("   Recommendation: Do not use MPS for AMPLIFY.")
        exit_code = 1

    print()
    print("Source: Nature Scientific Reports (2025)")
    print("https://www.nature.com/articles/s41598-025-05674-x")
    print(f"{'=' * 70}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
