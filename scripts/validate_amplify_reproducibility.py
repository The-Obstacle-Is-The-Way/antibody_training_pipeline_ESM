#!/usr/bin/env python3
"""
AMPLIFY Reproducibility Validation Script

Compares CPU float32 embeddings (gold standard) vs MPS/CUDA embeddings to verify
that AMPLIFY's padding bug workaround (batch_size=1) produces consistent results.

Usage:
    # First, run training on both CPU and MPS:
    uv run antibody-train model=amplify_350m hardware.device=cpu training.model_name=amplify_cpu_baseline
    uv run antibody-train model=amplify_350m hardware.device=mps training.model_name=amplify_mps

    # Then validate reproducibility:
    uv run python scripts/validate_amplify_reproducibility.py

Expected Output:
    ✅ Mean absolute difference: < 1e-6 (excellent)
    ⚠️  Mean absolute difference: 1e-6 to 1e-4 (acceptable)
    ❌ Mean absolute difference: > 1e-4 (problematic)

Source: Nature Scientific Reports (2025)
        https://www.nature.com/articles/s41598-025-05674-x

Date: 2025-11-24
Author: Claude Code (Opus 4.5)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_embeddings(cache_path: Path) -> np.ndarray:
    """Load embeddings from cache file"""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)  # nosec B301 - trusted local cache file

    if "embeddings" not in cache:
        raise ValueError(f"Cache file missing 'embeddings' key: {cache_path}")

    embeddings: np.ndarray = cache["embeddings"]
    return embeddings


def find_cache_file(cache_dir: Path, pattern: str) -> Path:
    """Find cache file matching pattern (prefers newest by mtime)"""
    matches = sorted(
        cache_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )

    if not matches:
        raise FileNotFoundError(f"No cache files matching pattern: {pattern}")
    elif len(matches) > 1:
        print(f"⚠️  Multiple cache files found, using most recent: {matches[-1]}")

    return matches[-1]


def validate_reproducibility(
    cpu_cache: Path, accelerator_cache: Path, accelerator_name: str = "MPS"
) -> int:
    """
    Compare CPU embeddings (gold standard) vs accelerator embeddings.

    Args:
        cpu_cache: Path to CPU cache file
        accelerator_cache: Path to MPS/CUDA cache file
        accelerator_name: Name of accelerator for logging

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print(f"CPU cache (gold standard): {cpu_cache.name}")
    print(f"{accelerator_name} cache (validation): {accelerator_cache.name}")
    print()

    # Load embeddings
    cpu_emb = load_embeddings(cpu_cache)
    accelerator_emb = load_embeddings(accelerator_cache)

    # Compare shapes
    if cpu_emb.shape != accelerator_emb.shape:
        print("❌ ERROR: Shape mismatch!")
        print(f"   CPU: {cpu_emb.shape}")
        print(f"   {accelerator_name}: {accelerator_emb.shape}")
        return 1

    # Calculate metrics
    mae = float(np.mean(np.abs(cpu_emb - accelerator_emb)))
    max_diff = float(np.max(np.abs(cpu_emb - accelerator_emb)))
    mse = float(np.mean((cpu_emb - accelerator_emb) ** 2))

    # Report results
    print("=" * 70)
    print("AMPLIFY Reproducibility Validation")
    print("=" * 70)
    print(f"Embeddings shape:         {cpu_emb.shape}")
    print(f"Mean Absolute Error:      {mae:.2e}")
    print(f"Max Absolute Difference:  {max_diff:.2e}")
    print(f"Mean Squared Error:       {mse:.2e}")
    print()

    # Thresholds from Nature Sci Rep recommendations
    exit_code: int
    if mae < 1e-6:
        print("✅ EXCELLENT: Embeddings are nearly identical (MAE < 1e-6)")
        print(f"   {accelerator_name} is safe to use for AMPLIFY.")
        print(f"   Recommendation: Use {accelerator_name} for faster inference.")
        exit_code = 0
    elif mae < 1e-4:
        print("⚠️  ACCEPTABLE: Small differences detected (1e-6 < MAE < 1e-4)")
        print(f"   {accelerator_name} may be used but prefer CPU for critical work.")
        print(
            f"   Recommendation: Use CPU for final benchmarks, {accelerator_name} for development."
        )
        exit_code = 0
    else:
        print("❌ PROBLEMATIC: Large differences detected (MAE > 1e-4)")
        print(f"   {accelerator_name} is NOT reliable for AMPLIFY. Use CPU only.")
        print(f"   Recommendation: Do not use {accelerator_name} for AMPLIFY.")
        exit_code = 1

    print()
    print("Source: Nature Scientific Reports (2025)")
    print("https://www.nature.com/articles/s41598-025-05674-x")
    print("=" * 70)

    return exit_code


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate AMPLIFY reproducibility between CPU and accelerator"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/cache"),
        help="Directory containing embedding cache files",
    )
    parser.add_argument(
        "--cpu-pattern",
        default="*amplify*cpu*.pkl",
        help="Glob pattern for CPU cache file",
    )
    parser.add_argument(
        "--accelerator-pattern",
        default="*amplify*mps*.pkl",
        help="Glob pattern for accelerator cache file",
    )
    parser.add_argument(
        "--accelerator-name",
        default="MPS",
        help="Name of accelerator (MPS, CUDA)",
    )
    args = parser.parse_args()

    # Find cache files
    try:
        cpu_cache = find_cache_file(args.cache_dir, args.cpu_pattern)
        accelerator_cache = find_cache_file(args.cache_dir, args.accelerator_pattern)
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nRun these commands first to generate embeddings:")
        print(
            "  uv run antibody-train model=amplify_350m hardware.device=cpu training.model_name=amplify_cpu_baseline"
        )
        print(
            "  uv run antibody-train model=amplify_350m hardware.device=mps training.model_name=amplify_mps"
        )
        sys.exit(1)

    # Validate reproducibility
    exit_code = validate_reproducibility(
        cpu_cache, accelerator_cache, args.accelerator_name
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
