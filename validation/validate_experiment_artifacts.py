#!/usr/bin/env python3
"""
Experiment Artifact Validator

Validates that experiment outputs (checkpoints, metrics) conform to
Phase 4 Pydantic schemas.

Usage:
    python validation/validate_experiment_artifacts.py <experiment_dir>
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from antibody_training_esm.models.artifact import (
    CVResults,
    ModelArtifactMetadata,
)


def validate_checkpoint(json_path: Path) -> bool:
    """Validate model config JSON."""
    print(f"Validating checkpoint: {json_path.name}...")
    try:
        with open(json_path) as f:
            data = json.load(f)
        ModelArtifactMetadata.model_validate(data)
        print("  ✓ Valid ModelArtifactMetadata")
        return True
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"  ✗ Invalid: {e}")
        return False


def validate_cv_results(yaml_path: Path) -> bool:
    """Validate CV results YAML."""
    print(f"Validating CV results: {yaml_path.name}...")
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if "cv_metrics" not in data:
            print("  ✗ Missing 'cv_metrics' key")
            return False

        CVResults.model_validate(data["cv_metrics"])
        print("  ✓ Valid CVResults")
        return True
    except (ValidationError, yaml.YAMLError) as e:
        print(f"  ✗ Invalid: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment artifacts")
    parser.add_argument(
        "experiment_dir", type=Path, help="Path to experiment directory"
    )
    args = parser.parse_args()

    if not args.experiment_dir.exists():
        print(f"Error: Directory not found: {args.experiment_dir}")
        return 1

    failures = 0

    # Find and validate checkpoints
    checkpoints = list(args.experiment_dir.rglob("*_config.json"))
    for cp in checkpoints:
        if not validate_checkpoint(cp):
            failures += 1

    # Find and validate CV results
    cv_results = list(args.experiment_dir.rglob("cv_results.yaml"))
    for cv in cv_results:
        if not validate_cv_results(cv):
            failures += 1

    if not checkpoints and not cv_results:
        print("No artifacts found to validate.")
        return 0

    if failures > 0:
        print(f"\nValidation failed with {failures} errors.")
        return 1

    print("\nAll artifacts valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
