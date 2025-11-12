#!/usr/bin/env python3
"""
Migration script to reorganize existing models into hierarchical structure

Moves models from flat structure to hierarchical:
    OLD: models/boughter_vh_esm1v_logreg.pkl
    NEW: models/esm1v/logreg/boughter_vh_esm1v_logreg.pkl

Usage:
    python scripts/migrate_model_directories.py --dry-run  # Preview changes
    python scripts/migrate_model_directories.py            # Execute migration
"""

import argparse
import shutil
from pathlib import Path

from antibody_training_esm.core.directory_utils import (
    extract_classifier_shortname,
    extract_model_shortname,
)


def parse_model_filename(filename: str) -> dict[str, str] | None:
    """
    Parse model filename to extract components

    Examples:
        boughter_vh_esm1v_logreg.pkl -> {
            "model": "esm1v",
            "classifier": "logreg",
            "filename": "boughter_vh_esm1v_logreg.pkl"
        }

    Args:
        filename: Model filename

    Returns:
        Dictionary with model/classifier info, or None if parsing fails
    """
    # Skip non-model files
    if not filename.endswith((".pkl", ".npz", ".json")):
        return None

    # Extract model shortname from filename
    lower_name = filename.lower()

    # Determine model
    if "esm1v" in lower_name:
        model = "esm1v"
    elif "esm2_650m" in lower_name or "esm2_t33_650m" in lower_name:
        model = "esm2_650m"
    elif "esm2_3b" in lower_name or "esm2_t36_3b" in lower_name:
        model = "esm2_3b"
    elif "antiberta" in lower_name:
        model = "antiberta"
    elif "protbert" in lower_name:
        model = "protbert"
    else:
        print(f"  ⚠️  Cannot determine model from filename: {filename}")
        return None

    # Determine classifier
    if "logreg" in lower_name or "logistic" in lower_name:
        classifier = "logreg"
    elif "xgboost" in lower_name:
        classifier = "xgboost"
    elif "mlp" in lower_name:
        classifier = "mlp"
    elif "svm" in lower_name:
        classifier = "svm"
    elif "rf" in lower_name or "random_forest" in lower_name:
        classifier = "rf"
    else:
        print(f"  ⚠️  Cannot determine classifier from filename: {filename}")
        return None

    return {"model": model, "classifier": classifier, "filename": filename}


def migrate_models(models_dir: Path, dry_run: bool = False) -> None:
    """
    Migrate models to hierarchical directory structure

    Args:
        models_dir: Path to models directory
        dry_run: If True, only print planned changes without executing
    """
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating models in: {models_dir}")
    print("=" * 70)

    # Find all model files in the root directory
    model_files = [
        f
        for f in models_dir.iterdir()
        if f.is_file() and f.suffix in [".pkl", ".npz", ".json"]
    ]

    if not model_files:
        print("✅ No models to migrate (already organized or empty directory)")
        return

    migrations: list[tuple[Path, Path]] = []

    # Plan migrations
    for file_path in model_files:
        parsed = parse_model_filename(file_path.name)
        if parsed is None:
            continue

        # Generate new path
        new_dir = models_dir / parsed["model"] / parsed["classifier"]
        new_path = new_dir / parsed["filename"]

        migrations.append((file_path, new_path))

    if not migrations:
        print("✅ No models to migrate")
        return

    # Print migration plan
    print(f"\nPlanned migrations ({len(migrations)} files):")
    print("-" * 70)
    for old_path, new_path in migrations:
        print(f"  {old_path.relative_to(models_dir)}")
        print(f"    → {new_path.relative_to(models_dir)}")
        print()

    if dry_run:
        print("=" * 70)
        print("🔍 DRY RUN - No changes made")
        print(f"   Run without --dry-run to execute {len(migrations)} migrations")
        return

    # Execute migrations
    print("=" * 70)
    print("Executing migrations...")
    print("-" * 70)

    for old_path, new_path in migrations:
        # Create target directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        shutil.move(str(old_path), str(new_path))
        print(f"✅ Moved: {old_path.name} → {new_path.relative_to(models_dir)}")

    print("=" * 70)
    print(f"✅ Migration complete! Migrated {len(migrations)} files")
    print(f"   Models organized in hierarchical structure:")
    print(f"   {models_dir}/{{model}}/{{classifier}}/{{filename}}")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate models to hierarchical directory structure"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Path to models directory (default: models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )

    args = parser.parse_args()

    migrate_models(args.models_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
