#!/usr/bin/env python3
"""Debug script to understand where Hydra is looking for configs"""

from pathlib import Path

print("=" * 80)
print("HYDRA CONFIG PATH DIAGNOSTIC")
print("=" * 80)

# Check module location
try:
    from antibody_training_esm.core import trainer

    trainer_path = Path(trainer.__file__)
    print("\n1. Trainer module location:")
    print(f"   {trainer_path}")

    # Check relative path resolution
    conf_relative = trainer_path.parent / ".." / "conf"
    conf_resolved = conf_relative.resolve()

    print("\n2. Config path (../conf from trainer.py):")
    print(f"   Relative: {conf_relative}")
    print(f"   Resolved: {conf_resolved}")
    print(f"   Exists: {conf_resolved.exists()}")

    # Check if model configs exist
    model_dir = conf_resolved / "model"
    print("\n3. Model config directory:")
    print(f"   {model_dir}")
    print(f"   Exists: {model_dir.exists()}")

    if model_dir.exists():
        print("   Contents:")
        for yaml_file in sorted(model_dir.glob("*.yaml")):
            print(f"     - {yaml_file.name}")

    # Check if esm2_650m.yaml exists
    esm2_config = model_dir / "esm2_650m.yaml"
    print("\n4. ESM2 config file:")
    print(f"   {esm2_config}")
    print(f"   Exists: {esm2_config.exists()}")

    if esm2_config.exists():
        print("   Contents:")
        with open(esm2_config) as f:
            print("   " + f.read().replace("\n", "\n   "))

    # Check ConfigStore registrations
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    print("\n5. ConfigStore registrations:")
    print(f"   Repo: {cs.repo}")

    # Check if we can access the model group
    if "model" in cs.repo:
        print("   Model group registered: YES")
        print("   Model group entries:")
        for name in sorted(cs.repo["model"].keys()):
            print(f"     - {name}")
    else:
        print("   Model group registered: NO")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
