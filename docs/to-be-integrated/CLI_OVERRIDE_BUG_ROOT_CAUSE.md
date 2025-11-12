# CLI Override Bug - Root Cause Analysis (CORRECTED)

**Date**: 2025-11-11
**Status**: ROOT CAUSE IDENTIFIED (VERIFIED)
**Severity**: CRITICAL (blocks config group overrides in production CLI)

---

## Executive Summary

**Problem**: `antibody-train model=esm2_650m` does NOT apply the config group override - it still uses ESM-1v instead of ESM2.

**Root Cause**: ConfigStore registrations in `conf/config_schema.py` use the SAME NAMES as YAML files. When Hydra loads configs via the package module (which the console script does), the ConfigStore entries OVERRIDE the YAML, causing config group overrides to revert to structured config defaults instead of loading the override YAML.

**Impact**: ALL config group overrides (`model=X`, `classifier=Y`, `data=Z`) fail when using the `antibody-train` console script. Field overrides (`model.name=X`) still work. This is exactly what Hydra's deprecation warnings have been telling us.

---

## Symptom Details

### Observed Behavior

```bash
# FAILS - config group override ignored
antibody-train --cfg job model=esm2_650m
# Output: model.name: facebook/esm1v_t33_650M_UR90S_1 (WRONG!)

# WORKS - field override applied
antibody-train --cfg job model.name=facebook/esm2_t33_650M_UR50D
# Output: model.name: facebook/esm2_t33_650M_UR50D (CORRECT)

# WORKS - python -m invocation applies config group override
python -m antibody_training_esm.core.trainer --cfg job model=esm2_650m
# Output: model.name: facebook/esm2_t33_650M_UR50D (CORRECT)
```

### Hydra Sees But Doesn't Apply Override

```bash
antibody-train --cfg hydra model=esm2_650m | grep -A5 "task:"
# Output:
#   overrides:
#     hydra: []
#     task:
#     - model=esm2_650m  # ✅ Hydra SEES the override
```

Yet the composed config still shows ESM-1v! This means Hydra:

1. ✅ Parses the command-line argument
2. ✅ Registers it as a task override
3. ❌ **FAILS to apply it** because ConfigStore entries override YAML

---

## Investigation Steps

### Step 1: Reproduce With Hydra Compose API (Package Module)

The exploration agent tested Hydra's `initialize_config_module` to reproduce the CLI behavior:

```python
from hydra import initialize_config_module, compose

with initialize_config_module(config_module="antibody_training_esm.conf"):
    cfg = compose(config_name="config", overrides=["model=esm2_650m"])
    print(cfg.model.name)
```

**Result**: Prints `facebook/esm1v_t33_650M_UR90S_1` (WRONG!)

This **exactly reproduces the CLI bug** when using package-based config loading.

### Step 2: Reproduce With Hydra Compose API (Filesystem Directory)

The exploration agent then tested `initialize_config_dir` with an absolute path:

```python
from pathlib import Path
from hydra import initialize_config_dir, compose

config_dir = Path("src/antibody_training_esm/conf").resolve()
with initialize_config_dir(config_dir=str(config_dir)):
    cfg = compose(config_name="config", overrides=["model=esm2_650m"])
    print(cfg.model.name)
```

**Result**: Prints `facebook/esm2_t33_650M_UR50D` (CORRECT!)

This proves the bug is **specific to package-based config loading**.

### Step 3: Prove ConfigStore Import Breaks Filesystem Loading

The exploration agent then imported the conf module BEFORE using filesystem loading:

```python
import antibody_training_esm.conf  # Registers ConfigStore

from pathlib import Path
from hydra import initialize_config_dir, compose

config_dir = Path("src/antibody_training_esm/conf").resolve()
with initialize_config_dir(config_dir=str(config_dir)):
    cfg = compose(config_name="config", overrides=["model=esm2_650m"])
    print(cfg.model.name)
```

**Result**: Prints `facebook/esm1v_t33_650M_UR90S_1` (WRONG AGAIN!)

This proves that **importing the conf module activates ConfigStore registrations that override YAML**, even when using filesystem-based loading.

### Step 4: Check Hydra Config Sources

Looking at the Hydra config output from both invocations:

**Console script** (`antibody-train --cfg hydra`):
```yaml
config_sources:
  - schema: pkg
    provider: antibody_training_esm.conf  # ← PACKAGE MODULE
```

**Python -m** (`python -m antibody_training_esm.core.trainer --cfg hydra`):
```yaml
config_sources:
  - schema: file
    provider: /Users/ray/.../src/antibody_training_esm/conf  # ← FILESYSTEM PATH
```

The console script uses **package module** loading, which imports `antibody_training_esm.conf` and activates the ConfigStore registrations. The `python -m` invocation uses **filesystem** loading, which never imports the conf module.

### Step 5: Examine ConfigStore Registrations

Looking at `src/antibody_training_esm/conf/config_schema.py` lines 127-135:

```python
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="esm1v", node=ModelConfig)
cs.store(group="classifier", name="logreg", node=ClassifierConfig)
cs.store(group="data", name="boughter_jain", node=DataConfig)
```

These registrations use the **SAME NAMES** as the YAML files:
- ConfigStore: `name="config"` → YAML: `conf/config.yaml`
- ConfigStore: `group="model", name="esm1v"` → YAML: `conf/model/esm1v.yaml`

This is the **deprecated "automatic schema matching"** that Hydra warns about!

### Step 6: Connect to Deprecation Warnings

Every CLI run shows these warnings:

```
'config' is validated against ConfigStore schema with the same name.
This behavior is deprecated in Hydra 1.1 and will be removed in Hydra 1.2.
See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/automatic_schema_matching for migration instructions.

'model/esm1v' is validated against ConfigStore schema with the same name.
...
```

These warnings are **literally telling us the problem**: When ConfigStore entries have the same names as YAML files, Hydra 1.1+ behavior causes the structured configs to override the YAML content.

---

## Root Cause Explanation

### The Problem

Hydra has two ways to load configs:

1. **Filesystem loading** (`config_path="../conf"`): Loads YAML files directly from disk
2. **Package loading** (when `config_path` resolves to a Python package): Imports the package module, then loads configs

When the console script runs, Hydra resolves `@hydra.main(config_path="../conf")` relative to the installed package, which points to the `antibody_training_esm.conf` Python module. This triggers:

1. Import `antibody_training_esm.conf`
2. Run `antibody_training_esm/conf/__init__.py`
3. Import `antibody_training_esm/conf/config_schema.py`
4. Execute ConfigStore registrations
5. ConfigStore entries with the **same names as YAML** become active
6. Hydra prefers ConfigStore over YAML (deprecated behavior)
7. Config group overrides fail because Hydra loads the structured config instead of the override YAML

### Why Python -m Works But Console Script Doesn't

**When you run `python -m antibody_training_esm.core.trainer`:**

1. Hydra decorator: `@hydra.main(config_path="../conf")`
2. Resolves relative to source file: `/Users/ray/.../src/antibody_training_esm/conf`
3. This is a **filesystem directory**, not a Python module
4. Hydra loads pure YAML files
5. ConfigStore never gets imported or activated
6. Config group overrides work ✅

**When you run `antibody-train` console script:**

1. Hydra decorator: `@hydra.main(config_path="../conf")`
2. Resolves relative to installed package: `antibody_training_esm.conf`
3. This is a **Python module**, so Hydra imports it
4. Import triggers ConfigStore registrations
5. ConfigStore entries override YAML content
6. Config group overrides fail ❌

### Why Field Overrides Still Work

Field overrides (`model.name=X`) work because they directly modify config values **AFTER** Hydra composes the config. They don't depend on which config source (YAML vs ConfigStore) was used.

Config group overrides (`model=X`) fail because Hydra needs to:

1. Look up the `X` config in the `model` group
2. Load the corresponding config (YAML or ConfigStore)
3. Merge it with defaults

When ConfigStore entries exist with the same names, Hydra prefers them over YAML, so `model=esm2_650m` loads the ConfigStore `ModelConfig` dataclass (with ESM-1v defaults) instead of the `conf/model/esm2_650m.yaml` file.

---

## The Fix

### Option 1: Remove ConfigStore Registrations (RECOMMENDED)

Delete or comment out all ConfigStore registrations in `src/antibody_training_esm/conf/config_schema.py`:

```python
# REMOVE OR COMMENT OUT:
# cs = ConfigStore.instance()
# cs.store(name="config", node=Config)
# cs.store(group="model", name="esm1v", node=ModelConfig)
# cs.store(group="classifier", name="logreg", node=ClassifierConfig)
# cs.store(group="data", name="boughter_jain", node=DataConfig)
```

This eliminates the conflict entirely and makes both invocation methods use pure YAML.

**Pros:**
- Minimal change (comment out 5 lines)
- Solves the immediate problem
- Eliminates deprecation warnings
- Works for both `python -m` and console script
- Aligns with Hydra 1.2+ behavior

**Cons:**
- Loses structured config type validation (if we were using it)
- Loses dataclass defaults (but we have YAML defaults instead)

### Option 2: Rename ConfigStore Registrations

Rename the ConfigStore entries to NOT conflict with YAML files:

```python
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)  # Was: "config"
cs.store(group="schema/model", name="base", node=ModelConfig)  # Was: "model", "esm1v"
cs.store(group="schema/classifier", name="base", node=ClassifierConfig)
cs.store(group="schema/data", name="base", node=DataConfig)
```

Then explicitly reference them in YAML for validation only:

```yaml
# conf/config.yaml
defaults:
  - _self_
  - model: esm1v
  - classifier: logreg
  - data: boughter_jain
  - override hydra/job_logging: colorlog

# @package _global_
_target_: antibody_training_esm.conf.config_schema.Config  # Validation only
```

**Pros:**
- Keeps type validation if needed
- Cleaner separation of schemas vs configs

**Cons:**
- More complex
- Requires YAML changes
- Still need to be careful about naming

### Option 3: Use Only ConfigStore (NOT RECOMMENDED)

Delete all YAML files and use only ConfigStore for everything.

**Pros:**
- No naming conflicts
- Pure Python config

**Cons:**
- Major refactor required
- Loses Hydra's composability features
- Makes config changes harder (requires code changes)

---

## Recommended Fix

**Use Option 1** - Remove/comment out ConfigStore registrations in `config_schema.py`.

This is the minimal, surgical fix that:
- Solves the immediate problem
- Eliminates deprecation warnings
- Works for all invocation methods
- Requires no YAML changes
- Can be implemented in ~5 lines (comment out registrations)

We can keep the dataclasses for type hints and validation in the code, but stop registering them with ConfigStore.

---

## Validation Plan

After implementing the fix:

1. **Test console script with config group override:**
   ```bash
   antibody-train --cfg job model=esm2_650m | grep "model.name"
   # Should show: facebook/esm2_t33_650M_UR50D
   ```

2. **Test python -m still works:**
   ```bash
   python -m antibody_training_esm.core.trainer --cfg job model=esm2_650m | grep "model.name"
   # Should show: facebook/esm2_t33_650M_UR50D
   ```

3. **Verify no deprecation warnings:**
   ```bash
   antibody-train --cfg job model=esm2_650m 2>&1 | grep -i "deprecated"
   # Should show: (no output)
   ```

4. **Test with Hydra Compose API (package module):**
   ```python
   from hydra import initialize_config_module, compose
   with initialize_config_module(config_module="antibody_training_esm.conf"):
       cfg = compose(config_name="config", overrides=["model=esm2_650m"])
       print(cfg.model.name)
   # Should show: facebook/esm2_t33_650M_UR50D
   ```

5. **Run actual training:**
   ```bash
   antibody-train model=esm2_650m classifier=logreg
   # Should use ESM2 embeddings
   ```

6. **Check saved config:**
   ```bash
   cat outputs/*/latest/.hydra/config.yaml | grep "model.name"
   # Should show: facebook/esm2_t33_650M_UR50D
   ```

---

## Related Issues

- **Hydra Deprecation Warnings**: These warnings were DIRECTLY RELATED to this bug. They were telling us that ConfigStore registrations with the same names as YAML files cause conflicts. Removing the registrations fixes both the bug AND the warnings.

- **Test Output Hierarchy**: Unrelated to this bug. That implementation is complete.

- **Embedding Cache**: Unrelated to this bug. That fix is complete.

---

## Lessons Learned

1. **Listen to deprecation warnings** - The Hydra 1.1 warnings were literally telling us the problem. "Automatic schema matching" means ConfigStore entries with the same names as YAML files, and this behavior causes overrides to fail.

2. **Hydra package vs filesystem loading** - `@hydra.main(config_path="../conf")` behaves differently depending on whether it resolves to a filesystem directory or a Python module. Package loading imports the module and activates ConfigStore.

3. **ConfigStore overrides YAML** - When both exist with the same name, ConfigStore takes precedence in Hydra 1.1+, which breaks config group overrides.

4. **Test both invocation methods** - Always test both `python -m` and console script entry points, as they resolve config paths differently.

---

## References

- **Hydra Automatic Schema Matching Upgrade Guide**: https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/automatic_schema_matching
- **Hydra ConfigStore Docs**: https://hydra.cc/docs/tutorials/structured_config/config_store/
- **Hydra Config Path Resolution**: https://hydra.cc/docs/advanced/search_path/

---

## Acknowledgments

This corrected root cause analysis was derived from systematic reproduction testing by the exploration agent. The original analysis incorrectly identified a "missing import" problem, which would have made the bug worse by breaking `python -m` invocation as well. The correct fix is to remove ConfigStore registrations that conflict with YAML files.
