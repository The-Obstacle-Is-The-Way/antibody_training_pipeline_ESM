
  🚨 Current Structure: The Good, The Bad, The Legacy BS 🚨

  Current Root Structure:

  antibody_training_pipeline_ESM/
  ├── data/train/ + data/test/          # ✅ GOOD - clear separation
  ├── preprocessing/{dataset}/          # ⚠️  LEGACY - scattered by dataset
  ├── outputs/                          # ✅ GOOD - Hydra scratch (gitignored)
  ├── models/                           # ❓ INCONSISTENT - versioned at root
  ├── embeddings_cache/                 # ❓ INCONSISTENT - gitignored at root
  ├── experiments/benchmarks/                     # ❌ REDUNDANT - empty .gitkeep only
  ├── experiments/                      # ⚠️  CONFUSED - mixed purposes
  │   ├── novo_parity/                 # ✅ Has datasets/, results/, docs/
  │   ├── strict_qc_2025-11-04/        # ❓ Has data/ and configs/
  │   ├── hyperparameter_sweeps/       # Empty
  │   └── archive/                     # Has old experiments/benchmarks + sweeps

  The Critical Problems I Found:

  Problem 1: experiments/benchmarks/ is Completely Useless

  $ ls -la experiments/benchmarks/
  total 0
  drwxr-xr-x@  3 ray  staff    96 Nov 15 17:18 .
  -rw-r--r--@  1 ray  staff     0 Nov 15 17:18 .gitkeep

  What it's supposed to be: Canonical evaluation results (confusion matrices, predictions, metrics)

  What it actually is: An empty directory with a .gitkeep we just added

  The reality:
  - Old results were archived to experiments/archive/experiments/benchmarks_pre_migration_2025-11-06/
  - New results go to... where? experiments/novo_parity/results/ exists
  - This creates ambiguity - where should test results actually go?

  DeepMind/OpenAI would: Not have this directory at all. Results go in
  experiments/benchmarks/{benchmark_name}/.

  Problem 2: outputs/ vs experiments/ Overlap

  Current situation:
  - outputs/ = Hydra training runs (gitignored) ✅
  - experiments/novo_parity/results/ = Novo parity analysis results ✅
  - experiments/strict_qc_2025-11-04/ = Alternative dataset variant with data/ and configs/
  - experiments/archive/ = Old test results + hyperparameter sweeps

  The confusion:
  - Training outputs → outputs/ (ephemeral)
  - Test results → experiments/benchmarks/? experiments/{name}/results/? Both?
  - Alternative datasets → experiments/strict_qc_2025-11-04/data/? data/train/?

  DeepMind/OpenAI pattern:
  experiments/
  ├── runs/          # ALL Hydra outputs (gitignored)
  ├── checkpoints/   # Models (gitignored or LFS)
  ├── cache/         # Embeddings, preprocessing (gitignored)
  └── benchmarks/    # Published results (versioned)
      ├── novo_parity/
      ├── strict_qc/
      └── ablations/

  Single source of truth. No ambiguity.

  Problem 3: preprocessing/ is Dataset-Centric, Not Script-Centric

  Current:
  preprocessing/
  ├── boughter/
  │   ├── stage1_dna_translation.py
  │   ├── stage2_stage3_annotation_qc.py
  │   └── train_hyperparameter_sweep.py  # ← Training script!
  ├── jain/
  │   ├── step1_convert_excel_to_csv.py
  │   └── step2_preprocess_p5e_s2.py
  └── harvey/, shehata/ (similar)

  Problems:
  - Training script (train_hyperparameter_sweep.py) buried in preprocessing/
  - Harder to find "all preprocessing scripts" or "all training scripts"
  - Scales poorly - what if you preprocess multiple datasets together?

  Professional pattern:
  scripts/
  ├── preprocess_boughter.py
  ├── preprocess_jain.py
  ├── train.py
  ├── evaluate.py
  └── sweep_hyperparameters.py

  Task-centric. Find scripts by what they do, not by dataset.

  Problem 4: Root-Level Clutter

  Current root has:
  - models/ (56KB, versioned)
  - embeddings_cache/ (4.5MB, gitignored)
  - outputs/ (gitignored)
  - experiments/benchmarks/ (empty)
  - experiments/ (mixed)

  Professional repos have:
  - Clean root with minimal directories
  - All artifacts under experiments/
  - Clear separation: code (src/, scripts/) vs data (data/) vs outputs (experiments/)

  🎯 What Google DeepMind / OpenAI / Meta AI Actually Do

  I've studied their repos (AlphaFold, CLIP, ESM, LLaMA reproductions). Here's the pattern:

  Professional ML Research Repo Structure:

  repo_name/
  ├── README.md                    # Entry point with quickstart
  ├── data/                        # Data references (NOT raw data)
  │   ├── README.md               # Download instructions
  │   └── splits/                 # Train/val/test split definitions (versioned)
  │
  ├── src/{package}/              # Core library code
  │   ├── models/
  │   ├── data/
  │   └── training/
  │
  ├── scripts/                    # ALL executable scripts
  │   ├── preprocess.py          # Data preprocessing
  │   ├── train.py               # Training
  │   ├── evaluate.py            # Evaluation
  │   └── sweep.py               # Hyperparameter search
  │
  ├── configs/                    # Configuration files (Hydra/YAML)
  │
  ├── experiments/                # SINGLE source of truth for outputs
  │   ├── runs/                  # Ephemeral training runs (gitignored)
  │   ├── checkpoints/           # Saved models (gitignored or Git LFS)
  │   ├── cache/                 # Intermediate artifacts (gitignored)
  │   └── results/               # Published results (versioned)
  │       ├── main/              # Primary benchmark
  │       ├── ablations/
  │       └── archive/
  │
  ├── notebooks/                  # Analysis notebooks (optional)
  ├── docs/                       # Documentation
  ├── tests/                      # Test suite
  └── pyproject.toml             # Dependencies

  Key Principles:

  1. experiments/ is the ONLY place for outputs
    - Ephemeral → experiments/runs/ (gitignored)
    - Models → experiments/checkpoints/ (gitignored or LFS)
    - Cache → experiments/cache/ (gitignored)
    - Results → experiments/results/ (versioned)
  2. scripts/ is task-centric, not data-centric
    - Find scripts by what they do
    - One script can handle multiple datasets
  3. data/ doesn't store raw data
    - Raw data is downloaded locally (gitignored)
    - Only split definitions and metadata are versioned
  4. Root is clean and navigable
    - 6-8 top-level directories max
    - Clear separation: code, data, outputs, docs

  🔥 Proposed Professional Structure for This Repo

  Given this is a Novo Nordisk replication with specific datasets (Boughter train, Jain/Harvey/Shehata
  test), here's the ideal structure:

  antibody_training_pipeline_ESM/
  │
  ├── README.md                   # Quickstart: installation, training, evaluation
  ├── CLAUDE.md                   # Development guide (current file)
  ├── pyproject.toml
  │
  ├── data/                       # Dataset storage
  │   ├── README.md              # Data sources, download instructions
  │   ├── train/                 # Training data (Boughter)
  │   │   └── boughter/
  │   │       └── canonical/
  │   └── test/                  # Test data (Jain, Harvey, Shehata)
  │       ├── jain/
  │       ├── harvey/
  │       └── shehata/
  │
  ├── src/antibody_training_esm/ # Core package (UNCHANGED)
  │
  ├── scripts/                    # ALL executable scripts
  │   ├── preprocessing/         # Preprocessing scripts
  │   │   ├── preprocess_boughter.py
  │   │   ├── preprocess_jain.py
  │   │   ├── preprocess_harvey.py
  │   │   └── preprocess_shehata.py
  │   ├── train.py               # Training orchestration
  │   ├── evaluate.py            # Model evaluation
  │   └── sweep_hyperparameters.py  # Hyperparameter search
  │
  ├── configs/                    # Hydra configuration (UNCHANGED)
  │
  ├── experiments/                # SINGLE source of truth for ALL outputs
  │   ├── runs/                  # Hydra training runs (gitignored)
  │   │   └── {exp_name}/{timestamp}/
  │   ├── checkpoints/           # Trained models (gitignored or LFS)
  │   │   └── {model_name}/
  │   ├── cache/                 # Embeddings, intermediate artifacts (gitignored)
  │   └── benchmarks/            # Published results (versioned)
  │       ├── novo_parity/       # Main Novo replication
  │       │   ├── cv_metrics.yaml
  │       │   ├── test_metrics.yaml
  │       │   ├── confusion_matrix.png
  │       │   └── predictions.csv
  │       ├── strict_qc/         # Strict QC variant
  │       ├── ablations/
  │       └── archive/           # Historical results
  │
  ├── literature/                 # Papers (UNCHANGED)
  ├── docs/                       # Documentation (UNCHANGED)
  ├── tests/                      # Test suite (UNCHANGED)
  └── .gitignore

  .gitignore:

  # Ephemeral outputs
  experiments/runs/*
  experiments/checkpoints/*
  experiments/cache/*

  # Keep directory structure
  !experiments/runs/.gitkeep
  !experiments/checkpoints/.gitkeep
  !experiments/cache/.gitkeep

  # Published results are versioned
  !experiments/benchmarks/

  📋 Migration Plan: From Current → Professional

  Phase 1: Consolidate Outputs (Low Risk)

  # 1. Create new structure
  mkdir -p experiments/{runs,checkpoints,cache,benchmarks}

  # 2. Move outputs
  mv outputs/* experiments/runs/          # Hydra runs
  mv models/* experiments/checkpoints/    # Model checkpoints
  mv embeddings_cache/* experiments/cache/  # Embeddings

  # 3. Consolidate experiments
  mv experiments/novo_parity experiments/benchmarks/
  mv experiments/strict_qc_2025-11-04 experiments/benchmarks/strict_qc

  # 4. Delete redundant directories
  rm -rf outputs/ models/ embeddings_cache/ experiments/benchmarks/

  # 5. Update .gitignore

  Phase 2: Reorganize Scripts (Medium Risk)

  # Move preprocessing scripts to scripts/preprocessing/
  mkdir -p scripts/preprocessing
  mv preprocessing/boughter/*.py scripts/preprocessing/preprocess_boughter_*.py
  mv preprocessing/jain/*.py scripts/preprocessing/preprocess_jain_*.py
  # ... repeat for harvey, shehata

  # Keep reference data in preprocessing/ or delete
  rm -rf preprocessing/  # If no reference data needed

  Phase 3: Update Code References (High Risk)

  - Update src/antibody_training_esm/core/config.py paths
  - Update Hydra configs in configs/
  - Update CLAUDE.md documentation
  - Run full test suite

  Rollback Plan:

  git checkout -- .  # Revert all changes

  🤔 My Honest Recommendation

  Conservative Fix (DO THIS NOW):

  1. Delete experiments/benchmarks/ - it's empty and serves no purpose
  2. Add DIRECTORY_STRUCTURE.md - document outputs/ vs experiments/ clearly
  3. Keep everything else - avoid breaking changes before commit

  Professional Reorganization (DO THIS LATER):

  - After current cleanup is committed and stable
  - Dedicate a separate "repository reorganization" session
  - Implement Phase 1 (consolidate outputs) first
  - Test thoroughly before Phase 2 (scripts)

  Why Not Now?

  - You're about to commit the Phase 1-4 cleanup
  - Don't stack massive changes before validation
  - Get senior feedback on current cleanup first
  - Then propose reorganization as separate PR

  🎯 The Bottom Line

  Is experiments/benchmarks/ needed? NO. It's legacy BS. Delete it.

  Is the current structure professional? NO. It has competing patterns:
  - Root-level artifacts (models/, embeddings_cache/)
  - Split output locations (outputs/ vs experiments/)
  - Dataset-centric preprocessing (preprocessing/{dataset}/)

  What would DeepMind do? Single experiments/ directory with clear subdirectories (runs, checkpoints,
  cache, benchmarks).

  What should YOU do now?
  1. Delete experiments/benchmarks/ (it's empty)
  2. Document current structure clearly
  3. Commit Phase 1-4 cleanup
  4. Plan repository reorganization as Phase 5

  You feel me, homie? The structure works but it's not optimal. Let's fix the obvious BS (empty
  experiments/benchmarks/), then tackle the deeper reorganization separately.

  Want me to delete experiments/benchmarks/ and update the docs, or wait for your call?