# Published Benchmarks

**Purpose**: Validated Novo Nordisk replication results (versioned in Git)

**Contents**:
- **novo_parity/** - EXACT 66.28% accuracy match on Jain dataset (Nov 3-5, 2025)
  - Reverse-engineered Novo's 86-antibody test set
  - Exact confusion matrix: [[40, 19], [10, 17]]
  - Methodology: P5e-S2 (PSR + AC-SINS tiebreaker)
  - See `novo_parity/README.md` for navigation

**Historical Artifacts** (removed from main branch):
- Experimental dead ends (strict_qc, hyperparameter sweeps, pre-migration results) are preserved in the `archive` branch
- Checkout `archive` branch to access: `git checkout archive`

**Versioning**: All validated results tracked in Git for reproducibility
