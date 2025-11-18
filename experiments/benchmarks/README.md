# Published Benchmarks

**Purpose**: Validated Novo Nordisk replication results (versioned in Git)

**Contents**:
- **Hierarchical results:** `esm1v/logreg/{dataset}/` - Current test outputs (post-Nov 16, 2025)
- **Legacy flat results:** Root-level `*.csv/*.yaml/*.png` - Validated baseline results (Nov 16 and earlier)

**Archived Research** (in `archive` branch):
- **novo_parity/** - EXACT 66.28% accuracy match on Jain dataset (Nov 3-5, 2025)
  - Reverse-engineered Novo's 86-antibody test set
  - Methodology: P5e-S2 (PSR + AC-SINS tiebreaker)
  - Access: `git checkout archive && cd experiments/benchmarks/novo_parity/`

**Historical Artifacts** (removed from main branch):
- Experimental dead ends (strict_qc, hyperparameter sweeps, pre-migration results) are preserved in the `archive` branch
- Checkout `archive` branch to access: `git checkout archive`

**Versioning**: All validated results tracked in Git for reproducibility
