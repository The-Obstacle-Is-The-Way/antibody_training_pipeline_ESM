# SPEC: Phase 0 — Comprehensive Audit

**Status:** DRAFT
**Parent:** [JAIN_PARITY_REMEDIATION_PLAN.md](./JAIN_PARITY_REMEDIATION_PLAN.md)
**Blocking:** All other phases

---

## Objective

Find ALL false parity claims in code and documentation before making any changes. This gives us complete visibility into the scope of the remediation.

---

## Deliverables

1. **`AUDIT_false_parity_claims.md`** — Comprehensive list of all files with issues
2. **Categorized checklist** — Organized by type and severity
3. **Effort estimate** — Time to fix each category

---

## Search Strategy

> **AUDITOR NOTE (2025-12-16):** Original commands were too `--type md/py`-restricted and missed `.github/workflows/*.yml`, `validation/baseline/*.txt/*.md5`, and `experiments/benchmarks/*.yaml`. Updated commands below broaden coverage while still avoiding scanning large data payloads.

### Search Locations

```
# Documentation
docs/**/*.md
README.md
CLAUDE.md

# Code comments
src/**/*.py
preprocessing/**/*.py
experiments/**/*.py
tests/**/*.py

# Configuration
*.yaml
*.yml
*.toml
pyproject.toml
.github/workflows/*.yml
.github/workflows/*.yaml

# Baselines / validation outputs (text)
validation/baseline/**/*.txt
validation/baseline/**/*.md5

# Benchmark outputs that are checked in (may contain stale claims)
experiments/benchmarks/**/*.yaml
experiments/benchmarks/**/*.yml
experiments/benchmarks/**/*.log

# Jupyter notebooks (if any)
**/*.ipynb
```

### Search Patterns

#### Category 1: Explicit Parity Claims

```bash
# Claims of achieving parity
rg -i "novo parity" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg -i "exact parity" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg -i "matches novo" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg -i "reproduces novo" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg -i "identical to novo" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg -i "same as novo" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
```

#### Category 2: Novo's Numbers (Claimed as Ours)

```bash
# Novo's accuracy (claimed as achieved)
rg "68\\.6%|68\\.60%|0\\.6860|0\\.686046" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"

# Novo's confusion matrix (claimed as achieved) - robust whitespace-insensitive regex
rg "\\[\\[\\s*40\\s*,\\s*17\\s*\\],\\s*\\[\\s*10\\s*,\\s*19\\s*\\]\\]" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"

# Novo's split (claimed as achieved)
rg "57 specific" --type md --type py
rg "29 non-specific" --type md --type py
rg "57\\s*/\\s*29" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
```

#### Category 3: Our Incorrect Numbers (Need Context Check)

```bash
# Our actual numbers (check if claimed as correct/parity)
rg "66\\.28%|66\\.28\\b|0\\.6628\\b" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "66\\.3" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
rg "\\[\\[\\s*40\\s*,\\s*19\\s*\\],\\s*\\[\\s*10\\s*,\\s*17\\s*\\]\\]" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"

rg "59 specific" --type md --type py
rg "27 non-specific" --type md --type py
rg "59\\s*/\\s*27" --type md --type py --type yaml --type json --glob "*.txt" --glob "*.log" --glob "*.md5"
```

#### Category 4: Methodology Claims

```bash
# Claims about preprocessing methodology
rg -i "p5e-s2" --type md --type py
rg -i "reclassif" --type md --type py
rg -i "tier a" --type md --type py
rg -i "tier b" --type md --type py
rg -i "tier c" --type md --type py
```

---

## Audit Output Format

### Template: AUDIT_false_parity_claims.md

```markdown
# Audit: False Parity Claims

**Date:** 2025-12-16
**Auditor:** [Agent Name]
**Status:** IN PROGRESS / COMPLETE

---

## Summary

| Category | Count | Fixed |
|----------|-------|-------|
| High visibility docs | X | 0 |
| Medium visibility docs | X | 0 |
| Low visibility docs | X | 0 |
| Code comments | X | 0 |
| Config files | X | 0 |
| **Total** | **X** | **0** |

---

## High Visibility (Fix First)

### File: docs/user-guide/training.md
- **Line:** 42
- **Issue:** Claims "achieves 68.6% accuracy matching Novo"
- **Fix:** Update to reflect this is the target, not achieved yet
- **Status:** [ ] Fixed

### File: README.md
- **Line:** 15
- **Issue:** States "exact reproduction of Novo results"
- **Fix:** Remove or caveat
- **Status:** [ ] Fixed

... (continue for all files)

---

## Medium Visibility

...

---

## Low Visibility

...

---

## Code Comments

...

---

## Config Files

...
```

---

## Severity Classification

### High Visibility
- README.md
- docs/index.md (landing page)
- docs/user-guide/*.md
- docs/research/novo-parity.md
- Any file a new user would read first

### Medium Visibility
- docs/developer-guide/*.md
- docs/datasets/*.md
- CLAUDE.md
- Architecture docs

### Low Visibility
- docs/archive/*.md (if not deleted)
- Code comments
- Test file comments
- Experiment scripts

---

## Context-Sensitive Evaluation

Not all mentions are false claims. Evaluate context:

### FALSE CLAIM (Fix Required)
```markdown
Our preprocessing achieves exact Novo parity with 68.6% accuracy.
```

### ACCURATE STATEMENT (No Fix)
```markdown
Novo reports 68.6% accuracy. Our current pipeline achieves 66.28%.
We are investigating the 2.32pp gap.
```

### TARGET STATEMENT (May Need Caveat)
```markdown
Target: Match Novo's 68.6% accuracy.
```

---

## Audit Checklist

- [ ] Search all documentation directories
- [ ] Search all Python source files
- [ ] Search CI/workflows (`.github/workflows/*`)
- [ ] Search baselines (`validation/baseline/*`)
- [ ] Search checked-in benchmark outputs (`experiments/benchmarks/*`)
- [ ] Confirm tests do not hardcode superseded counts (or update plan accordingly)
- [ ] Search all config files
- [ ] Search test files
- [ ] Search experiment scripts
- [ ] Categorize by severity
- [ ] Evaluate context for each match
- [ ] Create AUDIT_false_parity_claims.md
- [ ] Estimate fix effort
- [ ] Review with senior engineer

---

## Estimated Duration

- **Search execution:** 10 minutes
- **Context evaluation:** 15 minutes
- **Documentation:** 5 minutes
- **Total:** ~30 minutes

---

## Exit Criteria

Phase 0 is complete when:

1. [ ] All search patterns executed
2. [ ] All matches evaluated for context
3. [ ] AUDIT_false_parity_claims.md created
4. [ ] Checklist populated with all issues
5. [ ] Severity assigned to each issue
6. [ ] Reviewed and approved

---

**End of Spec**
