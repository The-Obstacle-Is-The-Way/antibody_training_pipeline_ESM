# CI/CD

**Target Audience:** Developers understanding/modifying CI pipelines

**Purpose:** Understand and work with CI/CD workflows, quality gates, and branch protection

---

## When to Use This Guide

Use this guide if you're:
- ✅ **Understanding the CI pipeline** (what runs on PRs, why builds fail)
- ✅ **Fixing CI failures** (quality gates, test failures, security scans)
- ✅ **Modifying quality gates** (adding checks, changing thresholds)
- ✅ **Setting up branch protection** (required checks, merge requirements)
- ✅ **Running CI locally** (validate before pushing)

---

## Related Documentation

- **Workflow:** [Development Workflow](development-workflow.md) - Git workflow, make commands
- **Testing:** [Testing Strategy](testing-strategy.md) - Test architecture, running tests
- **Security:** [Security Guide](security.md) - Security scanning, pickle policy
- **Docker:** [Docker Guide](docker.md) - Container builds, deployment

---

## Workflow Overview

### Active Workflows

The repository has **5 CI/CD workflows** in `.github/workflows/`:

| Workflow | File | Triggers | Duration | Purpose |
|----------|------|----------|----------|---------|
| **CI Pipeline** | `ci.yml` | Push, PR to dev/main | ~20 min | Quality gates + tests |
| **Docker CI** | `docker-ci.yml` | Push, PR | ~10 min | Verify Docker builds |
| **E2E Benchmarks** | `benchmark.yml` | Weekly (Sun 2am), Manual | ~120 min | Novo parity validation |
| **Security** | `codeql.yml` | Push to main, Weekly | ~5 min | CodeQL SAST scanning |
| **Dependencies** | `dependencies.yml` | Push, PR, Daily | ~3 min | Dependency security audit |

### Trigger Summary

**Every PR:**
- ✅ ci.yml - Quality gates, unit tests, integration tests, security
- ✅ docker-ci.yml - Docker build verification

**Weekly:**
- ✅ benchmark.yml - Full E2E suite (Sun 2am UTC)
- ✅ codeql.yml - CodeQL security scan (weekly)
- ✅ dependencies.yml - Dependency update check (daily)

**Manual:**
- ✅ benchmark.yml - Run E2E benchmarks on demand
- ✅ dependencies.yml - Check for dependency updates

---

## Quality Gate Workflow (ci.yml)

### Overview

**Primary CI pipeline** that runs on every push and PR to `dev` or `leroy-jenkins/full-send`.

**Total runtime:** ~20 minutes

**5 jobs run in parallel:**

1. **quality** - Code quality gates (ruff, mypy, bandit)
2. **test-unit** - Unit tests with coverage
3. **test-integration** - Integration tests
4. **security** - Dependency security audit
5. **ci-success** - Summary job (requires all others to pass)

### Job 1: Code Quality

**Duration:** ~3-5 minutes

**Steps:**

```yaml
- Ruff lint (uv run ruff check .)
- Ruff format check (uv run ruff format --check .)
- Mypy type checking (uv run mypy src/ --strict)
- Bandit security scan (uv run bandit -r src/)
```

**What it checks:**
- Code follows style guidelines (ruff)
- No formatting issues (ruff format)
- 100% type coverage (mypy strict mode)
- No security vulnerabilities in code (bandit)

**If it fails:**
- Run `make format` to fix formatting
- Run `make lint` to see linting errors
- Run `make typecheck` to see type errors
- Check bandit output for security issues

### Job 2: Unit Tests

**Duration:** ~5-8 minutes

**Steps:**

```yaml
- Run unit tests (pytest tests/unit/)
- Generate coverage report (--cov=src/antibody_training_esm)
- Upload coverage to Codecov
- Enforce coverage threshold (≥70%)
```

**Coverage requirement:** **≥70%** (currently at **90.80%**)

**What it checks:**
- All unit tests pass
- No test failures or errors
- Coverage doesn't drop below 70%

**If it fails:**
- Run `uv run pytest tests/unit/ -v` locally
- Check which tests failed
- Fix failing tests
- If coverage drops, add tests for uncovered code

### Job 3: Integration Tests

**Duration:** ~8-10 minutes

**Steps:**

```yaml
- Cache ESM-1v model (~2GB)
- Run integration tests (pytest tests/integration/)
- Upload test results
```

**What it checks:**
- Multi-component interactions work
- ESM model loading works
- Dataset loaders work
- End-to-end data flow works

**If it fails:**
- Run `uv run pytest tests/integration/ -v` locally
- Check integration test failures
- Verify ESM model can be loaded
- Check dataset files exist

### Job 4: Security

**Duration:** ~3-5 minutes

**Steps:**

```yaml
- Install security tools (pip-audit, safety)
- Run pip-audit (check for CVEs in dependencies)
- Run safety scan (PyPI vulnerability database)
- Upload security reports
```

**What it checks:**
- No HIGH/CRITICAL CVEs in dependencies
- Dependencies are up-to-date
- No known security vulnerabilities

**If it fails:**
- Check pip-audit.json artifact
- Update vulnerable dependencies: `uv lock --upgrade`
- Re-run tests to ensure updates don't break anything

### Job 5: CI Success Summary

**Duration:** ~10 seconds

**What it does:**
- Waits for all 4 jobs to complete
- Fails if any job failed
- Posts summary to PR

**This is the required check** for branch protection (ensures all gates passed).

---

## Docker CI Workflow (docker-ci.yml)

### Overview

**Verifies Docker builds work** without loading images (disk space constraints).

**2 jobs:**

1. **test-dev** - Build development container
2. **test-prod** - Build production container (includes model weights)

**Runtime:** ~10 minutes

### What it checks

- `Dockerfile.dev` builds successfully
- `Dockerfile.prod` builds successfully (if exists)
- No Docker build errors
- Layers cached for faster rebuilds

### GHCR Publishing

**Only on main branch:**
- Tags images with commit SHA and `latest`
- Pushes to GitHub Container Registry (ghcr.io)

### If it fails

- Check Dockerfile syntax
- Ensure base images are accessible
- Verify COPY paths exist
- Check Docker build logs in CI

---

## Security Workflows

### CodeQL (codeql.yml)

**Purpose:** Static Application Security Testing (SAST) for Python code

**Triggers:**
- Push to main branches
- Pull requests
- Weekly schedule

**Duration:** ~5 minutes

**What it checks:**
- SQL injection vulnerabilities
- Command injection
- Path traversal
- Hard-coded credentials
- Other OWASP Top 10 issues

**If it fails:**
- Review CodeQL alerts in GitHub Security tab
- Fix identified vulnerabilities
- Re-run scan

### Dependencies (dependencies.yml)

**Purpose:** Scan dependencies for known vulnerabilities

**Triggers:**
- Push to any branch
- Pull requests
- Daily schedule (6am UTC)

**Duration:** ~3 minutes

**What it checks:**
- Known CVEs in dependencies (pip-audit)
- Outdated packages with security fixes

**If it fails:**
- Check pip-audit.json artifact
- Update vulnerable dependencies: `uv lock --upgrade`
- Test that updates don't break functionality

---

## Benchmark Workflow (benchmark.yml)

### Overview

**Full E2E testing and Novo Nordisk parity validation**

**Triggers:**
- Weekly schedule (Sunday 2am UTC)
- Manual workflow dispatch
- Release tags (v*)

**Duration:** ~120 minutes (full suite with Harvey)

### What it validates

1. **Boughter (training set):**
   - 10-fold CV accuracy: 67-71%
   - Proper stratification
   - No data leakage

2. **Jain (test set):**
   - Confusion matrix: [[40,19],[10,17]] (exact match)
   - Accuracy: 66.28%
   - ELISA threshold: 0.5

3. **Shehata (PSR test set):**
   - Accuracy: ~58.8%
   - PSR threshold: 0.5495

4. **Harvey (nanobody test set):**
   - Accuracy: 61.5-61.7%
   - 141k sequences (~90 minutes)

### Why weekly, not every PR?

- Harvey alone takes 90 minutes
- Too expensive to run on every PR
- Weekly validation catches regressions

### If parity fails

- Check benchmark artifacts (confusion matrices)
- Compare with baseline results in `experiments/novo_parity/`
- Investigate changes since last passing run
- Issue created automatically on failure

---

## Branch Protection

### Protected Branches

**Main branches with protection:**
- `leroy-jenkins/full-send` (default branch)
- `main` (if exists)

### Required Checks

Before merging to protected branches, PRs must pass:

1. ✅ **Code Quality (ruff, mypy, bandit)** - ci.yml quality job
2. ✅ **Unit Tests (Python 3.12)** - ci.yml test-unit job
3. ✅ **Integration Tests (Python 3.12)** - ci.yml test-integration job
4. ✅ **Dependency Security Audit** - ci.yml security job
5. ✅ **CI Pipeline Success** - ci.yml ci-success job
6. ✅ **Test Development Container** - docker-ci.yml test-dev job
7. ✅ **Test Production Container** - docker-ci.yml test-prod job (if exists)

### Merge Requirements

- **1 approval required** - At least one reviewer must approve
- **Dismiss stale approvals** - Re-approval needed after new commits
- **Conversation resolution** - All review comments must be resolved
- **Up-to-date branches** - PR must be rebased with latest main
- **No force pushes** - Force push blocked on protected branches
- **No deletions** - Branch cannot be deleted

### Setting Up Branch Protection

**First-time setup:**

1. Go to: **Settings → Branches**
2. Click **"Add branch protection rule"**
3. Configure:

   **Branch name pattern:** `leroy-jenkins/full-send`

   **Enable:**
   - ✅ Require a pull request before merging (1 approval)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

   **Disable:**
   - ❌ Allow force pushes
   - ❌ Allow deletions

4. **Add required status checks:**
   - `Code Quality (ruff, mypy, bandit)`
   - `Unit Tests (Python 3.12)`
   - `Integration Tests (Python 3.12)`
   - `Dependency Security Audit`
   - `CI Pipeline Success`
   - `Test Development Container`

5. Click **"Create"**

**Note:** Status checks only appear after CI runs at least once. Save the rule first, then edit after first PR.

### Verifying Branch Protection

```bash
# Test 1: Try to push directly (should fail)
git checkout leroy-jenkins/full-send
git push origin leroy-jenkins/full-send
# Expected: "Protected branch update failed"

# Test 2: Create PR (should require checks)
git checkout -b test/branch-protection
git commit --allow-empty -m "test: Verify protection"
git push origin test/branch-protection
# Create PR on GitHub - should show required checks
```

---

## Local Testing

### Run Full CI Suite Locally

```bash
# Run all quality gates
make all
# Equivalent to: make format lint typecheck test

# Individual commands
make format      # Ruff format
make lint        # Ruff lint
make typecheck   # Mypy strict
make test        # Pytest all tests
make coverage    # Pytest with coverage report
```

### Run Specific Test Suites

```bash
# Unit tests only (fast)
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# E2E tests (expensive, use sparingly)
uv run pytest tests/e2e/ -v

# With coverage
uv run pytest tests/unit tests/integration \
  --cov=src/antibody_training_esm \
  --cov-report=term \
  --cov-fail-under=70
```

### Verify Before Pushing

```bash
# Full pre-push validation (recommended)
make all

# Quick validation (format + lint only)
make format lint

# Check if mypy will pass
uv run mypy src/ --strict

# Check if bandit will pass
uv run bandit -r src/
```

---

## Troubleshooting

### CI Failure: "Ruff format check failed"

**Symptom:** `ruff format --check .` fails

**Fix:**
```bash
# Auto-fix formatting
make format

# Or manually
uv run ruff format .

# Commit formatting changes
git add .
git commit -m "style: Fix ruff formatting"
```

### CI Failure: "Mypy type checking failed"

**Symptom:** `mypy src/ --strict` reports type errors

**Fix:**
```bash
# Check errors locally
uv run mypy src/ --strict

# Fix type annotations in reported files
# Common fixes:
# - Add return type annotations
# - Add parameter type annotations
# - Import types from typing module

# Verify fixed
uv run mypy src/ --strict
```

### CI Failure: "Coverage below threshold"

**Symptom:** `coverage report --fail-under=70` fails

**Fix:**
```bash
# Check current coverage
uv run pytest tests/unit tests/integration \
  --cov=src/antibody_training_esm \
  --cov-report=term-missing

# Identify uncovered lines (look for line numbers)
# Add tests for uncovered code

# Verify coverage increased
uv run coverage report
```

### CI Failure: "Bandit security scan failed"

**Symptom:** `bandit -r src/` reports HIGH/CRITICAL issues

**Fix:**
```bash
# Run bandit locally
uv run bandit -r src/ -v

# Review reported issues
# Common issues:
# - Hard-coded passwords/secrets (move to env vars)
# - Unsafe pickle usage (ensure controlled environment)
# - SQL injection (use parameterized queries)

# Fix issues and verify
uv run bandit -r src/
```

### CI Failure: "Unit tests failed"

**Symptom:** One or more unit tests failing

**Fix:**
```bash
# Run failing tests locally
uv run pytest tests/unit/ -v

# Run specific failing test
uv run pytest tests/unit/test_file.py::test_function -v

# Debug with print statements
uv run pytest tests/unit/ -v -s

# Drop into debugger on failure
uv run pytest tests/unit/ --pdb
```

### CI Failure: "Integration tests failed"

**Symptom:** Integration tests failing (often ESM model loading)

**Fix:**
```bash
# Run integration tests locally
uv run pytest tests/integration/ -v

# Check if ESM model can be downloaded
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/esm1v_t33_650M_UR90S_1')"

# Clear HuggingFace cache if corrupted
rm -rf ~/.cache/huggingface

# Re-run tests
uv run pytest tests/integration/ -v
```

### CI Failure: "pip-audit found vulnerabilities"

**Symptom:** Dependency security audit found CVEs

**Fix:**
```bash
# Check which dependencies are vulnerable
uv run pip-audit

# Update dependencies
uv lock --upgrade

# Test that updates don't break functionality
uv run pytest tests/

# Commit updated lock file
git add uv.lock
git commit -m "chore: Update dependencies (security fix)"
```

### CI Failure: "Docker build failed"

**Symptom:** Docker container build fails

**Fix:**
```bash
# Build locally to see full error
docker-compose build dev

# Common issues:
# - Missing files referenced in COPY
# - Base image not found
# - Dependency installation fails

# Check Dockerfile syntax
docker build -f Dockerfile.dev -t test-build .

# Fix issues and rebuild
docker-compose build dev
```

### CI Timeout: "Job exceeded time limit"

**Symptom:** CI job times out (>10 min for quality, >15 min for integration)

**Possible causes:**
- ESM model download very slow
- Large dataset causing long test runtime
- Infinite loop in code

**Fix:**
```bash
# Check if ESM model is cached properly
# (CI should cache ~/.cache/huggingface)

# Check test runtime locally
uv run pytest tests/ --durations=10

# Identify slow tests and optimize or mark as slow
# Mark slow tests:
# @pytest.mark.slow
# def test_expensive_operation():
#     ...
```

---

## Monitoring & Maintenance

### GitHub Actions Minutes

**Free tier:** 2000 minutes/month

**Current usage estimate:**
- CI per PR: ~20 min × 20 PR/month = **400 min**
- Weekly benchmarks: 120 min × 4 = **480 min**
- Weekly deps: 3 min × 7 = **21 min**
- **Total:** ~900 min/month ✅ **45% of limit**

**Monitor usage:**
1. Go to: **Settings → Billing and plans**
2. Check **Actions** usage
3. Alert if approaching 80% of limit

### Weekly Maintenance

**Every Monday:**
- [ ] Review dependency update PRs
- [ ] Check security audit results

**Every Sunday:**
- [ ] Review E2E benchmark results
- [ ] Verify Novo parity still holds

### Monthly Maintenance

- [ ] Review GitHub Actions minutes usage
- [ ] Check for workflow failures
- [ ] Rotate secrets if needed

---

## Advanced Topics

### Adding New Quality Gate

**Example: Add pytest-xdist for parallel tests**

1. Add to workflow:
```yaml
- name: Run tests in parallel
  run: uv run pytest tests/unit/ -n auto
```

2. Update requirements:
```bash
uv add --dev pytest-xdist
```

3. Test locally:
```bash
uv run pytest tests/unit/ -n auto
```

4. Commit workflow change

### Modifying Coverage Threshold

**Current:** ≥70%

**To increase to 75%:**

1. Edit `.github/workflows/ci.yml` line 115:
```yaml
uv run coverage report --fail-under=75
```

2. Ensure current coverage meets new threshold:
```bash
uv run pytest tests/unit tests/integration --cov=src/antibody_training_esm --cov-report=term
# Must show ≥75%
```

3. Commit change

### Running Benchmarks Manually

```bash
# Via GitHub UI:
# Actions → E2E Benchmarking & Novo Parity → Run workflow
# Select branch: main
# Run Harvey: false (for quick test)

# Via gh CLI:
gh workflow run benchmark.yml -f run_harvey=false
```

---

## Best Practices

### Before Opening PR

1. ✅ Run `make all` locally
2. ✅ Verify all tests pass
3. ✅ Check coverage didn't drop
4. ✅ Run bandit security scan
5. ✅ Commit formatting/lint fixes

### During PR Review

1. ✅ Wait for all CI checks to pass
2. ✅ Address reviewer comments
3. ✅ Resolve all conversations
4. ✅ Rebase if main has new commits

### After PR Merge

1. ✅ Delete branch
2. ✅ Monitor main branch CI
3. ✅ Check weekly benchmarks still pass

---

**Last Updated:** 2025-11-10
**Branch:** `docs/canonical-structure`
