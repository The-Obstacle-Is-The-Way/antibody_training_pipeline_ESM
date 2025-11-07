# GitHub Actions CI/CD Specification - Production Grade

**Document Status:** Draft for Senior Review
**Repository:** antibody_training_pipeline_ESM
**Date:** 2025-11-07
**Author:** Engineering Team
**Purpose:** Define comprehensive CI/CD pipeline for professional ML/bioinformatics repository

---

## Table of Contents

1. [Current State](#current-state)
2. [Target State - Complete CI/CD](#target-state)
3. [Workflow Specifications](#workflow-specifications)
4. [Implementation Phases](#implementation-phases)
5. [Success Metrics](#success-metrics)
6. [Senior Review Checklist](#senior-review-checklist)

---

## Current State

### ✅ What We Have Now

**Workflows:**
- `.github/workflows/docker-ci.yml` - Docker build and test (basic)

**Coverage:**
- ✅ Docker dev container build
- ✅ Docker prod container build
- ✅ Basic pytest execution in containers
- ⚠️ Build and Push job (skipped - only on main)

**Limitations:**
- No Python environment CI (only Docker)
- No code quality gates
- No security scanning
- No automated releases
- No performance benchmarks
- No deployment automation
- Limited test reporting
- No dependency updates
- No documentation builds

---

## Target State

### 🎯 Complete Professional CI/CD Pipeline

A world-class ML/bioinformatics repository should have:

1. **Multi-Environment Testing** (Python 3.11, 3.12)
2. **Comprehensive Quality Gates** (linting, formatting, type checking, security)
3. **Full Test Suite Execution** (unit, integration, e2e with reporting)
4. **Docker Multi-Architecture Builds** (amd64, arm64)
5. **Automated Dependency Management** (security updates, version bumps)
6. **Performance Benchmarking** (track model performance over time)
7. **Automated Releases** (semantic versioning, changelog generation)
8. **Documentation Deployment** (auto-publish docs to GitHub Pages)
9. **Security Scanning** (SAST, dependency vulnerabilities, Docker image scanning)
10. **Model Registry Integration** (HuggingFace, W&B)
11. **Reproducibility Validation** (Novo parity checks on every PR)
12. **Deployment Automation** (staging + production environments)

---

## Workflow Specifications

### 1. **Main CI Pipeline** (`.github/workflows/ci.yml`)

**Trigger:** Every push, every PR

**Jobs:**

#### Job 1: Code Quality (`quality`)
- **Runs on:** `ubuntu-latest`
- **Python versions:** 3.11, 3.12 (matrix)
- **Steps:**
  1. Checkout code
  2. Set up Python (with caching)
  3. Install dependencies (`uv sync`)
  4. **Ruff lint** (fail on any errors)
  5. **Ruff format check** (fail on formatting issues)
  6. **mypy** type checking (strict mode, fail on any errors)
  7. **bandit** security scan (SAST - Static Application Security Testing)
  8. Upload results as artifacts

#### Job 2: Unit Tests (`test-unit`)
- **Runs on:** `ubuntu-latest`
- **Python versions:** 3.11, 3.12 (matrix)
- **Steps:**
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Run pytest unit tests (`tests/unit/`)
  5. Generate coverage report (minimum 80% coverage)
  6. Upload coverage to Codecov
  7. Upload test results (JUnit XML)

#### Job 3: Integration Tests (`test-integration`)
- **Runs on:** `ubuntu-latest`
- **Python versions:** 3.12 only
- **Steps:**
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Run pytest integration tests (`tests/integration/`)
  5. Upload test results

#### Job 4: E2E Tests (`test-e2e`)
- **Runs on:** `ubuntu-latest`
- **Python versions:** 3.12 only
- **Conditional:** Only on PR to main, or manual trigger
- **Steps:**
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Download ESM-1v model (cache this!)
  5. Run E2E tests (`tests/e2e/`)
  6. Validate Novo parity on all benchmarks:
     - Boughter 10-fold CV: 67-71%
     - Jain: 66.28% exact
     - Shehata: 58.8% (with threshold calibration)
     - Harvey: 61.5-61.7%
  7. Upload confusion matrices as artifacts
  8. Comment results on PR

#### Job 5: Dependency Security (`security`)
- **Runs on:** `ubuntu-latest`
- **Steps:**
  1. Checkout code
  2. **pip-audit** (check for known vulnerabilities)
  3. **Safety** check (PyPI vulnerability database)
  4. **Trivy** scan on dependencies
  5. Fail if HIGH or CRITICAL vulnerabilities found

---

### 2. **Docker CI/CD Pipeline** (`.github/workflows/docker.yml`)

**Trigger:** Every push, every PR, manual dispatch

**Jobs:**

#### Job 1: Docker Lint (`docker-lint`)
- **Steps:**
  1. Checkout code
  2. **hadolint** on Dockerfile.dev and Dockerfile.prod
  3. Fail on errors

#### Job 2: Build Dev Container (`build-dev`)
- **Runs on:** `ubuntu-latest`
- **Steps:**
  1. Free up disk space (remove .NET, Haskell, etc.)
  2. Set up Docker Buildx
  3. Build dev container
  4. Run tests inside container (358 tests)
  5. Cache layers (GitHub Actions cache)

#### Job 3: Build Prod Container (`build-prod`)
- **Runs on:** `ubuntu-latest`
- **Steps:**
  1. Free up disk space
  2. Set up Docker Buildx
  3. Build production container (multi-arch: amd64, arm64)
  4. Run tests inside container
  5. Tag with git SHA
  6. Push to GitHub Container Registry (ghcr.io)

#### Job 4: Docker Security Scan (`docker-security`)
- **Runs on:** `ubuntu-latest`
- **Steps:**
  1. Pull built images
  2. **Trivy** scan for vulnerabilities
  3. **Grype** scan (alternative scanner)
  4. Generate SBOM (Software Bill of Materials)
  5. Upload scan results

#### Job 5: Multi-Architecture Build (`multi-arch`)
- **Conditional:** Only on main branch or release tags
- **Platforms:** linux/amd64, linux/arm64
- **Steps:**
  1. Set up QEMU for cross-platform builds
  2. Build for both architectures
  3. Push manifests to registry
  4. Tag with version (from pyproject.toml)

---

### 3. **Performance Benchmarking** (`.github/workflows/benchmark.yml`)

**Trigger:** Manual dispatch, weekly schedule, on release

**Jobs:**

#### Job 1: Model Performance (`benchmark-model`)
- **Runs on:** `ubuntu-latest` with GPU (if available)
- **Steps:**
  1. Checkout code
  2. Set up Python
  3. Download model + datasets
  4. Run comprehensive benchmarks:
     - 10-fold CV on Boughter
     - Test on Jain (PARITY_86)
     - Test on Shehata (with threshold calibration)
     - Test on Harvey (141k sequences)
  5. Compare against baseline (stored in repo):
     - Accuracy deltas
     - Confusion matrix differences
     - Inference time (mean ± std)
  6. Upload results to Weights & Biases (W&B)
  7. Comment performance report on PR
  8. **FAIL if performance regresses** by >2%

#### Job 2: Inference Speed (`benchmark-speed`)
- **Steps:**
  1. Measure embedding extraction time
  2. Measure prediction time
  3. Track memory usage
  4. Store results in GitHub Pages (trend charts)

---

### 4. **Dependency Management** (`.github/workflows/dependencies.yml`)

**Trigger:** Daily schedule (6am UTC)

**Jobs:**

#### Job 1: Update Dependencies (`update-deps`)
- **Steps:**
  1. Checkout code
  2. Run `uv sync --upgrade`
  3. Run full test suite
  4. If tests pass:
     - Create PR with updated `uv.lock`
     - Auto-assign reviewers
     - Label: `dependencies`

#### Job 2: Security Audit (`audit`)
- **Steps:**
  1. `pip-audit` on all dependencies
  2. Check for outdated packages with known CVEs
  3. Create issues for HIGH/CRITICAL vulnerabilities

---

### 5. **Release Automation** (`.github/workflows/release.yml`)

**Trigger:** On push of version tag (e.g., `v0.2.0`)

**Jobs:**

#### Job 1: Validate Release (`validate`)
- **Steps:**
  1. Checkout code
  2. Verify version in pyproject.toml matches tag
  3. Verify CHANGELOG.md updated
  4. Run full test suite
  5. Run E2E Novo parity checks

#### Job 2: Build Artifacts (`build`)
- **Steps:**
  1. Build Python wheel (`uv build`)
  2. Build Docker images (multi-arch)
  3. Generate SBOM
  4. Sign artifacts (cosign)

#### Job 3: Publish (`publish`)
- **Conditional:** Only if validate + build pass
- **Steps:**
  1. Publish wheel to PyPI (test.pypi.org first, then pypi.org)
  2. Push Docker images to ghcr.io with tags:
     - `latest`
     - `v0.2.0`
     - `v0.2`
     - `v0`
  3. Create GitHub Release with:
     - Auto-generated changelog
     - Artifacts (wheel, SBOM)
     - Benchmark results
     - Docker pull commands

#### Job 4: Deploy Docs (`deploy-docs`)
- **Steps:**
  1. Build documentation (if we add Sphinx/MkDocs)
  2. Deploy to GitHub Pages
  3. Update "latest" and versioned docs

---

### 6. **Documentation** (`.github/workflows/docs.yml`)

**Trigger:** Every push to main, manual dispatch

**Jobs:**

#### Job 1: Build Docs (`build`)
- **Steps:**
  1. Install Sphinx/MkDocs
  2. Build HTML documentation
  3. Check for broken links
  4. Upload as artifact

#### Job 2: Deploy to GitHub Pages (`deploy`)
- **Conditional:** Only on main branch
- **Steps:**
  1. Download built docs
  2. Deploy to `gh-pages` branch
  3. Available at: `https://the-obstacle-is-the-way.github.io/antibody_training_pipeline_ESM/`

---

### 7. **Reproducibility Check** (`.github/workflows/reproducibility.yml`)

**Trigger:** Every PR, weekly schedule

**Jobs:**

#### Job 1: Novo Parity Validation (`validate-novo`)
- **Purpose:** Ensure we maintain reproducibility of published results
- **Steps:**
  1. Download latest model
  2. Run inference on all test sets
  3. Compare confusion matrices:
     - **Jain:** Must match [[40,19],[10,17]] exactly
     - **Shehata:** Must match [[229,162],[2,5]] with threshold=0.5495
     - **Harvey:** Must be within 61.3-61.7% accuracy
  4. **FAIL if any benchmark regresses**
  5. Post results as PR comment

---

### 8. **Scheduled Maintenance** (`.github/workflows/maintenance.yml`)

**Trigger:** Weekly (Sundays 2am UTC)

**Jobs:**

#### Job 1: Clean Caches (`clean-cache`)
- **Steps:**
  1. Remove old GitHub Actions caches (>7 days)
  2. Remove old Docker layer caches

#### Job 2: Update Test Data (`update-data`)
- **Steps:**
  1. Check for updated datasets from sources
  2. Validate data integrity
  3. Create PR if updates found

---

### 9. **Model Registry Sync** (`.github/workflows/model-registry.yml`)

**Trigger:** On release, manual dispatch

**Jobs:**

#### Job 1: Upload to HuggingFace (`hf-upload`)
- **Steps:**
  1. Checkout code
  2. Download trained model
  3. Create model card (metadata)
  4. Upload to HuggingFace Hub:
     - Repository: `The-Obstacle-Is-The-Way/antibody-nonspecificity-esm1v`
     - Include: model, tokenizer, test results
  5. Tag with version

#### Job 2: Log to Weights & Biases (`wandb-log`)
- **Steps:**
  1. Log model artifacts
  2. Log benchmark results
  3. Track lineage (dataset versions, code commit)

---

### 10. **Pull Request Automation** (`.github/workflows/pr.yml`)

**Trigger:** On PR open/update

**Jobs:**

#### Job 1: PR Validation (`validate`)
- **Steps:**
  1. Verify PR title follows conventional commits
  2. Verify description not empty
  3. Verify linked issue exists
  4. Check for merge conflicts

#### Job 2: Auto-Label (`label`)
- **Steps:**
  1. Label based on changed files:
     - `docs` if only .md files
     - `tests` if only test files
     - `docker` if Dockerfile changes
     - `core` if src/ changes
  2. Label based on size: `size/XS`, `size/S`, `size/M`, etc.

#### Job 3: Auto-Assign Reviewers (`assign`)
- **Steps:**
  1. Auto-assign based on CODEOWNERS
  2. Request review from team members

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Priority:** HIGH
**Estimated Effort:** 2-3 days

- [ ] Create `ci.yml` with quality + unit tests
- [ ] Set up Codecov integration
- [ ] Configure branch protection rules on main
- [ ] Add status badges to README

**Success Criteria:**
- All PRs must pass quality gates
- 80%+ code coverage enforced
- No merges to main without CI passing

---

### Phase 2: Comprehensive Testing (Week 2)
**Priority:** HIGH
**Estimated Effort:** 3-4 days

- [ ] Add integration test job
- [ ] Add E2E test job (with Novo parity validation)
- [ ] Set up test result reporting
- [ ] Add performance regression detection

**Success Criteria:**
- All 358 tests run in CI
- Novo benchmarks validated on every PR
- Test results commented on PRs

---

### Phase 3: Security & Quality (Week 2-3)
**Priority:** MEDIUM
**Estimated Effort:** 2-3 days

- [ ] Add dependency security scanning
- [ ] Add Docker security scanning
- [ ] Add SAST with bandit
- [ ] Generate SBOMs

**Success Criteria:**
- No HIGH/CRITICAL vulnerabilities merged
- All Docker images scanned
- Security reports generated

---

### Phase 4: Docker Multi-Arch (Week 3)
**Priority:** MEDIUM
**Estimated Effort:** 2 days

- [ ] Set up multi-architecture builds (amd64, arm64)
- [ ] Optimize Docker layer caching
- [ ] Fix disk space issues permanently
- [ ] Push to ghcr.io with proper tags

**Success Criteria:**
- Builds succeed for both architectures
- Images published to registry
- Clear tagging strategy (latest, version, SHA)

---

### Phase 5: Automation & Release (Week 4)
**Priority:** MEDIUM
**Estimated Effort:** 3 days

- [ ] Automated dependency updates
- [ ] Release workflow with semantic versioning
- [ ] Changelog generation
- [ ] PyPI publishing

**Success Criteria:**
- Releases fully automated
- Changelog auto-generated
- Package published to PyPI

---

### Phase 6: Benchmarking & Monitoring (Week 5)
**Priority:** LOW
**Estimated Effort:** 3-4 days

- [ ] Performance benchmark workflow
- [ ] Model registry sync (HuggingFace)
- [ ] Weights & Biases integration
- [ ] GitHub Pages for results

**Success Criteria:**
- Performance tracked over time
- Models synced to HuggingFace
- Public benchmark dashboard

---

### Phase 7: Documentation (Week 6)
**Priority:** LOW
**Estimated Effort:** 2 days

- [ ] Set up Sphinx or MkDocs
- [ ] Auto-deploy to GitHub Pages
- [ ] API documentation generation
- [ ] Tutorial notebooks

**Success Criteria:**
- Docs auto-published on merge to main
- API docs auto-generated from docstrings
- Searchable documentation site

---

## Success Metrics

### Code Quality Metrics
- **Test Coverage:** ≥80% (current: ~85%)
- **Linting:** 100% pass rate (ruff)
- **Type Coverage:** 100% (mypy strict)
- **Security Issues:** 0 HIGH/CRITICAL

### CI/CD Performance Metrics
- **CI Run Time:** <15 minutes for full pipeline
- **Docker Build Time:** <10 minutes with caching
- **PR Feedback Time:** <5 minutes for basic checks

### Reproducibility Metrics
- **Novo Parity Checks:** 100% pass rate
- **Benchmark Stability:** <2% variance in accuracy

### Deployment Metrics
- **Release Frequency:** Weekly (if changes exist)
- **Failed Deployments:** <5%
- **Rollback Time:** <10 minutes

---

## Technology Stack

### CI/CD Tools
- **GitHub Actions** - Primary CI/CD platform
- **Docker Buildx** - Multi-arch container builds
- **uv** - Python package management
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **Codecov** - Coverage tracking service

### Code Quality
- **ruff** - Linting + formatting
- **mypy** - Static type checking
- **bandit** - Security linting (SAST)

### Security Scanning
- **pip-audit** - Python dependency vulnerabilities
- **Safety** - PyPI vulnerability database
- **Trivy** - Container + dependency scanning
- **Grype** - Alternative vulnerability scanner
- **hadolint** - Dockerfile linting

### Documentation
- **Sphinx** or **MkDocs** - Documentation generation
- **GitHub Pages** - Doc hosting

### Model Registry
- **HuggingFace Hub** - Model hosting
- **Weights & Biases** - Experiment tracking

### Artifact Storage
- **GitHub Container Registry (ghcr.io)** - Docker images
- **PyPI** - Python packages
- **GitHub Releases** - Release artifacts

---

## Branch Protection Rules

### Main Branch
**Required Checks:**
- ✅ Code quality (ruff, mypy)
- ✅ Unit tests (all Python versions)
- ✅ Integration tests
- ✅ Docker builds
- ✅ Security scans

**Settings:**
- Require PR reviews: 1 (or 2 for production)
- Dismiss stale reviews: Yes
- Require status checks to pass: Yes
- Require branches to be up to date: Yes
- Include administrators: Yes
- Restrict push: Yes (only via PR)

### Feature Branches
**Naming Convention:**
- `feature/<description>`
- `fix/<description>`
- `docs/<description>`
- `chore/<description>`

**Auto-Delete:** After merge

---

## Secrets & Environment Variables

### Required GitHub Secrets
```bash
# PyPI Publishing
PYPI_API_TOKEN=<token>
TEST_PYPI_API_TOKEN=<token>

# HuggingFace
HF_TOKEN=<token>

# Weights & Biases
WANDB_API_KEY=<key>

# Codecov
CODECOV_TOKEN=<token>

# Docker Registry (auto-provided by GitHub)
GITHUB_TOKEN=<auto>
```

### Environment Variables
```bash
# Python version matrix
PYTHON_VERSIONS="3.11,3.12"

# Test settings
PYTEST_TIMEOUT=300
COVERAGE_THRESHOLD=80

# Docker settings
DOCKER_BUILDKIT=1
BUILDKIT_PROGRESS=plain
```

---

## Cost Estimation

### GitHub Actions Minutes (Free Tier: 2000 min/month)

**Current Usage (estimated):**
- CI pipeline: ~10 min per push × 50 pushes/month = 500 min
- Docker builds: ~8 min per push × 50 pushes/month = 400 min
- **Total:** ~900 min/month ✅ Under limit

**With Full Implementation:**
- CI pipeline: ~15 min × 50 = 750 min
- Docker multi-arch: ~15 min × 50 = 750 min
- E2E tests: ~20 min × 10 PR/month = 200 min
- Benchmarks: ~30 min × 4 weekly = 120 min
- **Total:** ~1,820 min/month ✅ Still under limit

**Optimization Strategies:**
- Use caching aggressively (uv, pip, Docker layers)
- Run expensive jobs only on main/PR to main
- Skip redundant jobs (e.g., E2E only on critical PRs)

---

## Monitoring & Alerting

### What to Monitor
1. **CI Success Rate:** Should be >95%
2. **Build Duration Trends:** Should not increase over time
3. **Test Flakiness:** Track flaky tests
4. **Security Vulnerabilities:** Alert on new CVEs
5. **Model Performance:** Track benchmark drift

### Alerting Channels
- **GitHub Issues:** Auto-create for failed scheduled jobs
- **Email:** Critical security vulnerabilities
- **Slack/Discord:** Optional integration

---

## Rollback & Disaster Recovery

### Rollback Procedures
1. **Code Rollback:** Revert PR or cherry-pick fix
2. **Release Rollback:** Delete tag, re-release previous version
3. **Docker Image Rollback:** Pull previous tag from ghcr.io
4. **Model Rollback:** Restore from HuggingFace version history

### Backup Strategy
- **Code:** Git history (permanent)
- **Artifacts:** GitHub Releases (permanent)
- **Docker Images:** ghcr.io (retention: 90 days for untagged)
- **Models:** HuggingFace + local backups

---

## Senior Review Checklist

**Before Approving This Spec:**

- [ ] **Scope:** Is this comprehensive enough for a production ML repo?
- [ ] **Feasibility:** Can we implement this in 6 weeks?
- [ ] **Cost:** Are we within GitHub Actions free tier limits?
- [ ] **Priorities:** Are phases prioritized correctly?
- [ ] **Security:** Do we cover all security scanning needs?
- [ ] **Reproducibility:** Do we validate Novo parity sufficiently?
- [ ] **Team Capacity:** Do we have bandwidth for implementation?
- [ ] **Maintenance:** Is this sustainable long-term?

**Questions for Discussion:**

1. Should we use HuggingFace Hub or build our own model registry?
2. Do we need GPU runners for benchmarking? (costs $$)
3. Should we publish to PyPI immediately or wait?
4. Do we want Weights & Biases or stick with GitHub artifacts?
5. What's our release cadence target? (weekly, bi-weekly, monthly?)

---

## Next Steps

1. **Senior Review:** Schedule review meeting
2. **Approval:** Get sign-off on this spec
3. **Resource Allocation:** Assign team members to phases
4. **Timeline:** Finalize 6-week implementation plan
5. **Kickoff:** Start Phase 1 implementation

---

## References

### Industry Best Practices
- [GitHub Actions Best Practices](https://docs.github.com/en/actions/learn-github-actions/best-practices-for-github-actions)
- [Docker Build Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python CI/CD with GitHub Actions](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

### Similar ML Projects
- [HuggingFace Transformers CI](https://github.com/huggingface/transformers/tree/main/.github/workflows)
- [PyTorch CI](https://github.com/pytorch/pytorch/tree/main/.github/workflows)
- [scikit-learn CI](https://github.com/scikit-learn/scikit-learn/tree/main/.github/workflows)

### Security Standards
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Secure Software Development](https://csrc.nist.gov/publications/detail/sp/800-218/final)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** ⏳ Pending Senior Review
**Approval Required From:**
- [ ] Technical Lead
- [ ] DevOps/Platform Team
- [ ] Security Team
- [ ] Product Owner

---

**End of Specification**
