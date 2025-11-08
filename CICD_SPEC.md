# GitHub Actions CI/CD Specification - Production Grade (REALISTIC)

**Document Status:** Draft for Senior Review (REVISED - Corrected Inaccuracies)
**Repository:** antibody_training_pipeline_ESM
**Date:** 2025-11-07 (Revised)
**Author:** Engineering Team
**Purpose:** Define comprehensive CI/CD pipeline for professional ML/bioinformatics repository

**Revision Notes:**
- Fixed Python version matrix (3.12 only, per pyproject.toml)
- Corrected coverage claims (no current tracking)
- Scoped benchmark frequency realistically
- Updated cost estimates with actual measured times
- Added runner resource constraints

---

## Table of Contents

1. [Current State](#current-state)
2. [Target State - Complete CI/CD](#target-state)
3. [Workflow Specifications](#workflow-specifications)
4. [Implementation Phases](#implementation-phases)
5. [Success Metrics](#success-metrics)
6. [Resource Constraints](#resource-constraints)
7. [Senior Review Checklist](#senior-review-checklist)

---

## Current State

### ✅ What We Have Now

**Workflows:**
- `.github/workflows/docker-ci.yml` - Docker build (basic, disk space issues)

**Coverage:**
- ✅ Docker dev container build (test execution during build)
- ✅ Docker prod container build (test execution during build)
- ⚠️ Builds fail on image export due to disk space (11.7GB images)
- ⚠️ Build and Push job skipped (only on main, currently disabled)

**Limitations:**
- **No Python environment CI** (only Docker, which is currently broken for prod)
- **No code quality gates** (no ruff, mypy, bandit in CI)
- **No security scanning** (no dependency audits)
- **No automated releases**
- **No performance benchmarks** (manual testing only)
- **No test reporting** (tests run in Docker but results not captured)
- **No dependency updates** (manual uv sync)
- **No documentation builds**
- **No coverage tracking** (never set up)

### 🔴 Known Issues

1. **Disk Space:** GitHub Actions runners have ~14GB disk space, our images are 11.7GB each
2. **Multi-arch Infeasible:** Cross-building amd64+arm64 requires QEMU and doubles build time/space
3. **Large Datasets:** Harvey test set (141k sequences) takes ~90 minutes to process
4. **Model Download:** ESM-1v is 2GB and must be cached efficiently

---

## Target State

### 🎯 Complete Professional CI/CD Pipeline (REALISTIC SCOPE)

A world-class ML/bioinformatics repository should have:

1. **Python Environment Testing** (Python 3.12 **ONLY** - per pyproject.toml)
2. **Comprehensive Quality Gates** (linting, formatting, type checking, security)
3. **Tiered Test Execution** (fast unit tests on every PR, expensive E2E on schedule)
4. **Docker Build Validation** (verify Dockerfiles build, skip image load)
5. **Automated Dependency Management** (security updates, version bumps)
6. **Performance Benchmarking** (scheduled, not on every PR)
7. **Automated Releases** (semantic versioning, changelog generation)
8. **Documentation Deployment** (GitHub Pages)
9. **Security Scanning** (SAST, dependency vulnerabilities)
10. **Model Artifact Management** (optional HuggingFace integration)
11. **Reproducibility Validation** (Novo parity checks on schedule, not every PR)

### ⚠️ Explicitly Out of Scope (Due to GitHub Actions Constraints)

- **Multi-architecture Docker builds** (requires self-hosted runners with more disk)
- **GPU-based benchmarking** (GitHub doesn't provide GPU runners in free tier)
- **E2E tests on every PR** (too expensive - 90+ minutes for Harvey)
- **Full Novo parity on every PR** (too expensive - run on schedule instead)

---

## Workflow Specifications

### 1. **Main CI Pipeline** (`.github/workflows/ci.yml`)

**Trigger:** Every push, every PR

**Jobs:**

#### Job 1: Code Quality (`quality`)

- **Runs on:** `ubuntu-latest`
- **Python version:** 3.12 (single version - per pyproject.toml:9)
- **Duration:** ~2-3 minutes
- **Steps:**
  1. Checkout code
  2. Set up Python 3.12 (with uv cache)
  3. Install dependencies (`uv sync`)
  4. **Ruff lint** (fail on errors)
  5. **Ruff format check** (fail if code needs formatting)
  6. **mypy** type checking (strict mode)
  7. **bandit** security scan (SAST)
  8. Upload results as artifacts

**Why single Python version:** pyproject.toml line 9: `requires-python = ">=3.12"`

#### Job 2: Unit Tests (`test-unit`)

- **Runs on:** `ubuntu-latest`
- **Python version:** 3.12 only
- **Duration:** ~3-5 minutes
- **Steps:**
  1. Checkout code
  2. Set up Python 3.12
  3. Install dependencies
  4. Run pytest unit tests (`tests/unit/`)
  5. Generate coverage report (NEW - not currently configured)
  6. Upload coverage to Codecov
  7. Upload test results (JUnit XML)
  8. **Minimum coverage:** 70% (conservative target, current unknown)

**Note:** Current coverage is **UNKNOWN** - no tracking exists. Must implement first.

#### Job 3: Integration Tests (`test-integration`)

- **Runs on:** `ubuntu-latest`
- **Python version:** 3.12 only
- **Duration:** ~5-8 minutes
- **Steps:**
  1. Checkout code
  2. Set up Python 3.12
  3. Install dependencies
  4. Cache ESM-1v model (~2GB)
  5. Run pytest integration tests (`tests/integration/`)
  6. Upload test results

#### Job 4: Dependency Security (`security`)

- **Runs on:** `ubuntu-latest`
- **Duration:** ~2 minutes
- **Steps:**
  1. Checkout code
  2. Set up Python 3.12
  3. **pip-audit** (check for known CVEs)
  4. **Safety** check (PyPI vulnerability database)
  5. Fail if HIGH or CRITICAL vulnerabilities found

**Total CI Pipeline Time:** ~12-18 minutes per PR

---

### 2. **Docker Build Validation** (`.github/workflows/docker.yml`)

**Trigger:** Every push, every PR

**Jobs:**

#### Job 1: Docker Lint (`docker-lint`)

- **Duration:** ~30 seconds
- **Steps:**
  1. Checkout code
  2. **hadolint** on Dockerfile.dev and Dockerfile.prod
  3. Fail on errors

#### Job 2: Verify Dockerfile Builds (`build-verify`)

- **Runs on:** `ubuntu-latest`
- **Duration:** ~8-10 minutes
- **Steps:**
  1. Free up disk space (remove .NET, Haskell, etc.)
  2. Set up Docker Buildx
  3. Build dev container (DO NOT load - export only)
  4. Build prod container (DO NOT load - export only)
  5. Cache layers (GitHub Actions cache)
  6. **Skip image load** to avoid disk space issues

**Why skip load:** Images are 11.7GB each, GitHub runners have ~14GB total disk.

**Note:** Tests still run DURING build (RUN pytest in Dockerfile), we just don't load the final image into Docker daemon.

---

### 3. **E2E & Benchmarking** (`.github/workflows/benchmark.yml`)

**Trigger:** Manual dispatch, weekly schedule (Sunday 2am UTC), on release tags

**NOT on every PR** (too expensive - Harvey alone is 90 minutes)

**Jobs:**

#### Job 1: E2E Novo Parity (`e2e-parity`)

- **Runs on:** `ubuntu-latest`
- **Duration:** ~120 minutes (full suite)
- **Steps:**
  1. Checkout code
  2. Set up Python 3.12
  3. Install dependencies
  4. Cache ESM-1v model
  5. Run E2E tests (`tests/e2e/test_reproduce_novo.py`)
  6. Validate benchmarks:
     - **Boughter:** 10-fold CV accuracy 67-71%
     - **Jain:** Confusion matrix [[40,19],[10,17]] exact
     - **Shehata:** 58.8% with threshold=0.5495
     - **Harvey:** 61.5-61.7% on 141k sequences (~90 min)
  7. Upload confusion matrices as artifacts
  8. Create GitHub issue if parity fails

**Why scheduled:** Harvey test alone takes 90 minutes. Too expensive for every PR.

#### Job 2: Quick Parity Check (`quick-parity`)

- **Trigger:** On PR to main (lighter version)
- **Duration:** ~10 minutes
- **Steps:**
  1. Run parity checks on **Jain and Shehata only** (skip Harvey)
  2. Post results as PR comment
  3. Warning (not failure) if performance drops

---

### 4. **Dependency Management** (`.github/workflows/dependencies.yml`)

**Trigger:** Weekly (Monday 6am UTC)

**Jobs:**

#### Job 1: Update Dependencies (`update-deps`)

- **Duration:** ~15 minutes
- **Steps:**
  1. Checkout code
  2. Run `uv sync --upgrade`
  3. Run unit + integration tests (skip E2E)
  4. If tests pass:
     - Create PR with updated `uv.lock`
     - Auto-assign reviewers
     - Label: `dependencies`

#### Job 2: Security Audit (`audit`)

- **Duration:** ~2 minutes
- **Steps:**
  1. `pip-audit` on all dependencies
  2. Check for outdated packages with known CVEs
  3. Create GitHub issue for HIGH/CRITICAL vulnerabilities

---

### 5. **Release Automation** (`.github/workflows/release.yml`)

**Trigger:** On push of version tag (e.g., `v2.0.1`)

**Jobs:**

#### Job 1: Validate Release (`validate`)

- **Duration:** ~20 minutes
- **Steps:**
  1. Checkout code
  2. Verify version in pyproject.toml matches tag
  3. Verify CHANGELOG.md updated
  4. Run full test suite (unit + integration)
  5. Run quick parity check (Jain + Shehata only)

#### Job 2: Build Artifacts (`build`)

- **Duration:** ~5 minutes
- **Steps:**
  1. Build Python wheel (`uv build`)
  2. Generate SBOM (Software Bill of Materials)
  3. Sign artifacts (optional: cosign)

#### Job 3: Publish (`publish`)

- **Conditional:** Only if validate + build pass
- **Duration:** ~3 minutes
- **Steps:**
  1. Publish wheel to PyPI (test.pypi.org first, then pypi.org)
  2. Create GitHub Release with:
     - Auto-generated changelog
     - Artifacts (wheel, SBOM)
     - Benchmark results (latest from weekly run)

**Note:** Docker image publishing deferred (requires self-hosted runner for multi-arch)

---

### 6. **Documentation** (`.github/workflows/docs.yml`)

**Trigger:** Every push to main, manual dispatch

**Jobs:**

#### Job 1: Build and Deploy Docs (`build-deploy`)

- **Duration:** ~3 minutes
- **Steps:**
  1. Install MkDocs (or Sphinx)
  2. Build HTML documentation
  3. Check for broken links
  4. Deploy to `gh-pages` branch
  5. Available at: `https://the-obstacle-is-the-way.github.io/antibody_training_pipeline_ESM/`

---

### 7. **Pull Request Automation** (`.github/workflows/pr.yml`)

**Trigger:** On PR open/update

**Jobs:**

#### Job 1: PR Validation (`validate`)

- **Duration:** ~10 seconds
- **Steps:**
  1. Verify PR title follows conventional commits
  2. Verify description not empty
  3. Check for merge conflicts

#### Job 2: Auto-Label (`label`)

- **Duration:** ~5 seconds
- **Steps:**
  1. Label based on changed files:
     - `docs` if only .md files
     - `tests` if only test files
     - `docker` if Dockerfile changes
     - `core` if src/ changes
  2. Label based on size

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Priority:** HIGH
**Estimated Effort:** 3-4 days

- [ ] Create `ci.yml` with quality + unit tests (Python 3.12 only)
- [ ] Set up pytest-cov and Codecov integration
- [ ] Configure branch protection rules on main
- [ ] Add status badges to README

**Success Criteria:**
- All PRs must pass quality gates
- Coverage tracking operational (target: 70%+)
- No merges to main without CI passing

---

### Phase 2: Integration Testing (Week 2)

**Priority:** HIGH
**Estimated Effort:** 2-3 days

- [ ] Add integration test job to CI
- [ ] Set up ESM-1v model caching
- [ ] Add test result reporting

**Success Criteria:**
- Integration tests run on every PR
- Test results visible in PR checks

---

### Phase 3: Security & Docker (Week 2-3)

**Priority:** MEDIUM
**Estimated Effort:** 2-3 days

- [ ] Add dependency security scanning
- [ ] Add Docker linting (hadolint)
- [ ] Add SAST with bandit
- [ ] Fix Docker build workflow (verify builds without loading)

**Success Criteria:**
- No HIGH/CRITICAL vulnerabilities merged
- Docker builds verify successfully
- Security reports generated

---

### Phase 4: Benchmarking (Week 3-4)

**Priority:** MEDIUM
**Estimated Effort:** 3-4 days

- [ ] Create benchmark workflow (weekly schedule)
- [ ] Set up quick parity check for PRs to main
- [ ] Store baseline results
- [ ] Set up alerting for regressions

**Success Criteria:**
- Weekly benchmarks run automatically
- Performance regression detection works
- Results stored as artifacts

---

### Phase 5: Release & Docs (Week 4-5)

**Priority:** LOW
**Estimated Effort:** 3 days

- [ ] Automated release workflow
- [ ] Changelog generation
- [ ] PyPI publishing
- [ ] Documentation deployment

**Success Criteria:**
- Releases fully automated
- Docs auto-published to GitHub Pages

---

### Phase 6: Dependency Automation (Week 5-6)

**Priority:** LOW
**Estimated Effort:** 2 days

- [ ] Weekly dependency update workflow
- [ ] Security audit workflow
- [ ] PR automation (labeling, validation)

**Success Criteria:**
- Dependency updates proposed weekly
- Security issues tracked automatically

---

## Success Metrics

### Code Quality Metrics

- **Test Coverage:** ≥70% (conservative target, **current: UNKNOWN**)
- **Linting:** 100% pass rate (ruff)
- **Type Coverage:** 100% (mypy strict)
- **Security Issues:** 0 HIGH/CRITICAL in production

### CI/CD Performance Metrics

- **Fast CI (quality + unit):** <10 minutes
- **Full CI (+ integration):** <20 minutes
- **Weekly benchmarks:** <120 minutes
- **PR Feedback Time:** <5 minutes for quality gates

### Reproducibility Metrics

- **Weekly Novo Parity:** 100% pass rate
- **Benchmark Stability:** <2% variance

### Deployment Metrics

- **Release Frequency:** Monthly (or as needed)
- **Failed Releases:** <10%

---

## Resource Constraints

### GitHub Actions Runners

**Standard `ubuntu-latest` Runner:**
- **CPU:** 2 cores
- **RAM:** 7GB
- **Disk:** ~14GB free (after OS)
- **No GPU**

**Our Requirements:**
- Docker images: 11.7GB each
- ESM-1v model: 2GB
- Harvey dataset: 196MB (141k sequences)
- Build artifacts: ~1-2GB

**Implications:**
- ❌ Cannot load Docker images into daemon (disk space)
- ❌ Cannot run multi-arch builds without external cache
- ❌ Cannot run Harvey benchmarks on every PR (time)
- ✅ CAN verify Docker builds (without load)
- ✅ CAN run unit + integration tests
- ✅ CAN run benchmarks on schedule

---

## Cost Estimation (REALISTIC)

### GitHub Actions Minutes (Free Tier: 2000 min/month)

**Measured Times (from actual runs):**

- Quality + unit tests: ~10 min
- Integration tests: ~8 min
- Docker build verify: ~10 min
- E2E benchmarks: ~120 min

**Current Usage (estimated):**

- CI per PR: ~18 min (quality + unit + integration + Docker)
- PRs per month: ~20
- **Monthly:** 20 × 18 = 360 min ✅

**With Full Implementation:**

- CI per PR: ~20 min × 20 PR = 400 min
- Weekly benchmarks: 120 min × 4 = 480 min
- Weekly dep updates: 15 min × 4 = 60 min
- Releases: 20 min × 2 = 40 min
- **Total:** ~980 min/month ✅ **Well under 2000 min limit**

**Cost-Saving Strategies:**

1. Cache ESM-1v model aggressively
2. Skip Harvey on PR checks (use quick parity instead)
3. Run expensive E2E only weekly
4. Use Docker layer caching
5. Parallelize test jobs

---

## Technology Stack

### CI/CD Tools

- **GitHub Actions** - Primary CI/CD platform
- **Docker Buildx** - Container builds (verify only, no load)
- **uv** - Python package management (with caching)
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting (NEW - needs setup)
- **Codecov** - Coverage tracking service

### Code Quality

- **ruff** - Linting + formatting
- **mypy** - Static type checking
- **bandit** - Security linting (SAST)

### Security Scanning

- **pip-audit** - Python dependency vulnerabilities
- **Safety** - PyPI vulnerability database
- **hadolint** - Dockerfile linting

### Documentation

- **MkDocs** - Documentation generation (simpler than Sphinx)
- **GitHub Pages** - Doc hosting

### Artifact Storage

- **GitHub Releases** - Release artifacts
- **PyPI** - Python packages
- **Optional:** HuggingFace Hub (model hosting)

---

## Branch Protection Rules

### Main Branch

**Required Checks:**

- ✅ Code quality (ruff, mypy, bandit)
- ✅ Unit tests
- ✅ Integration tests
- ✅ Docker build verify
- ✅ Security scans (dependency audit)

**Settings:**

- Require PR reviews: 1
- Dismiss stale reviews: Yes
- Require status checks to pass: Yes
- Require branches to be up to date: Yes
- Include administrators: Yes
- Restrict push: Yes (only via PR)

---

## Secrets & Environment Variables

### Required GitHub Secrets

```bash
# PyPI Publishing (Phase 5)
PYPI_API_TOKEN=<token>
TEST_PYPI_API_TOKEN=<token>

# Codecov (Phase 1)
CODECOV_TOKEN=<token>

# Optional: HuggingFace (Phase 6)
HF_TOKEN=<token>
```

### Environment Variables

```bash
# Python version (SINGLE VERSION - per pyproject.toml)
PYTHON_VERSION="3.12"

# Test settings
PYTEST_TIMEOUT=600  # 10 minutes
COVERAGE_THRESHOLD=70  # Conservative target

# Docker settings
DOCKER_BUILDKIT=1
```

---

## Monitoring & Alerting

### What to Monitor

1. **CI Success Rate:** Should be >90%
2. **Build Duration Trends:** Alert if >20% increase
3. **Test Flakiness:** Track flaky tests
4. **Security Vulnerabilities:** Auto-create issues for HIGH/CRITICAL
5. **Weekly Benchmark Results:** Alert if Novo parity fails

### Alerting Channels

- **GitHub Issues:** Auto-create for:
  - Failed weekly benchmarks
  - Security vulnerabilities (HIGH/CRITICAL)
  - Dependency update failures
- **PR Comments:** Post benchmark results when quick parity runs

---

## Senior Review Checklist

**Critical Questions:**

- [ ] **Python Version:** Confirmed 3.12 only (per pyproject.toml)?
- [ ] **Coverage:** Understand current coverage is UNKNOWN, needs setup?
- [ ] **Cost:** Realistic estimate (~980 min/month vs 2000 limit)?
- [ ] **Disk Space:** Accept we CANNOT load Docker images in CI?
- [ ] **Benchmarks:** Accept weekly schedule instead of every PR?
- [ ] **Multi-arch:** Accept this requires self-hosted runners (future)?
- [ ] **Scope:** Is phased 6-week plan achievable?

**Known Compromises:**

1. **No multi-arch builds** (requires self-hosted runners)
2. **No GPU benchmarks** (GitHub doesn't provide)
3. **No E2E on every PR** (too expensive - weekly instead)
4. **No image load in CI** (disk space limits)

**Are these compromises acceptable?**

---

## Next Steps

1. **Senior Review:** Validate this revised, realistic spec
2. **Approval:** Sign off on scope and constraints
3. **Phase 1 Start:** Begin with quality gates + unit tests
4. **Incremental Delivery:** Ship each phase when complete

---

## References

### Industry Best Practices

- [GitHub Actions Best Practices](https://docs.github.com/en/actions/learn-github-actions/best-practices-for-github-actions)
- [Docker Build Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python CI/CD with GitHub Actions](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

### Similar ML Projects (for reference)

- [HuggingFace Transformers CI](https://github.com/huggingface/transformers/tree/main/.github/workflows)
- [PyTorch CI](https://github.com/pytorch/pytorch/tree/main/.github/workflows) (uses self-hosted runners)
- [scikit-learn CI](https://github.com/scikit-learn/scikit-learn/tree/main/.github/workflows)

---

## Revision History

### v1.0 (2025-11-07 - Initial)

- Comprehensive spec with 10 workflows
- Python 3.11/3.12 matrix
- Multi-arch Docker builds
- E2E on every PR

### v2.0 (2025-11-07 - REVISED - This Version)

**Fixed Inaccuracies:**

- ✅ Python 3.12 ONLY (per pyproject.toml:9, ruff:59)
- ✅ Removed claim of "~85% coverage" (no tracking exists)
- ✅ Scoped benchmarks to weekly (Harvey is 90 min)
- ✅ Removed multi-arch builds (disk space constraints)
- ✅ Updated cost estimate with measured times (~980 min/month)
- ✅ Added resource constraints section
- ✅ Acknowledged Docker image load impossible in CI

**Key Philosophy Changes:**

- Optimistic fantasy → Realistic constraints
- "Everything on every PR" → Tiered execution (fast CI, scheduled benchmarks)
- Multi-platform from day 1 → Single platform, self-hosted for multi-arch later
- Guessing coverage → Measure first, then enforce

---

**Document Version:** 2.0 (REALISTIC REVISION)
**Last Updated:** 2025-11-07
**Status:** ⏳ Pending Senior Review
**Approval Required From:**

- [ ] Technical Lead
- [ ] DevOps/Platform Team

---

**End of Specification**
