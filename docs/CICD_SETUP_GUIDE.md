# CI/CD Setup Guide - Manual Steps

**Last Updated:** 2025-11-07
**Status:** Phase 1-3 Implemented
**Repository:** antibody_training_pipeline_ESM

This guide covers the **manual configuration steps** required to complete the CI/CD setup after workflows are deployed.

---

## 📋 Overview

**What's Automated:**
- ✅ CI pipeline (quality, tests, security)
- ✅ Docker build verification
- ✅ Weekly E2E benchmarks
- ✅ Weekly dependency updates
- ✅ Security audits

**What Requires Manual Setup:**
- ⏳ Codecov token (coverage reporting)
- ⏳ Branch protection rules
- ⏳ Optional: Release automation (future)

---

## 1. Codecov Setup

### 1.1. Enable Codecov for Repository

1. Go to [codecov.io](https://codecov.io)
2. Sign in with GitHub
3. Click **Add New Repository**
4. Find and enable: `the-obstacle-is-the-way/antibody_training_pipeline_ESM`

### 1.2. Get Codecov Token

1. In Codecov dashboard, navigate to your repository
2. Go to **Settings** → **General**
3. Copy the **Repository Upload Token**

### 1.3. Add Token to GitHub Secrets

1. Go to GitHub repository: [https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM](https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM)
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add:
   - **Name:** `CODECOV_TOKEN`
   - **Secret:** (paste the token from step 1.2)
5. Click **Add secret**

### 1.4. Verify Coverage Reporting

After next PR:
- Check for Codecov comment on PR
- Verify coverage badge updates in README
- Review coverage report at: `https://codecov.io/gh/the-obstacle-is-the-way/antibody_training_pipeline_ESM`

---

## 2. Branch Protection Rules

### 2.1. Configure Main Branch Protection

1. Go to: **Settings** → **Branches**
2. Click **Add branch protection rule**
3. Configure:

**Branch name pattern:** `main`

**Protect matching branches:**

- ✅ **Require a pull request before merging**
  - Required approvals: `1`
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require review from Code Owners (optional)

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging

  **Status checks to require:**
  - `Code Quality (ruff, mypy, bandit)`
  - `Unit Tests (Python 3.12)`
  - `Integration Tests (Python 3.12)`
  - `CI Pipeline Success`
  - `Test Development Container` (Docker)
  - `Test Production Container` (Docker)

- ✅ **Require conversation resolution before merging**

- ✅ **Require signed commits** (recommended, optional)

- ✅ **Require linear history** (optional, prevents merge commits)

- ✅ **Include administrators**

- ✅ **Restrict who can push to matching branches**
  - Allow only via pull requests

4. Click **Create** or **Save changes**

### 2.2. Verify Branch Protection

1. Try pushing directly to `main` (should fail)
2. Create a test PR and verify required checks appear
3. Try merging without approvals (should be blocked)

---

## 3. Workflow Verification

### 3.1. Test CI Pipeline

**Trigger:** Push to any branch or create PR

**Expected behavior:**
- 4 jobs run: quality, test-unit, test-integration, security
- All must pass for PR to be mergeable
- Coverage report posted to PR (after Codecov setup)
- Artifacts uploaded (test results, security reports)

**Timeline:** ~15-20 minutes

**Manual test:**
```bash
# Create a test branch
git checkout -b test/ci-verification

# Make a trivial change
echo "# CI Test" >> docs/test.md

# Commit and push
git add docs/test.md
git commit -m "test: Verify CI pipeline works"
git push origin test/ci-verification

# Create PR on GitHub and watch CI run
```

### 3.2. Test Docker CI

**Trigger:** Push to any branch or PR

**Expected behavior:**
- 2 jobs run: test-dev, test-prod
- Docker builds complete (without loading images)
- Buildx cache used (speeds up subsequent builds)

**Timeline:** ~8-10 minutes

### 3.3. Test Benchmark Workflow (Manual Trigger)

**Trigger:** Workflow dispatch (manual)

**Manual test:**
```bash
# In GitHub UI:
# Actions → E2E Benchmarking & Novo Parity → Run workflow
# Select branch: main
# Run Harvey: false (for quick test)
# Click "Run workflow"
```

**Expected behavior:**
- E2E tests run (Jain, Shehata, optionally Harvey)
- Results posted to workflow summary
- Artifacts uploaded with benchmark outputs
- Issue created on failure

**Timeline:** ~120 minutes (full), ~15 minutes (without Harvey)

**Auto-triggers:**
- Weekly (Sunday 2am UTC)
- On release tags (v*)

### 3.4. Test Dependency Management (Manual Trigger)

**Trigger:** Workflow dispatch (manual)

**Manual test:**
```bash
# In GitHub UI:
# Actions → Dependency Management & Security → Run workflow
# Select branch: main
# Click "Run workflow"
```

**Expected behavior:**
- Dependencies updated via `uv sync --upgrade`
- Tests run to validate updates
- PR created if dependencies changed
- Security audit runs and reports vulnerabilities

**Timeline:** ~30 minutes

**Auto-triggers:**
- Weekly (Monday 6am UTC)

---

## 4. Workflow Schedule Summary

| Workflow | Trigger | Frequency | Duration | Purpose |
|----------|---------|-----------|----------|---------|
| **CI Pipeline** | Push, PR | Every commit | ~20 min | Quality gates + tests |
| **Docker CI** | Push, PR | Every commit | ~10 min | Verify Dockerfiles build |
| **E2E Benchmarks** | Schedule, Manual, Tags | Weekly (Sun 2am UTC) | ~120 min | Validate Novo parity |
| **Quick Parity** | PR to main | On PR to main | ~15 min | Fast parity check |
| **Dependency Updates** | Schedule, Manual | Weekly (Mon 6am UTC) | ~30 min | Auto-update deps |
| **Security Audit** | Schedule, Manual | Weekly (Mon 6am UTC) | ~10 min | Vulnerability scanning |

---

## 5. Cost Monitoring

### 5.1. GitHub Actions Minutes

**Free Tier:** 2000 minutes/month

**Expected usage:**
- CI per PR: ~20 min × 20 PR/month = **400 min**
- Weekly benchmarks: 120 min × 4 = **480 min**
- Weekly deps: 30 min × 4 = **120 min**
- **Total:** ~1000 min/month ✅ **50% of limit**

### 5.2. Monitor Usage

View usage:
1. Go to: **Settings** → **Billing and plans** → **Plans and usage**
2. Check **Actions** usage
3. Alert if approaching 80% of limit

**Optimization tips if approaching limit:**
- Skip Harvey benchmark more frequently
- Reduce PR frequency
- Cache more aggressively

---

## 6. Troubleshooting

### 6.1. CI Fails on Codecov Upload

**Symptom:** `codecov/codecov-action` fails with 401 Unauthorized

**Fix:**
1. Verify `CODECOV_TOKEN` is set in GitHub Secrets
2. Re-generate token from Codecov dashboard
3. Update secret in GitHub

### 6.2. Benchmark Workflow Fails with Disk Space Error

**Symptom:** "No space left on device"

**Fix:**
1. Reduce ESM model cache size
2. Clean up artifacts more aggressively
3. Skip Harvey benchmark if needed

### 6.3. Dependency Update PR Not Created

**Symptom:** Workflow runs but no PR appears

**Fix:**
1. Check workflow logs for errors
2. Verify `GITHUB_TOKEN` has `contents: write` and `pull-requests: write` permissions
3. Ensure `peter-evans/create-pull-request` action is up to date

### 6.4. Branch Protection Blocks Admin Push

**Symptom:** Cannot push to main even as admin

**Fix:**
1. Temporarily disable "Include administrators" in branch protection
2. Make emergency fix
3. Re-enable protection

**Better:** Always use PRs, even for admins

---

## 7. Next Steps (Future Phases)

### Phase 4: Release Automation (Not Yet Implemented)

**Trigger:** Tag push (e.g., `v2.1.0`)

**Would automate:**
- Version validation
- Changelog generation
- PyPI publishing
- GitHub Release creation
- Docker image publishing (requires self-hosted runner)

**Estimated effort:** 2-3 days

### Phase 5: Documentation Deployment (Not Yet Implemented)

**Trigger:** Push to main

**Would automate:**
- MkDocs build
- Link checking
- Deploy to GitHub Pages

**Estimated effort:** 1-2 days

### Phase 6: PR Automation (Not Yet Implemented)

**Trigger:** PR open/update

**Would automate:**
- Auto-labeling (based on changed files)
- Conventional commits validation
- PR size labeling
- Auto-assignment

**Estimated effort:** 1 day

---

## 8. Maintenance

### Weekly

- [ ] Review benchmark results (Sunday)
- [ ] Review dependency update PRs (Monday)
- [ ] Check security audit results (Monday)

### Monthly

- [ ] Review GitHub Actions minutes usage
- [ ] Check for workflow failures
- [ ] Update this guide if workflows change

### Quarterly

- [ ] Review branch protection rules
- [ ] Audit GitHub Secrets (rotate if needed)
- [ ] Update CI/CD roadmap

---

## 9. Support

**Issues with CI/CD:**
- Create issue: [New Issue](https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM/issues/new)
- Label: `ci-cd`, `infrastructure`
- Assign: @the-obstacle-is-the-way

**Questions:**
- Check workflow logs first
- Review this guide
- Open discussion in repository

---

## 10. Reference Links

### Documentation
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Codecov Docs](https://docs.codecov.com/)
- [pytest Docs](https://docs.pytest.org/)

### Workflows
- [CI Pipeline](.github/workflows/ci.yml)
- [Docker CI](.github/workflows/docker-ci.yml)
- [Benchmarks](.github/workflows/benchmark.yml)
- [Dependencies](.github/workflows/dependencies.yml)

### Related Docs
- [CICD_SPEC.md](../docs/CICD_SPEC.md) - Full specification
- [README.md](../README.md) - Project overview
- [USAGE.md](../USAGE.md) - User guide

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** Phase 1-3 Complete
**Maintainer:** @the-obstacle-is-the-way
