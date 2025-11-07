# Docker Deployment Plan

**Status:** Planning
**Date:** 2025-11-07
**Author:** Claude Code
**Philosophy:** Reproducibility, portability, and ease of deployment

---

## Executive Summary

This document outlines the Docker containerization strategy for the antibody non-specificity prediction pipeline. Docker is **NOT YAGNI** for this project - it provides critical benefits for research reproducibility, collaboration, and potential deployment to platforms like HuggingFace Spaces.

**Goals:**
- ✅ Lock down Python version + dependencies for bit-for-bit reproducibility
- ✅ Eliminate "works on my machine" issues
- ✅ Enable easy deployment to HuggingFace Spaces or other platforms
- ✅ Provide clean environment proof (no sys.path hacks needed)
- ✅ Optional: Pre-cache ESM model weights (~650MB) for faster startup
- ✅ Support CI/CD workflows (GitHub Actions)

---

## Why Docker? (Not YAGNI)

### 1. Research Reproducibility
**Problem:** Scientific results must be reproducible years later on different hardware/OS.
**Solution:** Docker locks down:
- Python version (3.12 - matches pyproject.toml:10)
- All dependencies with exact versions (via `uv.lock`)
- System libraries (transformers, CUDA drivers if needed)
- Model weights (ESM-1v checkpoint)

### 2. HuggingFace Integration
**Problem:** If published to HuggingFace Spaces, platform expects Docker or Gradio app.
**Solution:** Pre-built Docker image can deploy directly to Spaces without modification.

### 3. Collaboration
**Problem:** New contributors spend hours debugging environment setup.
**Solution:** `docker-compose up` gets them running in <5 minutes.

### 4. Clean Environment Validation
**Problem:** Need to prove package works without local hacks (sys.path, editable installs).
**Solution:** Docker builds from scratch, installs package via `uv sync`, validates tests pass.

### 5. CI/CD Readiness
**Problem:** GitHub Actions needs consistent, fast test environment.
**Solution:** Pre-built Docker image with cached dependencies = faster CI runs.

---

## Architecture

### Two-Tier Strategy

#### Tier 1: Development Container (Fast Iteration)
**Purpose:** Local development, testing, debugging
**Image:** `antibody-training-dev:latest`
**Features:**
- Installs package in editable mode (`uv sync`)
- Mounts local source code as volume (hot reload)
- No model weights cached (downloads on first run)
- Smaller image size (~1.5GB)

**Use cases:**
- Running tests locally: `docker-compose run dev pytest`
- Interactive development: `docker-compose run dev bash`
- Training on small datasets

#### Tier 2: Production Container (Reproducibility)
**Purpose:** Deployment, published results, long-term archival
**Image:** `antibody-training-prod:latest`
**Features:**
- Installs package from built wheel (non-editable)
- Bakes in ESM model weights (~650MB)
- Frozen dependency versions from `uv.lock`
- Includes preprocessed canonical datasets (optional)
- Larger image size (~3-4GB)

**Use cases:**
- Reproducing paper results
- Deploying to HuggingFace Spaces
- CI/CD for releases
- Long-term archival (Docker Hub, GitHub Container Registry)

---

## Implementation Roadmap

### Phase 1: Basic Dockerfile (Development)
**Goal:** Get a working dev container with minimal features

**Deliverables:**
1. `Dockerfile.dev` - Development container
2. `docker-compose.yml` - Easy local orchestration
3. `.dockerignore` - Exclude unnecessary files
4. `docs/DOCKER_USAGE.md` - User guide

**Dockerfile.dev Structure:**
```dockerfile
FROM python:3.12-slim

# Install uv package manager
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
# uv sync installs to .venv/ and includes editable install automatically
RUN uv sync

# Copy source code (invalidates cache on code changes)
COPY src/ ./src/
COPY tests/ ./tests/

# Export .venv/bin to PATH so installed commands are available
ENV PATH="/app/.venv/bin:$PATH"

# Run tests to verify build (use 'not e2e' to skip E2E tests)
RUN pytest tests/ -m "not e2e" --tb=short -q

CMD ["bash"]
```

**docker-compose.yml Structure:**
```yaml
version: '3.8'

services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./src:/app/src  # Hot reload source code
      - ./tests:/app/tests  # Hot reload tests
      - ./data:/app/data  # Mount data directory
    environment:
      - PYTHONUNBUFFERED=1
      - HF_HOME=/app/.cache/huggingface  # Cache model downloads
    command: bash
```

**Validation:**
- `docker-compose build dev` succeeds (builds and runs 372 tests)
- `docker-compose run dev pytest tests/` passes 372 tests
- `docker-compose run dev antibody-train --help` works (PATH includes .venv/bin)
- `docker-compose run dev bash` drops into shell with all commands available

---

### Phase 2: Model Caching (Optional)
**Goal:** Pre-download ESM model weights to avoid first-run delays

**Strategy:**
```dockerfile
# Download ESM model during build (cached layer)
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('facebook/esm1v_t33_650M_UR90S_1'); \
    AutoTokenizer.from_pretrained('facebook/esm1v_t33_650M_UR90S_1')"
```

**Trade-offs:**
- ✅ Faster first run (no 650MB download)
- ✅ Works offline
- ❌ Larger image size (~2GB → ~3GB)
- ❌ Slower builds (downloads model every time Dockerfile changes)

**Recommendation:** Add this as a separate `Dockerfile.prod` for production builds.

---

### Phase 3: Production Container
**Goal:** Frozen, reproducible environment for publishing results

**Dockerfile.prod Structure:**
```dockerfile
FROM python:3.12-slim

# Install system dependencies (if needed for sklearn/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy locked dependencies
COPY pyproject.toml uv.lock ./

# Install exact versions (no updates, includes editable install)
RUN uv sync --frozen

# Export .venv/bin to PATH for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code (editable install will pick this up)
COPY src/ ./src/

# Pre-download ESM model weights
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('facebook/esm1v_t33_650M_UR90S_1'); \
    AutoTokenizer.from_pretrained('facebook/esm1v_t33_650M_UR90S_1')"

# Optional: Copy preprocessed datasets
# COPY data/processed/ ./data/processed/

# Set entrypoint (now available via PATH)
ENTRYPOINT ["antibody-train"]
CMD ["--help"]
```

**Validation:**
- Image builds successfully
- `docker run antibody-training-prod --help` works
- Model inference works without downloading weights

---

### Phase 4: CI/CD Integration
**Goal:** Use Docker in GitHub Actions for consistent test environment

**.github/workflows/test.yml:**
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build dev container
      run: docker-compose build dev

    - name: Run tests
      run: docker-compose run dev pytest tests/ -m "not e2e" --cov=src/antibody_training_esm --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

**Benefits:**
- Consistent test environment (no "works on my machine" in CI)
- Faster CI runs (cached Docker layers)
- Same container for local dev and CI

---

## File Structure

```
antibody_training_pipeline_ESM/
├── Dockerfile.dev              # Development container
├── Dockerfile.prod             # Production container (frozen deps, model cached)
├── docker-compose.yml          # Easy local orchestration
├── .dockerignore               # Exclude .git, __pycache__, etc.
├── docs/
│   ├── DOCKER_DEPLOYMENT.md    # This document (architecture, rationale)
│   └── DOCKER_USAGE.md         # User guide (how to build/run/debug)
├── .github/
│   └── workflows/
│       └── docker-ci.yml       # CI workflow using Docker
└── scripts/
    └── docker-entrypoint.sh    # Optional entrypoint script for prod container
```

---

## Usage Examples

### Local Development
```bash
# Build dev container
docker-compose build dev

# Run tests
docker-compose run dev pytest tests/

# Interactive shell
docker-compose run dev bash

# Train model on mock data
docker-compose run dev antibody-train --dataset boughter --output /app/data/model.pkl
```

### Production Build
```bash
# Build production container
docker build -f Dockerfile.prod -t antibody-training-prod:1.0 .

# Run training pipeline
docker run -v $(pwd)/data:/app/data antibody-training-prod:1.0 \
    --dataset boughter --output /app/data/model.pkl

# Test model
docker run -v $(pwd)/data:/app/data antibody-training-prod:1.0 \
    antibody-test --model /app/data/model.pkl --dataset jain
```

### HuggingFace Spaces Deployment
```bash
# Tag for HuggingFace Container Registry
docker tag antibody-training-prod:1.0 registry.huggingface.co/USERNAME/antibody-training:latest

# Push to HuggingFace
docker push registry.huggingface.co/USERNAME/antibody-training:latest
```

---

## Maintenance Plan

### Regular Updates
1. **Monthly:** Rebuild prod container with latest `uv.lock` (security patches)
2. **Per Release:** Tag prod container with version (e.g., `v1.0.0`, `v1.1.0`)
3. **Continuous:** Dev container rebuilt automatically in CI on every push

### Image Registry Strategy
- **Development images:** Keep only `latest` tag
- **Production images:** Tag with semantic versions (`1.0.0`, `1.1.0`, etc.)
- **Storage:** Push to GitHub Container Registry (ghcr.io) for free, unlimited storage

### Cleanup
```bash
# Remove old images
docker image prune -a --filter "until=30d"

# Remove dangling images
docker image prune
```

---

## Security Considerations

### 1. Base Image Selection
- Use official `python:3.12-slim` (matches pyproject.toml:10 requirement)
- Regularly update base image for security patches
- Consider `python:3.12-slim-bookworm` for latest Debian base

### 2. Dependency Scanning
- Run `docker scan` before pushing images
- Use GitHub Dependabot for dependency updates
- Pin all versions in `uv.lock` (no floating dependencies)

### 3. Secrets Management
- Never bake secrets into images (API keys, passwords)
- Use environment variables or mounted secrets
- For HuggingFace: Use `HF_TOKEN` env var, not hardcoded

### 4. Multi-Stage Builds
- Consider multi-stage builds to reduce final image size
- Build stage: Install build tools, compile dependencies
- Runtime stage: Copy only necessary artifacts

---

## Cost Analysis

### Storage (GitHub Container Registry)
- Dev image: ~1.5GB × 1 tag = **1.5GB**
- Prod images: ~3GB × 5 versions = **15GB**
- Total: **~16.5GB** (free on GitHub)

### Build Time
- Dev container: ~5 minutes (without model caching)
- Prod container: ~15 minutes (with model caching)
- CI runtime: ~3 minutes per test run (cached layers)

### Compute
- Local development: Negligible (Docker overhead ~5%)
- CI: ~$0 (GitHub Actions free tier sufficient)
- HuggingFace Spaces: Depends on usage (free tier available)

---

## Alternatives Considered

### ❌ Alternative 1: Virtual Environments Only (uv/venv)
**Why rejected:** Doesn't solve system-level reproducibility (Python version, OS, system libs).

### ❌ Alternative 2: Conda/Mamba
**Why rejected:** Heavier than Docker, less portable, not standard for deployment.

### ❌ Alternative 3: Singularity/Apptainer
**Why considered:** Popular in HPC environments, better for multi-user systems.
**Why deferred:** Docker is more widely supported (HuggingFace, GitHub, cloud platforms). Consider later if deploying to HPC clusters.

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `Dockerfile.dev` builds without errors
- [ ] `docker-compose run dev pytest` passes 372 tests
- [ ] `docker-compose run dev bash` provides interactive shell
- [ ] Source code hot-reloads when mounted as volume
- [ ] Documentation (`DOCKER_USAGE.md`) exists
- [ ] PATH includes `/app/.venv/bin` so all commands work

### Phase 2 Complete When:
- [ ] ESM model weights cached in production image
- [ ] First run doesn't trigger 650MB download
- [ ] Image size ≤4GB

### Phase 3 Complete When:
- [ ] Production container builds with frozen dependencies
- [ ] Wheel installation (non-editable) works
- [ ] Tagged versions pushed to GitHub Container Registry
- [ ] Example deployment to HuggingFace Spaces documented

### Phase 4 Complete When:
- [ ] GitHub Actions workflow uses Docker
- [ ] CI runs complete in <5 minutes
- [ ] Coverage reports upload successfully

---

## FAQ

### Q: Why not use Poetry instead of uv?
**A:** `uv` is faster, simpler, and already in use for this project. Docker works with any Python package manager.

### Q: Should we use `uv run` instead of modifying PATH?
**A:** Both approaches work:

**Option 1: Export PATH (recommended for Docker):**
```dockerfile
ENV PATH="/app/.venv/bin:$PATH"
CMD ["pytest", "tests/"]
```
✅ Pro: Standard Docker pattern, works with ENTRYPOINT/CMD
✅ Pro: Interactive shells (`bash`) work naturally
❌ Con: Less explicit about virtualenv usage

**Option 2: Use `uv run` prefix (alternative):**
```dockerfile
# Don't export PATH
CMD ["uv", "run", "pytest", "tests/"]
```
✅ Pro: Explicit about using virtualenv
✅ Pro: Matches local `uv run` workflow
❌ Con: Awkward with ENTRYPOINT/CMD patterns
❌ Con: Interactive shells require `uv run bash` (weird)

**Decision:** Use `ENV PATH` approach in Dockerfiles for consistency with standard container patterns.

### Q: Should we use Docker for local development?
**A:** Optional. If `uv` works well on your machine, keep using it. Docker is for reproducibility and deployment, not mandatory for dev.

### Q: How do we handle GPU support?
**A:** For GPU training:
1. Use `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` as base image
2. Install Python 3.12 manually (CUDA images don't include it)
3. Install CUDA-compatible PyTorch via `uv sync`
4. Run with `docker run --gpus all ...`

Currently CPU-only is sufficient for testing and small datasets.

### Q: Can we run the full test suite in Docker?
**A:** Yes, but E2E tests requiring real datasets will still be skipped (same as local environment). Docker doesn't magically provide missing data.

---

## Next Steps

1. **Immediate:** Senior review and approval of this plan
2. **Phase 1:** Implement `Dockerfile.dev` + `docker-compose.yml`
3. **Validation:** Verify tests pass in container
4. **Phase 2-4:** Iteratively add model caching, prod container, CI integration

**Timeline Estimate:**
- Phase 1: 2-4 hours
- Phase 2: 1-2 hours
- Phase 3: 2-3 hours
- Phase 4: 1-2 hours
- **Total:** ~1 day of focused work

---

**Author:** Claude Code
**Date:** 2025-11-07
**Status:** Awaiting Senior Approval

---

## Revision History

### 2025-11-07 - Critical Corrections (Post-Review)
**Summary:** Fixed Python version mismatch, virtual environment handling, and premature success claims.

**Critical Corrections Made:**

1. **Python Version Fix (lines 29, 101, 182, 350, 443)**
   - ❌ **Before:** `FROM python:3.11-slim`
   - ✅ **After:** `FROM python:3.12-slim`
   - **Reason:** pyproject.toml:10 requires `>=3.12`, ruff configured for py312
   - **Impact:** Prevents `uv sync --frozen` failures due to incompatible Python version

2. **Virtual Environment Activation (lines 120-121, 200-201)**
   - ❌ **Before:** Commands run directly after `uv sync` (would fail)
   - ✅ **After:** Added `ENV PATH="/app/.venv/bin:$PATH"` to Dockerfile
   - **Reason:** `uv sync` installs to `.venv/`, not system PATH (README.md:79-82)
   - **Impact:** Enables `pytest`, `antibody-train`, etc. to work without `uv run` prefix

3. **Removed Redundant Install Step (lines 113-114 comment)**
   - ❌ **Before:** `RUN uv pip install -e .` after `uv sync`
   - ✅ **After:** Removed (uv sync does editable install automatically)
   - **Reason:** `uv sync` already installs package in editable mode per pyproject.toml

4. **Success Criteria Accuracy (lines 407-412)**
   - ❌ **Before:** Phase 1 marked `[x]` complete
   - ✅ **After:** All boxes marked `[ ]` (not yet implemented)
   - **Reason:** No Docker files exist in repo (`ls` confirmed)
   - **Added:** PATH requirement to success checklist

5. **GPU Support Clarification (line 443)**
   - ✅ **Added:** Note that CUDA base images don't include Python 3.12
   - **Impact:** Prevents confusion when switching to GPU support

**Validation:** All corrections verified against:
- `pyproject.toml:10` (Python ≥3.12 requirement)
- `pyproject.toml:59` (Ruff py312 target)
- `README.md:79-82` (uv workflow with .venv activation)
- Filesystem check (no Docker files present)

**Status:** Document now accurately reflects implementation requirements.

---

### 2025-11-07 - Initial Plan
- Outlined two-tier Docker strategy (dev + prod)
- Justified Docker as NOT YAGNI for this project
- Defined 4-phase implementation roadmap
- Estimated costs, timelines, and success criteria
