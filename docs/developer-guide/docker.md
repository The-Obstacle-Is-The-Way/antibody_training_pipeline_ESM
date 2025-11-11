# Docker

**Target Audience:** Developers using Docker for development and deployment

**Purpose:** Run the pipeline in reproducible, portable containers for local development, testing, and deployment

---

## When to Use This Guide

Use this guide if you're:
- ✅ **Setting up development environment** (first-time setup with Docker)
- ✅ **Running tests in containers** (isolated environment)
- ✅ **Deploying to production** (HuggingFace Spaces, cloud platforms)
- ✅ **Ensuring reproducibility** (lock down Python version, dependencies, models)
- ✅ **Troubleshooting Docker issues** (build failures, performance)

---

## Related Documentation

- **Workflow:** [Development Workflow](development-workflow.md) - Non-Docker development commands
- **Architecture:** [Architecture](architecture.md) - System design
- **Security:** [Security Guide](security.md) - Security best practices

---

## Why Docker?

### Research Reproducibility

**Problem:** Scientific results must be reproducible years later on different hardware/OS.

**Solution:** Docker locks down:
- Python version (3.12 matches `pyproject.toml`)
- All dependencies with exact versions (via `uv.lock`)
- System libraries (transformers, CUDA drivers if needed)
- Model weights (ESM-1v checkpoint)

### Collaboration

**Problem:** New contributors spend hours debugging environment setup.

**Solution:** `docker-compose up` gets them running in <5 minutes.

### Deployment Ready

**Problem:** Need to deploy to platforms like HuggingFace Spaces or cloud services.

**Solution:** Pre-built Docker image deploys directly without modification.

### Clean Environment Validation

**Problem:** Need to prove package works without local hacks (sys.path, editable installs).

**Solution:** Docker builds from scratch, installs package via `uv sync`, validates tests pass.

---

## Container Types

### Development Container

**Image:** `antibody-training-dev:latest`

**Purpose:** Local development, testing, debugging

**Features:**
- Installs package in editable mode (`uv sync`)
- Mounts local source code as volume (hot reload)
- No model weights cached (downloads on first run)
- Smaller image size (~1.5GB)

**Use cases:**
- Running tests locally
- Interactive development
- Training on small datasets

### Production Container

**Image:** `antibody-training-prod:latest`

**Purpose:** Deployment, published results, long-term archival

**Features:**
- Installs package from built wheel (non-editable)
- Bakes in ESM model weights (~650MB)
- Frozen dependency versions from `uv.lock`
- Larger image size (~3-4GB)

**Use cases:**
- Reproducing paper results
- Deploying to HuggingFace Spaces
- CI/CD for releases
- Long-term archival

---

## Quick Start (Development)

### Prerequisites

**Install Docker Desktop:**
- **macOS/Windows:** Download from https://www.docker.com/products/docker-desktop
- **Linux:** Install via package manager (`apt install docker.io docker-compose`)

**Verify installation:**
```bash
docker --version
docker-compose --version
```

### Build Development Container

```bash
# Build dev container (first time ~5-10 min)
docker-compose build dev
```

This will:
1. Download `python:3.12-slim` base image (~50MB)
2. Install `uv` package manager
3. Install all dependencies via `uv sync` (~200MB)
4. Copy source code
5. Run test suite (validates build)
6. Cache everything for fast rebuilds

### Run Tests

```bash
# Run full test suite
docker-compose run dev pytest tests/

# Run only unit tests (faster)
docker-compose run dev pytest tests/unit/

# Run with coverage
docker-compose run dev pytest tests/ --cov=src/antibody_training_esm --cov-report=term
```

### Interactive Development

```bash
# Drop into bash shell
docker-compose run dev bash

# Inside container, you have access to:
pytest tests/                      # Run tests
antibody-train --help              # Training CLI
antibody-test --help               # Testing CLI
antibody-preprocess --help         # Preprocessing CLI
```

### Hot Reload Development

The container mounts `./src` and `./tests` as volumes, so code changes on your host machine are immediately reflected in the container:

```bash
# Terminal 1: Run container
docker-compose run dev bash

# Terminal 2: Edit code on host machine
# Changes are instantly available in container
```

---

## Common Workflows

### Train Model

```bash
# Use default Hydra config
docker-compose run dev antibody-train

# OR override parameters
docker-compose run dev antibody-train \
    hardware.device=cpu training.batch_size=16
```

### Test Trained Model

```bash
docker-compose run dev antibody-test \
    --model models/model.pkl \
    --data test_datasets/jain/fragments/VH_only_jain.csv
```

### Run Preprocessing

```bash
# Preprocess specific dataset
python preprocessing/jain/step1_convert_excel_to_csv.py
```

### Run Code Quality Checks

```bash
# Ruff linting
docker-compose run dev ruff check src/ tests/

# Ruff formatting
docker-compose run dev ruff format src/ tests/

# Mypy type checking
docker-compose run dev mypy src/
```

---

## Production Deployment

### Build Production Container

**Create `Dockerfile.prod`:**
```dockerfile
FROM python:3.12-slim

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy locked dependencies
COPY pyproject.toml uv.lock ./

# Install exact versions (frozen)
RUN uv sync --frozen

# Export .venv/bin to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/

# Pre-download ESM model weights (~650MB)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('facebook/esm1v_t33_650M_UR90S_1'); \
    AutoTokenizer.from_pretrained('facebook/esm1v_t33_650M_UR90S_1')"

# Set entrypoint
ENTRYPOINT ["antibody-train"]
CMD ["--help"]
```

**Build:**
```bash
docker build -f Dockerfile.prod -t antibody-training-prod:1.0 .
```

### Run Production Container

```bash
# Run training pipeline (uses default Hydra config)
docker run -v $(pwd)/data:/app/data antibody-training-prod:1.0

# OR with parameter overrides
docker run -v $(pwd)/data:/app/data antibody-training-prod:1.0 \
    hardware.device=cpu

# Test model
docker run -v $(pwd)/data:/app/data antibody-training-prod:1.0 \
    antibody-test --model /app/data/model.pkl \
    --data /app/test_datasets/jain/fragments/VH_only_jain.csv
```

### Deploy to HuggingFace Spaces

```bash
# Tag for HuggingFace Container Registry
docker tag antibody-training-prod:1.0 \
    registry.huggingface.co/USERNAME/antibody-training:latest

# Push to HuggingFace
docker push registry.huggingface.co/USERNAME/antibody-training:latest
```

---

## CI/CD Integration

### GitHub Actions Example

**`.github/workflows/docker-test.yml`:**
```yaml
name: Docker Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build dev container
      run: docker-compose build dev

    - name: Run tests
      run: docker-compose run dev pytest tests/ --cov=src/antibody_training_esm --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Push to GitHub Container Registry

```bash
# Tag for GHCR
docker tag antibody-training-prod:1.0 ghcr.io/USERNAME/antibody-training:1.0
docker tag antibody-training-prod:1.0 ghcr.io/USERNAME/antibody-training:latest

# Push to GHCR
docker push ghcr.io/USERNAME/antibody-training:1.0
docker push ghcr.io/USERNAME/antibody-training:latest
```

---

## Troubleshooting

### Build Fails: "Cannot connect to Docker daemon"

**Problem:** Docker Desktop isn't running.

**Solution:** Start Docker Desktop app and wait for it to fully start (green icon).

### Build Fails: "uv sync" errors

**Problem:** Dependency resolution issues or corrupted `uv.lock`.

**Solution:**
```bash
# On host machine, regenerate lock file
uv lock --upgrade

# Rebuild container
docker-compose build dev --no-cache
```

### Tests Fail During Build

**Problem:** Code changes broke tests.

**Solution:**
```bash
# Run tests locally first
uv run pytest tests/

# Fix failing tests, then rebuild
docker-compose build dev
```

### Container Runs Out of Space

**Problem:** Docker images and volumes fill up disk.

**Solution:**
```bash
# Remove old images
docker image prune -a

# Remove all stopped containers
docker container prune

# Remove unused volumes
docker volume prune
```

### "Module not found" errors in container

**Problem:** PATH doesn't include `.venv/bin`.

**Solution:** This should be automatic via `ENV PATH="/app/.venv/bin:$PATH"` in Dockerfile. If not working:
```bash
# Inside container, manually check PATH
echo $PATH
# Should include: /app/.venv/bin

# If missing, export manually
export PATH="/app/.venv/bin:$PATH"
```

### Slow first-time model download

**Problem:** ESM model weights (~650MB) download on first inference.

**Solution:** Use production container with pre-cached weights (see "Build Production Container" above).

---

## Best Practices

### 1. Cache Model Weights for Production

**Avoid re-downloading ESM model (~650MB) on every container:**

```dockerfile
# In Dockerfile.prod
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('facebook/esm1v_t33_650M_UR90S_1'); \
    AutoTokenizer.from_pretrained('facebook/esm1v_t33_650M_UR90S_1')"
```

**Or use named volume:**
```yaml
# In docker-compose.yml
volumes:
  - hf-cache:/app/.cache/huggingface

volumes:
  hf-cache:
```

### 2. Use BuildKit for Faster Builds

```bash
# Enable BuildKit (faster, better caching)
export DOCKER_BUILDKIT=1
docker-compose build dev
```

### 3. Multi-Stage Builds for Smaller Images

```dockerfile
# Build stage
FROM python:3.12-slim as builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY src/ /app/src/
```

### 4. Pin Base Image Versions

```dockerfile
# ✅ GOOD: Pinned version
FROM python:3.12.1-slim

# ❌ AVOID: Floating tag
FROM python:3.12-slim
```

### 5. Regular Security Scans

```bash
# Scan image for vulnerabilities
docker scan antibody-training-prod:1.0

# Or use GitHub Dependabot for automated scans
```

### 6. Never Bake Secrets into Images

```bash
# ❌ WRONG: Secret in Dockerfile
ENV HF_TOKEN=hf_xxxxxxxxxxxxx

# ✅ RIGHT: Secret via environment variable
docker run -e HF_TOKEN=$HF_TOKEN antibody-training-prod:1.0
```

---

## Container Architecture

### What's in the Container?

```
/app/
├── .venv/                      # Virtual environment (from uv sync)
│   ├── bin/                    # Installed commands (pytest, antibody-train, etc.)
│   └── lib/python3.12/         # Installed packages
├── src/                        # Source code (mounted from host in dev)
│   └── antibody_training_esm/
├── tests/                      # Tests (mounted from host in dev)
├── data/                       # Data directory (mounted from host)
├── pyproject.toml              # Project metadata
└── uv.lock                     # Locked dependencies
```

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PATH` | `/app/.venv/bin:$PATH` | Makes installed commands available |
| `PYTHONUNBUFFERED` | `1` | Force Python to print output immediately |
| `HF_HOME` | `/app/.cache/huggingface` | Cache HuggingFace model downloads |

### Volume Mounts (Development)

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./src` | `/app/src` | Hot reload source code |
| `./tests` | `/app/tests` | Hot reload tests |
| `./data` | `/app/data` | Persist trained models |

---

## Performance Tips

### Build Performance

**Optimize layer caching:**
```dockerfile
# Copy dependency files first (changes less frequently)
COPY pyproject.toml uv.lock ./
RUN uv sync

# Copy source code last (changes more frequently)
COPY src/ ./src/
```

### Runtime Performance

**Use named volumes for model cache:**
```yaml
volumes:
  - hf-cache:/app/.cache/huggingface
```

**Limit memory/CPU if needed:**
```yaml
services:
  dev:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
```

---

## Cleanup

### Stop All Containers

```bash
docker-compose down
```

### Remove Development Image

```bash
docker rmi antibody_training_pipeline_esm-dev
```

### Full Cleanup (Nuclear Option)

```bash
# Remove ALL Docker images, containers, volumes
docker system prune -a --volumes

# WARNING: This will delete EVERYTHING Docker-related on your machine!
```

---

## FAQ

### Q: Can I use this for production deployments?

**A:** Yes, use `Dockerfile.prod` for production deployments with:
- Frozen dependencies (`uv sync --frozen`)
- Pre-cached ESM model weights
- Non-editable package install

### Q: Why does the first build take so long?

**A:** The first build:
1. Downloads Python 3.12 base image (~50MB)
2. Installs uv (~10MB)
3. Installs all dependencies (~200MB, including PyTorch)
4. Runs full test suite (validates build)

Subsequent builds are MUCH faster due to Docker layer caching.

### Q: Can I use this on Windows?

**A:** Yes, install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop

### Q: Do I need to install Python on my host machine?

**A:** No! Docker provides a completely isolated Python 3.12 environment. You only need Docker Desktop.

### Q: How do I update dependencies?

**A:** Update `uv.lock` on host, then rebuild:
```bash
# Host machine
uv lock --upgrade

# Rebuild container
docker-compose build dev --no-cache
```

### Q: Can I use GPU acceleration in containers?

**A:** Yes, with NVIDIA Docker runtime:
```yaml
# docker-compose.yml
services:
  dev:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
