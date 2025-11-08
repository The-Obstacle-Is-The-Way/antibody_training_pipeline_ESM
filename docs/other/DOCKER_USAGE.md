# Docker Usage Guide

**Status:** Ready for Testing
**Date:** 2025-11-07
**Prerequisites:** Docker Desktop installed and running

---

## Quick Start

### 1. Start Docker Desktop

Make sure Docker Desktop is running:
- **macOS/Windows:** Launch Docker Desktop app
- **Linux:** `sudo systemctl start docker`

Verify Docker is running:
```bash
docker --version
docker-compose --version
```

### 2. Build Development Container

```bash
# Build dev container (first time ~5-10 min, includes running tests)
docker-compose build dev
```

This will:
1. Download `python:3.12-slim` base image (~50MB)
2. Install `uv` package manager
3. Install all dependencies via `uv sync` (~200MB)
4. Copy source code
5. Run test suite (372 tests, ~53s)
6. Cache everything for fast rebuilds

### 3. Run Tests

```bash
# Run full test suite (372 tests)
docker-compose run dev pytest tests/

# Run only unit tests (300 tests, faster)
docker-compose run dev pytest tests/unit/

# Run with coverage
docker-compose run dev pytest tests/ --cov=src/antibody_training_esm --cov-report=term
```

### 4. Interactive Development

```bash
# Drop into bash shell
docker-compose run dev bash

# Inside container, you have access to:
pytest tests/                      # Run tests
antibody-train --help              # Training CLI
antibody-test --help               # Testing CLI
antibody-preprocess --help         # Preprocessing CLI
python -m antibody_training_esm.cli.train --help  # Direct module access
```

### 5. Hot Reload Development

The container mounts `./src` and `./tests` as volumes, so code changes on your host machine are immediately reflected in the container:

```bash
# Terminal 1: Run container in watch mode
docker-compose run dev bash

# Inside container, run tests in watch mode
pytest tests/ --watch  # (if you install pytest-watch)

# Terminal 2: Edit code on host machine
# Changes are instantly available in container
```

---

## Common Workflows

### Train Model on Mock Data

```bash
docker-compose run dev antibody-train \
    --dataset boughter \
    --output /app/data/model.pkl \
    --config /app/configs/default.yaml
```

### Test Trained Model

```bash
docker-compose run dev antibody-test \
    --model /app/data/model.pkl \
    --dataset jain \
    --device cpu
```

### Run Preprocessing

```bash
docker-compose run dev antibody-preprocess \
    --dataset jain
```

### Run Linting & Type Checking

```bash
# Ruff linting
docker-compose run dev ruff check src/ tests/

# Ruff formatting
docker-compose run dev ruff format src/ tests/

# Mypy type checking
docker-compose run dev mypy src/
```

---

## Troubleshooting

### Build Fails: "Cannot connect to Docker daemon"

**Problem:** Docker Desktop isn't running.

**Solution:** Start Docker Desktop app and wait for it to fully start (you'll see a green icon).

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

---

## Performance Tips

### Speed Up Builds

```bash
# Use BuildKit (faster, better caching)
export DOCKER_BUILDKIT=1
docker-compose build dev

# Multi-core builds (if you have multiple cores)
docker-compose build dev --parallel
```

### Reduce Image Size

Development image is intentionally verbose for debugging. For production:
```bash
# Build production image (coming in Phase 3)
docker build -f Dockerfile.prod -t antibody-training-prod:latest .
```

### Cache Model Weights

To avoid re-downloading ESM model weights (~650MB) on every container:
```bash
# Create named volume for HuggingFace cache
docker volume create hf-cache

# Mount volume in docker-compose.yml
volumes:
  - hf-cache:/app/.cache/huggingface
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
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
      run: docker-compose run dev pytest tests/ --cov=src/antibody_training_esm --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
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

## Architecture Details

### What's in the Container?

```
/app/
├── .venv/                      # Virtual environment (from uv sync)
│   ├── bin/                    # Installed commands (pytest, antibody-train, etc.)
│   └── lib/python3.12/         # Installed packages
├── src/                        # Source code (mounted from host)
│   └── antibody_training_esm/
├── tests/                      # Tests (mounted from host)
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

### Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./src` | `/app/src` | Hot reload source code |
| `./tests` | `/app/tests` | Hot reload tests |
| `./data` | `/app/data` | Persist trained models |

---

## Next Steps

1. **Phase 1 Complete:** Development container is ready for testing.
2. **Phase 2 (Optional):** Add ESM model weight caching to production image.
3. **Phase 3:** Build production container with frozen dependencies.
4. **Phase 4:** Integrate with CI/CD (GitHub Actions).

---

## FAQ

### Q: Can I use this for production deployments?

**A:** Not yet. This is the **development** container. Use `Dockerfile.prod` (Phase 3) for production deployments with model weight caching and frozen dependencies.

### Q: Why does the first build take so long?

**A:** The first build:
1. Downloads Python 3.12 base image (~50MB)
2. Installs uv (~10MB)
3. Installs all dependencies (~200MB, including PyTorch)
4. Runs full test suite (372 tests, ~53s)

Subsequent builds are MUCH faster due to Docker layer caching.

### Q: Can I use this on Windows?

**A:** Yes, but you need Docker Desktop for Windows. Install from https://www.docker.com/products/docker-desktop

### Q: Do I need to install Python on my host machine?

**A:** No! Docker provides a completely isolated Python 3.12 environment. You only need Docker Desktop.

---

**Author:** Claude Code
**Date:** 2025-11-07
**Status:** Ready for Testing (Phase 1 Complete)
