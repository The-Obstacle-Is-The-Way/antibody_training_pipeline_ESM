# Security Remediation Plan

_Last updated: 2025-11-08_

## Objectives
- Eliminate all outstanding Bandit (code-level) and dependency scanner (pip-audit, safety) findings.
- Replace unsafe serialization patterns (`pickle`) with integrity-checked, schema-aware formats.
- Pin all external model downloads to immutable revisions and verify artifacts at runtime.
- Raise CI to fail on any new security regressions (code or dependency).

## Findings Snapshot

| Source      | Issue Type                               | Count | Notes |
|-------------|------------------------------------------|-------|-------|
| Bandit      | Pickle import/usage (B403/B301)          | 6     | `cli/test.py`, `core/trainer.py`, `data/loaders.py`. |
| Bandit      | HuggingFace unpinned downloads (B615)    | 3     | `core/embeddings.py`, `data/loaders.py`. |
| pip-audit   | Dependency CVEs                          | 24    | keras, transformers, torch, authlib, etc. |
| Safety      | Dependency CVEs                          | 27    | Overlaps with pip-audit + xmltodict, starlette. |

## Remediation Streams

### 1. Artifact Serialization Hardening
**Goal:** Remove Bandit B301/B403 warnings by avoiding unsafe `pickle.load`/`dump` calls and by verifying artifact integrity.

**Plan:**
1. Introduce `antibody_training_esm/utils/artifacts.py` with helpers:
   - `write_model_artifact(model_metadata: dict, weights: np.ndarray, path: Path)` storing metadata as JSON + numpy `.npz`.
   - `read_model_artifact(path: Path, expected_schema: ArtifactSchema)` validating SHA256 and schema version.
   - Equivalent helpers for embedding caches and CLI cached embeddings using `np.savez_compressed` + JSON manifest.
2. Update `core/trainer.py`:
   - Replace pickle caching with `(cache.json + cache.npz)` pair; manifest includes hash + dataset name.
   - Persist sklearn classifier coefficients/intercepts explicitly in JSON (no pickled estimator).
3. Update `cli/test.py`:
   - Load classifiers via new artifact reader; refuse unsigned/unknown schema files.
   - Cache embeddings via new helper; include hash validation before reuse.
4. Update `data/loaders.py`:
   - Replace `pickle` for preprocessed data with `msgpack` or JSON + numpy `.npz`.
5. Delete all `import pickle` statements in production code; keep a single well-documented exception (if absolutely necessary) guarded by integrity checks and `# nosec` justification.
6. Migrate existing `.pkl` artifacts (models, cache) via conversion script to prevent breaking operators.

**Verification:** Bandit B301/B403 suppressed to zero; regression tests + manual load/save roundtrip tests.

### 2. External Model & Dataset Integrity
**Goal:** Resolve Bandit B615 by pinning downloads and optionally verifying checksums.

**Plan:**
1. Extend config YAMLs to accept `hf_revision` and `hf_sha256`.
2. Update `ESMEmbeddingExtractor` and `load_hf_dataset` to require explicit revision:
   ```python
   AutoModel.from_pretrained(model_name, revision=hf_revision)
   AutoTokenizer.from_pretrained(model_name, revision=hf_revision)
   load_dataset(..., revision=hf_revision)
   ```
3. Add optional checksum verification post-download using Hugging Face file metadata (compare against `hf_sha256`).
4. Document procedure for bumping revisions (record hash, update config, rerun CI).

**Verification:** Bandit B615 eliminated; smoke test to ensure models still load.

### 3. Dependency Vulnerability Remediation
**Goal:** Bring all first-party dependencies to patched versions or formally document exceptions.

**Plan:**
1. Group CVEs by package:
   - **Transformers 4.52.4** → upgrade to ≥4.53.0 (verify compatibility with ESM-1v + torch 2.7.1).
   - **Torch 2.7.1** → upgrade to 2.8.0 (validate CUDA/mps paths, update wheels cache).
   - **Keras 3.10.0** → upgrade to ≥3.12.0 or remove if unused.
   - **Authlib, langchain, langgraph, xmltodict, starlette, brotli, python-socketio, pip** → upgrade to fixed versions listed in pip-audit output.
2. Update `pyproject.toml` + `uv.lock`, run `uv sync`, and re-run full test suite.
3. For vendored/experimental packages not on PyPI (`big-mood-detector`, etc.), add SBOM entry clarifying they are internal and not part of deployable artifacts; mark as ignored in safety policy.
4. If any upstream fix is incompatible, open an issue + temporary waiver with mitigation notes (e.g., not exposed to untrusted input).

**Verification:** `uv run pip-audit` and `uv run safety scan` report zero vulnerabilities (or documented waivers).

### 4. CI Security Enforcement
**Goal:** Fail CI whenever new security issues appear.

**Plan:**
1. Once Streams 1–3 are complete, set `continue-on-error: false` for Bandit, pip-audit, and safety jobs in `.github/workflows/ci.yml`.
2. Add artifact uploads for new serialization-manifest tests + SBOM.
3. Gate merges on CI summary job.

### 5. Documentation & Migration
**Plan:**
1. Update `USAGE.md` with instructions for creating signed model artifacts and hashed caches.
2. Provide `scripts/migrate_artifacts.py` to convert existing `.pkl` assets.
3. Record new operational runbooks (how to refresh HF revision, how to regenerate manifests).

## Execution Checklist

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Design artifact schema + helpers | TBD | ☐ | JSON manifest + NPZ payload. |
| Replace pickle usages in `core/trainer.py` | TBD | ☐ | Includes classifier persistence. |
| Replace pickle usages in `cli/test.py` | TBD | ☐ | Model + embedding caches. |
| Replace pickle usages in `data/loaders.py` | TBD | ☐ | Preprocessed dataset persistence. |
| Add HF revision + checksum enforcement | TBD | ☐ | Config + runtime validation. |
| Upgrade vulnerable dependencies | TBD | ☐ | Track via uv.lock diff. |
| Update CI to enforce security jobs | TBD | ☐ | After fixes land. |
| Update docs + migration script | TBD | ☐ | USAGE.md + scripts/. |

## Verification Matrix

| Verification Step | Tools | Pass Criteria |
|-------------------|-------|---------------|
| Static security scan | `uv run bandit -r src/antibody_training_esm` | 0 issues |
| Dependency audit | `uv run pip-audit`, `uv run safety scan` | 0 vulnerabilities (or documented waivers) |
| Serialization regression tests | `pytest tests/unit/core/test_trainer.py::TestArtifactPersistence` | Round-trip equality |
| End-to-end pipeline | `pytest tests/e2e` | Existing coverage unchanged |
| CI run on PR | GitHub Actions | All jobs succeed, including security gates |

## Open Questions / Risks
- **Artifact size**: Switching to JSON+NPZ may increase storage overhead; mitigate via compression.
- **Compatibility**: Upgrading torch/transformers could impact availability on Apple Silicon; verify wheel support.
- **Operational migration**: Need coordination to convert historic `.pkl` models without breaking analysts’ workflows.

## Next Steps
1. Implement artifact serialization helpers and convert all call sites.
2. Land HF revision pinning + config updates.
3. Bump dependencies and regenerate `uv.lock`.
4. Re-run scanners → if clean, enforce CI gates.
