# Test Suite Senior Review Checklist

This document is the single checklist we use during every review cycle (end of each phase **and** before merge) to keep the antibody non-specificity test suite aligned with the master plan. Work through each section; do not mark a phase complete until every box that applies is checked.

---

## 1. Scope & Phase Gates

- [ ] Current phase deliverables match the plan (Phase 1: core unit tests, Phase 2: datasets, Phase 3: integrations, Phase 4: CLI/E2E).
- [ ] No code paths outside the declared phase were touched unless part of a documented defect fix.
- [ ] Regression gaps identified in previous phases (e.g., `datasets/base.py` coverage) are captured in the current phase backlog.

## 2. Critical API Contracts (SSOT)

Verify tests only target the public behaviors listed below—never the implementation details.

- [ ] `BinaryClassifier` tests only feed/prefer `np.ndarray` embeddings (`shape=(n,1280)`), respect `predict(..., assay_type)` thresholds (`ELISA=0.5`, `PSR=0.5495`), and never pass raw sequences.
- [ ] `ESMEmbeddingExtractor` usage sticks to `embed_sequence` / `extract_batch_embeddings`, mocks **HuggingFace transformers** (not `esm.pretrained`), and asserts validation/placeholder behaviors exactly as implemented.
- [ ] Dataset loaders call `load_data(...)` without phantom `fragment` args, expect `VH_sequence`/`VL_sequence` columns, and use `create_fragment_csvs(df, suffix="")` only after `output_dir` is set on init.
- [ ] Fragment helpers always use `get_fragment_types()` (never `get_supported_fragments()`).
- [ ] TODO: Add trainer/loaders/CLI API contracts at the start of Phases 3 and 4 after auditing those modules.

## 3. Mocking & Fixture Policy

- [ ] Only I/O boundaries are mocked (model downloads, filesystem, network). Domain logic, pandas transforms, sklearn estimators remain real.
- [ ] Transformers mocks patch `transformers.AutoModel/AutoTokenizer.from_pretrained` and return deterministic tensors.
- [ ] Temporary files use `tmp_path`/`tmp_path_factory`—no writes into repo fixtures or shared paths that could clash under `pytest -n auto`.
- [ ] Fixtures produce valid biological sequences; invalid data is introduced intentionally inside tests, not in baseline fixtures.

## 4. Test Design Rules

- [ ] Every test follows AAA and asserts observable outcomes (no spying on privates or checking specific class attributes).
- [ ] Edge cases are covered: empty inputs, invalid sequences, single-sample runs, threshold boundaries, missing files.
- [ ] No duplicate assertions for the same behavior; prefer parametrization/fixtures over copy-paste.
- [ ] Integration/E2E tests verify workflows (dataset → embeddings → trainer → CLI) without over-mocking intermediate steps.

## 5. Coverage Targets (per module)

| Module / Area                                  | Target | Status |
| ---------------------------------------------- | ------ | ------ |
| `core/classifier.py`                           | ≥90%   |        |
| `core/embeddings.py`                           | ≥85%   |        |
| `core/trainer.py`                              | ≥85%   |        |
| `datasets/*.py` (each concrete loader)         | ≥80%   |        |
| `datasets/base.py` utilities (annotate/fragment)| ≥80% (current 72.68%; gap covers ANARCI-dependent code lines 274-326. Remediation: mock ANARCI/annotation flows in Phases 3–4 when integration tests land.) |        |
| `data/loaders.py`                              | ≥80%   |        |
| `cli/*.py`                                     | ≥70%   |        |
| Integration suites (`tests/integration/…`)     | ≥70% branch coverage through dataset→embedding→trainer stack (measure with `pytest --cov-branch`) | |
| End-to-end suites (`tests/e2e/…`)              | Smoke-level behavioral coverage | |

> **Action:** If a target cannot be met in the current phase, document the gap, rationale, and remediation plan before sign-off.

## 6. Quality Gates & Tooling

- [ ] `pytest` (phase-specific selection) passes locally with `uv run`.
- [ ] `ruff check`, `ruff format`, and `mypy` pass on touched paths (or repo-wide when feasible).
- [ ] Coverage command (`pytest --cov=src/antibody_training_esm …`) runs without sandbox issues; artifacts (HTML/text) are archived when helpful.
- [ ] CI workflow definition covers unit + integration suites, applies coverage gate (`--cov-fail-under=80`), and uploads reports (Codecov).

## 7. Data & Artifact Hygiene

- [ ] No large model downloads or external network calls happen inside tests; all heavy assets are mocked.
- [ ] Test artifacts (tmp CSV/Excel files, generated fragments, models) are created under `tmp_path` and cleaned up automatically.
- [ ] Mock datasets remain small (≈10–20 rows) and balanced enough to exercise filtering logic without slowing the suite.
- [ ] Logged warnings/errors in tests are intentional and asserted when meaningful; otherwise logging noise is muted.

## 8. Phase Exit Checklist

- [ ] All findings from the senior review (e.g., fragile assertions, fixture pollution) are fixed or ticketed with owners and deadlines.
- [ ] Documentation (plan + checklist) reflects the actual state—no stale instructions.
- [ ] Next phase entry criteria are satisfied (e.g., Phase 3 can start only after dataset coverage + hygiene items are green).
- [x] Phase 2 exit complete (datasets >80% with documented base.py exception); proceed to Phase 3 integration tests.
- [ ] Final “ready” comment records test counts, runtime, coverage percentages, and outstanding risks.

Use this checklist as a living gate. After a phase passes, return to senior-review mode and keep the repo clean—no additional docs unless instructed.
