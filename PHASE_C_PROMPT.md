# Phase C Execution Prompt

## Context
You are working on the **antibody_training_pipeline_ESM** repository. Phases A (Quick Wins) and B (Path Centralization) are complete and merged to `leroy-jenkins/full-send`.

You are now starting **Phase C: File Splitting** - splitting 4 massive files (>500 lines) into modular components.

## Your Task

**Read these files IN ORDER:**
1. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM/PHASE_C_AGENT_GUIDANCE.md` - **CRITICAL RULES & WORKFLOW**
2. `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM/PHASE_C_FILE_SPLITTING.md` - **TECHNICAL DETAILS**

**Then execute Phase C following the guidance EXACTLY.**

## Critical Requirements

⚠️ **READ THE GUIDANCE DOCUMENT FIRST** - It contains lessons learned from Phase A/B mistakes.

**DO NOT:**
- ❌ Skip the pre-flight checklist
- ❌ Remove mock fixtures from tests
- ❌ Change test logic during refactoring
- ❌ Continue after test failures
- ❌ Make multiple changes before testing

**DO:**
- ✅ Follow the 5-step workflow for EACH file split
- ✅ Run `make test` after EVERY file split
- ✅ Commit incrementally (1 commit per file split)
- ✅ Preserve ALL type hints, docstrings, functionality
- ✅ Stop and debug if ANY test fails

## Expected Outcome

**When Phase C is complete:**
- ✅ 4 files split into ~10 modular components
- ✅ All tests passing (513 passed, 20 deselected)
- ✅ All quality gates passing (format, lint, typecheck)
- ✅ 4 incremental commits with clear messages
- ✅ Ready to merge to `dev`

## Starting Command

```bash
# 1. Read the guidance document
cat /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM/PHASE_C_AGENT_GUIDANCE.md

# 2. Execute pre-flight checklist
git checkout dev
git pull origin dev
git status
make test 2>&1 | tail -10

# 3. Create Phase C branch
git checkout -b claude/refactor-phase-c

# 4. Proceed with Task C1 (Split trainer.py)
```

## Success Metrics

- Time: 3-4 hours
- Files split: 4/4
- Tests passing: 513/513
- Commits: 4 (incremental)
- Regressions: 0

---

**You are an AI coding agent. Begin Phase C now. Read PHASE_C_AGENT_GUIDANCE.md first, then execute the plan step-by-step.**
