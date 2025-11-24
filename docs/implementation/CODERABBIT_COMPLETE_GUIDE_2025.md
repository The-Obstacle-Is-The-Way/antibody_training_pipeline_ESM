# CodeRabbit Complete Guide 2025

> **Your AI Senior Code Reviewer: From PR Reviews to CLI Pre-Commit Analysis**
>
> Last Updated: 2025-11-23 | Ray's Learning Guide 🔥

---

## Table of Contents

1. [What is CodeRabbit?](#what-is-coderabbit)
2. [CodeRabbit vs Sourcery](#coderabbit-vs-sourcery)
3. [Installation & Setup](#installation--setup)
4. [CLI Usage](#cli-usage)
5. [Configuration Deep Dive](#configuration-deep-dive)
6. [Integration with AI Agents](#integration-with-ai-agents)
7. [Best Practices for This Repo](#best-practices-for-this-repo)
8. [Workflows & Examples](#workflows--examples)
9. [Troubleshooting](#troubleshooting)
10. [Resources](#resources)

---

## What is CodeRabbit?

CodeRabbit is an **AI-powered code review assistant** that reviews code like a senior developer who knows your entire codebase. Think of it as having a tireless senior engineer who:

- **Reviews PRs automatically** on GitHub/GitLab/Bitbucket
- **Works in your CLI** to catch issues BEFORE you commit
- **Understands context** by mapping dependencies and downstream effects
- **Learns your preferences** and applies them across all future reviews
- **Runs 40+ tools** (linters, security analyzers, performance checkers) every review

### Key Differentiators (2025)

1. **Context-Aware Intelligence**
   - Maps dependencies across your codebase (not just changed files)
   - Understands how changes affect downstream consumers
   - Learns from your review patterns

2. **Multi-Platform Support**
   - **PR Reviews**: GitHub, GitHub Enterprise, Bitbucket
   - **CLI**: MacOS/Linux terminal integration
   - **IDE**: Free VSCode extension (also works in Cursor, Windsurf)

3. **AI Agent Integration**
   - Works with Claude Code, Cursor CLI, Gemini CLI
   - Enables autonomous generate-review-iterate cycles
   - Special `--prompt-only` mode for agent consumption

4. **New Features (2025)**
   - **Model Context Protocol (MCP)** client (planned)
   - **Pre-merge checks** with auto-generated tests
   - **Multi-source context**: Fetches feature requirements, engineering docs

---

## CodeRabbit vs Sourcery

You mentioned using **Sourcery** (formerly "Sorcery AI") - here's how they compare:

| Feature | CodeRabbit | Sourcery |
|---------|------------|----------|
| **Review Style** | Comprehensive, detailed (can be verbose) | Focused, less noisy, actionable |
| **Platform** | GitHub, GitLab, Bitbucket | GitHub only |
| **CLI** | ✅ Full CLI support (`coderabbit`) | ❌ No standalone CLI |
| **IDE Integration** | VSCode only (free extension) | VSCode, Cursor, Windsurf, JetBrains |
| **Static Analysis** | 40+ integrated tools | Focused on complexity reduction |
| **AI Agent Integration** | ✅ `--prompt-only` mode for agents | ❌ Not optimized for agents |
| **Learning** | Remembers repo-specific preferences | Learns from review interactions |
| **Pricing (Free)** | 1 review/hour (rate limited) | Unlimited for public repos |
| **Pricing (Paid)** | Pro: 5 reviews/hr, Lite: 1 review/hr | Team/Pro plans required for private repos |
| **Best For** | Teams wanting comprehensive, context-aware reviews | Teams seeking focused, less verbose reviews |
| **Unique Strength** | Cross-file dependency analysis | Complexity reduction focus |

### Which Should You Use?

**Use CodeRabbit if:**
- You want CLI-based pre-commit reviews
- You're integrating with AI coding agents (Claude Code, Cursor)
- You need multi-platform support (GitHub + GitLab + Bitbucket)
- You want deep context-aware analysis

**Use Sourcery if:**
- You find CodeRabbit too noisy/verbose
- You want strong JetBrains IDE integration
- You prioritize code complexity reduction
- You prefer focused, actionable feedback

**Use BOTH:**
- CodeRabbit CLI for pre-commit checks
- Sourcery IDE extension for real-time feedback while coding

---

## Installation & Setup

### 1. Install CodeRabbit CLI

```bash
# One-line installation (macOS/Linux only)
curl -fsSL https://cli.coderabbit.ai/install.sh | sh

# Restart shell or reload config
source ~/.zshrc  # or ~/.bashrc
```

**Platform Support:**
- ✅ Apple (Intel and Apple Silicon)
- ✅ Linux
- ❌ Windows (not supported)

### 2. Authenticate

```bash
# Full command
coderabbit auth login

# Short alias
cr auth login
```

This opens a browser for GitHub authentication.

### 3. Verify Installation

```bash
# Check version
coderabbit --version

# Run first review (interactive mode)
coderabbit
```

### 4. Install PR Review Integration

For automatic PR reviews on GitHub:

1. Visit [coderabbit.ai](https://www.coderabbit.ai/)
2. Connect your GitHub account
3. Install the CodeRabbit GitHub App on your repo
4. CodeRabbit will now comment on all new PRs

---

## CLI Usage

### Basic Commands

```bash
# Interactive review (default)
coderabbit

# Plain text output (detailed feedback)
coderabbit --plain

# Prompt-only mode (for AI agents like Claude Code)
coderabbit --prompt-only

# Review specific files
coderabbit src/core/embeddings.py

# Review against base branch
coderabbit --base main
```

### Review Modes Explained

| Mode | Use Case | Output |
|------|----------|--------|
| **Interactive** (default) | Manual code review | Full TUI with browsable findings |
| **Plain text** (`--plain`) | CI/CD logs, documentation | Detailed feedback with fix suggestions |
| **Prompt-only** (`--prompt-only`) | AI agents (Claude Code, Cursor) | Minimal JSON output optimized for agents |

### Configuration Options

```bash
# Review types
coderabbit --type all           # Review all changes (staged + unstaged)
coderabbit --type committed     # Review only committed changes
coderabbit --type uncommitted   # Review only uncommitted changes

# Custom config files (like .cursorrules, claude.md)
coderabbit --config .cursorrules --config CLAUDE.md

# Working directory
coderabbit --cwd /path/to/project

# Disable colors (for logs)
coderabbit --no-color
```

### Common Workflows

#### Pre-Commit Review (Manual)

```bash
# 1. Make changes to code
git add src/core/embeddings.py

# 2. Review staged changes
coderabbit --type committed

# 3. Fix issues
# (make edits based on feedback)

# 4. Commit
git commit -m "fix(core): address CodeRabbit feedback"
```

#### Pre-Commit Review (AI Agent - Claude Code)

```bash
# 1. Let Claude Code generate changes
# (Claude creates embeddings_amplify.py)

# 2. Run CodeRabbit in prompt-only mode
coderabbit --prompt-only > coderabbit_feedback.txt

# 3. Pass feedback to Claude Code
# (Claude reads coderabbit_feedback.txt and fixes issues)

# 4. Verify fixes
coderabbit --type uncommitted

# 5. Commit when clean
git add . && git commit -m "feat(core): add AMPLIFY extractor with CodeRabbit validation"
```

#### Full Repo Audit

```bash
# Review entire codebase
coderabbit --type all --base main

# Focus on critical paths
coderabbit src/antibody_training_esm/core/ --plain
```

---

## Configuration Deep Dive

CodeRabbit reads configuration from `.coderabbit.yaml` in your repo root.

### Minimal Configuration

```yaml
# .coderabbit.yaml
# yaml-language-server: $schema=https://coderabbit.ai/integrations/schema.v2.json

reviews:
  profile: chill  # or "assertive" for more feedback
  high_level_summary: true
  auto_review:
    enabled: true
    drafts: false  # Skip draft PRs
```

### Recommended Configuration for This Repo

```yaml
# .coderabbit.yaml
# yaml-language-server: $schema=https://coderabbit.ai/integrations/schema.v2.json

# General Settings
language: en-US
tone_instructions: |
  Professional, technical, concise. Focus on:
  - Type safety (mypy strict mode required)
  - Scientific correctness (antibody domain)
  - Performance (GPU memory management)
  - Security (pickle usage, input validation)

# Review Configuration
reviews:
  profile: assertive  # We want thorough feedback
  high_level_summary: true
  collapse_walkthrough: false
  request_changes_workflow: false

  auto_review:
    enabled: true
    auto_incremental_review: true
    drafts: false
    ignore_title_keywords:
      - "WIP"
      - "[skip-cr]"
    base_branches:
      - "dev"
      - "leroy-jenkins/full-send"

  # Path-Based Instructions
  path_instructions:
    - path: "src/antibody_training_esm/core/**/*.py"
      instructions: |
        CRITICAL: This is core ML pipeline code. Enforce:
        - 100% type coverage (mypy --strict)
        - Comprehensive error handling
        - GPU memory management (clear cache after batches)
        - Input validation (amino acid sequences only)
        - Scientific correctness (embedding dimensions, thresholds)

    - path: "src/antibody_training_esm/datasets/**/*.py"
      instructions: |
        Dataset loaders must:
        - Validate CSV structure on load
        - Check for required columns (sequence, label)
        - Handle missing values explicitly
        - Document expected file formats

    - path: "tests/**/*.py"
      instructions: |
        Test code must:
        - Use pytest markers (unit/integration/e2e)
        - Mock external dependencies (HuggingFace, GPU)
        - Have clear docstrings explaining test intent
        - Follow AAA pattern (Arrange, Act, Assert)

    - path: "preprocessing/**/*.py"
      instructions: |
        Preprocessing scripts are one-time ETL:
        - Validate input data before processing
        - Save intermediate outputs for debugging
        - Log progress for long-running operations
        - Document expected input/output formats

    - path: "**/*.yaml"
      instructions: |
        Hydra configs must:
        - Include comments explaining parameters
        - Use type hints in schema (when available)
        - Default to cross-platform values (device: auto)

    - path: "docs/**/*.md"
      instructions: |
        Documentation must:
        - Use code fences with language tags
        - Include real examples from the repo
        - Link to source code (file:line format)
        - Update "Last Updated" dates

  # Pre-Merge Checks
  pre_merge_checks:
    docstrings:
      enabled: true
      threshold: 70  # Match repo coverage requirement
    title_check:
      enabled: true  # Enforce conventional commits
    description_check:
      enabled: warning  # Warn if PR description is empty
    custom_checks:
      - name: "type-safety"
        instructions: |
          Verify all new functions have complete type annotations.
          Check that mypy --strict would pass on changed files.
      - name: "test-coverage"
        instructions: |
          Ensure new code has corresponding tests.
          Coverage should be ≥90% for new code, ≥70% overall.
      - name: "gpu-memory-management"
        instructions: |
          If code uses torch.cuda or torch.mps, verify:
          - Cache is cleared after batches
          - Memory is freed on exceptions
          - Device allocation is conditional

  # Tools Configuration
  tools:
    ruff:
      enabled: true
    mypy:
      enabled: true
    gitleaks:
      enabled: true
    yamllint:
      enabled: true
    markdownlint:
      enabled: true
    shellcheck:
      enabled: true
    bandit:
      enabled: true  # Security linter (pickle usage)

# Knowledge Base
knowledge_base:
  learnings_scope: local  # Repo-specific patterns only
  code_guidelines:
    - CLAUDE.md           # Claude Code instructions
    - .cursorrules        # Cursor instructions
    - USAGE.md            # Command references
  issue_integrations:
    github:
      enabled: true

# Chat Configuration (PR comments)
chat:
  auto_reply: true
  art: false  # No emojis (matches repo style)
```

### Configuration Sections Explained

#### 1. **Review Profile**

```yaml
reviews:
  profile: chill      # Fewer comments, focus on critical issues
  # OR
  profile: assertive  # More feedback, may be nitpicky
```

**Choose:**
- `chill`: Fast iteration, trust senior devs
- `assertive`: Learning team, catch everything

#### 2. **Auto Review Triggers**

```yaml
auto_review:
  enabled: true
  auto_incremental_review: true  # Review each push
  ignore_title_keywords:
    - "WIP"
    - "[skip-cr]"
    - "DRAFT"
  labels:
    - "needs-review"  # Only review PRs with this label
```

#### 3. **Path-Based Instructions**

Target specific file patterns:

```yaml
path_instructions:
  - path: "**/test_*.py"
    instructions: "Ensure tests use pytest markers and mock external deps"

  - path: "src/core/**"
    instructions: "Critical ML code - verify type safety and error handling"

  - path: "**/*.yaml"
    instructions: "Validate YAML syntax and Hydra compatibility"
```

**Glob Patterns:**
- `**/*.py` - All Python files
- `src/core/**` - Everything in src/core/
- `**/test_*.py` - Test files anywhere

#### 4. **Pre-Merge Checks**

Custom quality gates:

```yaml
pre_merge_checks:
  docstrings:
    enabled: true
    threshold: 80  # % of functions with docstrings

  title_check:
    enabled: true  # Validate PR title format

  custom_checks:
    - name: "breaking-changes"
      instructions: |
        Check if changes break public API.
        Require "BREAKING CHANGE" in commit message if so.
```

**Modes:**
- `off` - Disabled
- `warning` - Show warning, don't block merge
- `error` - Block merge until fixed

#### 5. **Tools Integration**

Enable static analyzers:

```yaml
tools:
  ruff:
    enabled: true
  mypy:
    enabled: true
  gitleaks:
    enabled: true   # Secret detection
  bandit:
    enabled: true   # Python security linter
  yamllint:
    enabled: true
  markdownlint:
    enabled: true
```

**Available Tools (40+):**
- **Python**: Ruff, Flake8, Pylint, Mypy, Bandit
- **JS/TS**: ESLint, Biome, Oxlint
- **Go**: golangci-lint
- **Rust**: Clippy
- **Security**: Gitleaks, Semgrep, OSV Scanner
- **Infrastructure**: Hadolint (Docker), Checkov (IaC)
- **Config**: YAMLlint, ShellCheck, HTMLHint

---

## Integration with AI Agents

CodeRabbit is designed to work seamlessly with AI coding agents like **Claude Code**, **Cursor CLI**, and **Gemini CLI**.

### Workflow: Generate → Review → Iterate

```bash
# 1. AI Agent generates code (e.g., Claude Code)
# Claude creates: src/core/embeddings_amplify.py

# 2. Run CodeRabbit in prompt-only mode
coderabbit --prompt-only --type uncommitted > review.json

# 3. AI Agent reads review.json and fixes issues
# Claude: "I see CodeRabbit found 3 issues, let me fix them..."

# 4. Verify fixes
coderabbit --type uncommitted

# 5. Commit when clean
git add . && git commit -m "feat(core): add AMPLIFY extractor (CodeRabbit validated)"
```

### Why `--prompt-only` Mode?

**Interactive Mode:**
```bash
$ coderabbit
📊 Found 5 issues:
  1. [HIGH] Missing type annotation on line 42
  2. [MED] Unused import on line 3
  ...
👉 Press 'j' to navigate, 'q' to quit
```

**Prompt-Only Mode:**
```bash
$ coderabbit --prompt-only
{
  "issues": [
    {"severity": "high", "line": 42, "message": "Missing type annotation", ...},
    {"severity": "medium", "line": 3, "message": "Unused import", ...}
  ]
}
```

The AI agent can parse JSON, but can't interact with TUI.

### CodeRabbit Auto-Detection

CodeRabbit automatically reads:
- `claude.md` (Claude Code instructions)
- `.cursorrules` (Cursor instructions)
- Custom team standards

Example:

**CLAUDE.md:**
```markdown
## Type Safety
- 100% type coverage enforced via mypy --strict
- All functions require complete type annotations
```

**CodeRabbit Review:**
```
❌ Missing return type annotation (line 42)
   Violates repo standard: "100% type coverage enforced"
```

### Limiting Iterations

**Problem:** AI agents can loop endlessly trying to fix issues.

**Solution:** Cap iterations in your agent script:

```python
# pseudocode for AI agent integration
max_iterations = 3
for i in range(max_iterations):
    result = subprocess.run(["coderabbit", "--prompt-only"], capture_output=True)
    issues = json.loads(result.stdout)

    if not issues:
        print("✅ CodeRabbit clean!")
        break

    # Pass issues to AI agent
    ai_agent.fix_issues(issues)

    if i == max_iterations - 1:
        print("⚠️  Max iterations reached, manual review required")
```

---

## Best Practices for This Repo

### 1. Pre-Commit Workflow

```bash
# Before committing ANY code:
make all              # format, lint, typecheck, test
coderabbit --plain    # CodeRabbit review

# Fix issues, then commit
git add .
git commit -m "fix(core): address linting + CodeRabbit feedback"
```

### 2. PR Review Workflow

```bash
# 1. Push branch
git push origin feature/amplify-integration

# 2. Create PR on GitHub
gh pr create --title "feat: Add AMPLIFY integration" --body "..."

# 3. Wait for CodeRabbit comment
# (CodeRabbit posts review within 1-2 minutes)

# 4. Address feedback
git add .
git commit -m "fix: address CodeRabbit feedback"
git push

# 5. CodeRabbit automatically reviews new push
```

### 3. Ignore False Positives

If CodeRabbit flags valid code:

```python
# Tell CodeRabbit to skip this
def complex_but_correct_function():
    # coderabbit:skip - This complexity is intentional for performance
    # (Complex but optimized code here)
    pass
```

### 4. Path-Based Rules

For this repo's structure:

```yaml
path_instructions:
  - path: "src/antibody_training_esm/core/**"
    instructions: "Critical ML code - enforce strict type safety"

  - path: "preprocessing/**"
    instructions: "One-time ETL scripts - prioritize clarity over optimization"

  - path: "tests/**"
    instructions: "Test code - verify pytest markers and mocking"
```

### 5. Custom Pre-Merge Checks

```yaml
pre_merge_checks:
  custom_checks:
    - name: "pickle-security"
      instructions: |
        If code uses pickle.load():
        - Verify input is from trusted local source only
        - Check threat model documented in SECURITY_REMEDIATION_PLAN.md
        - Ensure no untrusted network sources
```

---

## Workflows & Examples

### Example 1: Pre-Commit Review (Type Error)

```bash
$ git status
Modified: src/antibody_training_esm/core/embeddings_amplify.py

$ coderabbit --plain
╔════════════════════════════════════════════════════════════════╗
║ CodeRabbit Review: Found 2 issues                             ║
╚════════════════════════════════════════════════════════════════╝

📁 src/antibody_training_esm/core/embeddings_amplify.py

  ❌ [HIGH] Missing return type annotation (line 42)
     Function: embed_sequence
     Issue: Return type not specified

     Fix:
     - def embed_sequence(self, sequence: str):
     + def embed_sequence(self, sequence: str) -> np.ndarray:

  ⚠️  [MEDIUM] Unused import (line 5)
     Import: from typing import Optional

     Fix: Remove unused import

$ # Fix issues
$ vim src/antibody_training_esm/core/embeddings_amplify.py

$ coderabbit --plain
✅ No issues found!

$ git commit -m "feat(core): add AMPLIFYEmbeddingExtractor with type annotations"
```

### Example 2: AI Agent Integration (Claude Code)

**Scenario:** Claude Code implements AMPLIFY integration (Phase A)

```bash
# 1. Claude Code generates embeddings_amplify.py
$ ls src/antibody_training_esm/core/
embeddings.py
embeddings_amplify.py  # NEW

# 2. Run CodeRabbit for Claude
$ coderabbit --prompt-only --type uncommitted > /tmp/cr_review.json

# 3. Claude Code reads review
$ cat /tmp/cr_review.json
{
  "issues": [
    {
      "file": "src/antibody_training_esm/core/embeddings_amplify.py",
      "line": 28,
      "severity": "high",
      "message": "batch_size forced to 1 but no warning logged",
      "suggestion": "Add logger.warning() when batch_size > 1"
    }
  ]
}

# 4. Tell Claude Code to fix
User: "Claude, please address the CodeRabbit feedback in /tmp/cr_review.json"

# 5. Claude fixes and re-runs
$ coderabbit --prompt-only --type uncommitted
{"issues": []}

$ git add . && git commit -m "feat(core): add AMPLIFY extractor (CodeRabbit validated)"
```

### Example 3: Full Repo Audit

```bash
# Audit entire codebase
$ coderabbit --type all --base main --plain > audit_2025-11-23.txt

# Review results
$ cat audit_2025-11-23.txt
╔════════════════════════════════════════════════════════════════╗
║ CodeRabbit Audit: 47 files reviewed, 12 issues found          ║
╚════════════════════════════════════════════════════════════════╝

📁 src/antibody_training_esm/core/classifier.py
  ⚠️  [LOW] Docstring coverage: 65% (threshold: 70%)

📁 src/antibody_training_esm/datasets/harvey.py
  ❌ [HIGH] Unused variable 'df_temp' (line 102)

...

# Create GitHub issue from audit
$ gh issue create \
  --title "Code quality audit (2025-11-23)" \
  --body-file audit_2025-11-23.txt \
  --label "quality"
```

### Example 4: PR Review with Auto-Fix

```bash
# 1. Create PR
$ git push origin feature/amplify
$ gh pr create --title "feat: Add AMPLIFY integration"

# 2. CodeRabbit comments on PR:
"""
@coderabbitai

Found 3 issues:

1. Missing type annotation: embeddings_amplify.py:42
2. Unused import: embeddings_amplify.py:5
3. Docstring missing: embeddings_amplify.py:28

I can auto-fix issues 2-3. Reply with '@coderabbitai apply suggestions' to fix.
"""

# 3. Auto-fix via PR comment
User: "@coderabbitai apply suggestions"

# 4. CodeRabbit commits fixes
# (New commit appears: "fix: apply CodeRabbit suggestions")

# 5. Manual fix for issue 1
$ git pull
$ vim src/antibody_training_esm/core/embeddings_amplify.py
$ git add . && git commit -m "fix: add type annotation for embed_sequence"
$ git push
```

---

## Troubleshooting

### Issue 1: CLI Not Found After Installation

**Error:**
```bash
$ coderabbit
zsh: command not found: coderabbit
```

**Fix:**
```bash
# Reload shell config
source ~/.zshrc  # or ~/.bashrc

# Verify installation
which coderabbit
# Should output: /Users/ray/.local/bin/coderabbit (or similar)
```

### Issue 2: Authentication Failed

**Error:**
```bash
$ coderabbit
Error: Not authenticated. Please run 'coderabbit auth login'
```

**Fix:**
```bash
coderabbit auth login
# Opens browser for GitHub authentication
```

### Issue 3: Review Takes Too Long (>5 minutes)

**Cause:** CodeRabbit is analyzing large repo or many files.

**Fix:**
```bash
# Review only specific files
coderabbit src/antibody_training_esm/core/ --type uncommitted

# Use plain mode (faster)
coderabbit --plain

# Limit to committed changes only
coderabbit --type committed
```

### Issue 4: Too Many False Positives

**Cause:** CodeRabbit's "assertive" profile is too strict.

**Fix:**

**Option 1: Change profile**
```yaml
# .coderabbit.yaml
reviews:
  profile: chill  # Less nitpicky
```

**Option 2: Ignore specific rules**
```python
# In code
def intentionally_complex_function():
    # coderabbit:skip - Complexity is intentional for performance
    ...
```

**Option 3: Adjust path instructions**
```yaml
path_instructions:
  - path: "preprocessing/**"
    instructions: "ETL scripts - prioritize clarity, allow higher complexity"
```

### Issue 5: Review Doesn't Detect Issue

**Cause:** CodeRabbit might not have context about repo-specific rules.

**Fix:**

**Option 1: Add to CLAUDE.md**
```markdown
## Security
- NEVER load pickle files from untrusted sources
- Validate all user input with amino acid regex
```

**Option 2: Add custom pre-merge check**
```yaml
pre_merge_checks:
  custom_checks:
    - name: "pickle-security"
      instructions: "Flag any pickle.load() from non-local sources"
```

### Issue 6: Windows Support

**Error:**
```bash
This platform is not supported.
```

**Fix:**
CodeRabbit CLI **does not support Windows**. Alternatives:
- Use WSL2 (Windows Subsystem for Linux)
- Use CodeRabbit PR reviews only (no CLI needed)
- Use Sourcery IDE extension instead

---

## Resources

### Official Documentation
- [CodeRabbit Docs](https://docs.coderabbit.ai/) - Official documentation
- [CLI Reference](https://docs.coderabbit.ai/cli/overview) - CLI commands and usage
- [Configuration Reference](https://docs.coderabbit.ai/reference/configuration) - `.coderabbit.yaml` guide
- [GitHub - CodeRabbit Docs](https://github.com/coderabbitai/coderabbit-docs) - Documentation source

### Comparisons & Reviews
- [Qodo: Best CodeRabbit Alternatives](https://www.qodo.ai/blog/coderabbit-alternatives/) - Comparison with Sourcery, Greptile, others
- [Sourcery vs CodeRabbit](https://www.sourcery.ai/comparisons/coderabbit-alternative) - Direct comparison
- [BlueDot: AI Code Review Tools 2025](https://bluedot.org/blog/best-ai-code-review-tools-2025) - 8-tool comparison
- [DevTools Academy: State of AI Code Review 2025](https://www.devtoolsacademy.com/blog/state-of-ai-code-review-tools-2025/) - Industry overview

### News & Announcements
- [CodeRabbit Blog: 2025 AI Dev Tool Stack](https://www.coderabbit.ai/blog/2025-the-year-of-the-ai-dev-tool-tech-stack) - Vision for 2025
- [DevOps.com: CodeRabbit Adds CLI Support](https://devops.com/coderabbit-adds-cli-support-to-code-review-platform-based-on-ai/) - CLI launch announcement
- [SiliconANGLE: CodeRabbit $60M Funding](https://siliconangle.com/2025/09/16/coderabbit-gets-60m-fix-ai-generated-code-quality/) - Series B funding news

### Tutorials & Guides
- [Honra.io: CodeRabbit CLI Guide](https://www.honra.io/articles/coderabbit-just-launched-a-cli-tool) - Getting started
- [Charly Wargnier: CodeRabbit CLI Review](https://medium.com/@charly-wargnier/coderabbit-cli-makes-ai-code-reviews-effortless-6f3a5de78496) - Medium tutorial
- [Juanma Codes: Maximize Code Quality](https://juanma.codes/2025/09/25/coderabbit-cli-catch-issues-locally-before-you-open-a-pr/) - Developer workflow

### Support
- **Email**: sales@coderabbit.ai (enterprise support)
- **GitHub Issues**: [coderabbitai/coderabbit-docs](https://github.com/coderabbitai/coderabbit-docs/issues)
- **Pricing**: Free tier (1 review/hour), Pro (5 reviews/hour), Enterprise (custom)

---

## Quick Reference Card

```bash
# Installation
curl -fsSL https://cli.coderabbit.ai/install.sh | sh

# Authentication
coderabbit auth login

# Basic Usage
coderabbit                          # Interactive review
coderabbit --plain                  # Plain text output
coderabbit --prompt-only            # AI agent mode

# Review Types
coderabbit --type all               # All changes
coderabbit --type committed         # Staged only
coderabbit --type uncommitted       # Unstaged only

# Configuration
coderabbit --config CLAUDE.md       # Use custom config
coderabbit --base main              # Compare to main branch
coderabbit --cwd /path/to/repo      # Set working directory

# Common Workflows
make all && coderabbit --plain      # Pre-commit check
coderabbit --prompt-only > cr.json  # AI agent integration
coderabbit src/core/ --type uncommitted  # Targeted review
```

---

**Last Updated:** 2025-11-23
**Author:** Ray (learning CodeRabbit like a boss)
**Status:** Production-ready guide 🔥

---

## Appendix: Example .coderabbit.yaml for This Repo

```yaml
# .coderabbit.yaml
# Complete configuration for antibody_training_pipeline_ESM
# yaml-language-server: $schema=https://coderabbit.ai/integrations/schema.v2.json

language: en-US
early_access: true

tone_instructions: |
  Professional, technical, concise. Focus on:
  - Type safety (mypy strict mode)
  - Scientific correctness
  - GPU memory management
  - Security (pickle usage)

reviews:
  profile: assertive
  high_level_summary: true
  collapse_walkthrough: false
  request_changes_workflow: false

  auto_review:
    enabled: true
    auto_incremental_review: true
    drafts: false
    ignore_title_keywords: ["WIP", "[skip-cr]"]
    base_branches: ["dev", "leroy-jenkins/full-send"]

  path_instructions:
    - path: "src/antibody_training_esm/core/**/*.py"
      instructions: |
        CRITICAL ML PIPELINE CODE:
        - 100% type coverage (mypy --strict)
        - GPU memory management (clear cache)
        - Input validation (amino acids only)
        - Error handling with context

    - path: "tests/**/*.py"
      instructions: |
        TEST CODE:
        - Use pytest markers (unit/integration/e2e)
        - Mock HuggingFace/GPU dependencies
        - Clear docstrings (AAA pattern)

    - path: "preprocessing/**/*.py"
      instructions: |
        ETL SCRIPTS:
        - Validate input data
        - Save intermediate outputs
        - Log progress
        - Document formats

    - path: "**/*.yaml"
      instructions: |
        HYDRA CONFIGS:
        - Comment all parameters
        - Cross-platform defaults (device: auto)

  pre_merge_checks:
    docstrings:
      enabled: true
      threshold: 70
    title_check:
      enabled: true
    description_check:
      enabled: warning
    custom_checks:
      - name: "type-safety"
        instructions: "Verify mypy --strict compliance"
      - name: "test-coverage"
        instructions: "New code: ≥90%, Overall: ≥70%"
      - name: "pickle-security"
        instructions: "Flag pickle.load() from non-local sources"

  tools:
    ruff: {enabled: true}
    mypy: {enabled: true}
    gitleaks: {enabled: true}
    bandit: {enabled: true}
    yamllint: {enabled: true}
    markdownlint: {enabled: true}
    shellcheck: {enabled: true}

knowledge_base:
  learnings_scope: local
  code_guidelines: ["CLAUDE.md", ".cursorrules", "USAGE.md"]
  issue_integrations:
    github: {enabled: true}

chat:
  auto_reply: true
  art: false
```

**Boom.** 🔥 You're now a CodeRabbit expert.
