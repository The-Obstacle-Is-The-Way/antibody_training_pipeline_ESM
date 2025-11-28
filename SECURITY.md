# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to: [jj@novamindnyc.com](mailto:jj@novamindnyc.com)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity (critical: ASAP, high: 30 days, medium: 90 days)

### Scope

This security policy applies to:
- The `antibody_training_pipeline_ESM` Python package
- CLI tools (`antibody-train`, `antibody-test`, `antibody-predict`)
- Docker images published to GitHub Container Registry

### Out of Scope

- Vulnerabilities in dependencies (report to upstream maintainers)
- Issues in forked/modified versions
- Social engineering attacks

## Security Considerations

### Data Handling

This pipeline processes antibody sequence data. Users should:
- Never commit sensitive/proprietary sequences to version control
- Use `.gitignore` patterns for local data files
- Review `data/` contents before pushing

### Model Artifacts

Trained models are saved as pickle files (`.pkl`). These are:
- Generated locally by trusted code
- **Never** load pickle files from untrusted sources
- For production deployment, consider migrating to safer formats (ONNX, JSON+NPZ)

### Dependencies

- Dependencies are pinned in `uv.lock` for reproducibility
- Security audits run via `uv run pip-audit` in CI
- Bandit static analysis enforced (0 findings required)

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities (with permission).
