# GitHub Configuration and Automation for ai-mlops-project

This document outlines all GitHub-related activities and configurations performed on the `ai-mlops-project` repository to establish a robust and secure CI/CD pipeline with automation and developer-friendly workflows.

---

## ✅ Repository Structure and Branching Strategy

- **Main Branch**: Production-ready code (`main`) is protected. No direct pushes allowed.
- **Development Branch**: Feature testing and pre-production work (`dev`).
- **Feature/Hotfix Branches**: Each feature, bug fix, or refactor is done in separate branches and merged into `dev` via PR.

---

## ✅ Pull Request Workflow

### Structure

- All PRs must go through checks before merging.
- Protected branches (`main`, `dev`) enforce CI checks.

### Templates

- Created four PR templates:
  - `add-feature.md`
  - `fix-bug.md`
  - `code-refactor.md`
  - `pull_request_template.md` (default)

### Location

```
.github/PULL_REQUEST_TEMPLATE/
.github/ (default)
```

### Usage

- Default template: `pull_request_template.md`
- Others can be used via URL:
  ```
  https://github.com/Impesud/ai-mlops-project/compare/main...dev?template=add-feature.md
  ```

---

## ✅ GitHub Actions: CI/CD

### Main Workflow

File: `.github/workflows/ci-cd.yml`

- Branches: Triggers on `push` and `pull_request` to `main` and `dev`
- Conditional logic:
  - On `dev`: run `make test-dev`
  - On `main`: run `make test-prod`
- MLflow artifacts uploaded only for push events

### Jobs

- `build-and-test`
- `docker-build-and-push` (main only)
- `model-smoke-test` (conditional on model changes)
- `semantic-release` (automatic versioning and changelog updates)

---

## ✅ Semantic Release Configuration

- Triggered only on `main` pushes
- Automatically:
  - Bumps version based on conventional commits
  - Updates `CHANGELOG.md`
  - Creates release tags

### Fix Applied

Added required permissions to `ci-cd.yml`:

```yaml
permissions:
  contents: write
  issues: write
  pull-requests: write
```

---

## ✅ Git Ignore & Data Handling

### .gitignore

- Avoid committing intermediate `.parquet`, `.crc`, and other ML-generated files
- Preserved folder structure via `.gitkeep`, re-added when Spark overwrites them

### Safe Local Workflow

- Prevent `git add .` from staging untracked data files by using `.gitignore` rules
- Use `git restore` to avoid committing unintended changes

---

## ✅ Makefile Utilities

Added:

```makefile
.PHONY: test-dev test-prod

test-dev:
	pytest tests/ --env=dev

test-prod:
	pytest tests/ --env=prod
```

---

## 🔐 Branch Protection Rules

- `main` and `dev` are protected.
- All pushes must go through PRs.
- Required check: `build-and-test`

---

## 📌 Notes

- GitHub only supports **one default PR template**.
- To use multiple templates, users must use **URL parameters**.

---

## 🧹 Housekeeping

- Cleaned up old local & remote branches via:

```bash
git branch -d old-branch-name
git push origin --delete old-branch-name
```

---

## 🔧 To Do Next

- Automate changelog commits with personal access token if needed for protected branches
- Evaluate if more fine-grained environment support is needed in tests

---

**Documentation updated: June 2025 (Github Workflow Update)**

