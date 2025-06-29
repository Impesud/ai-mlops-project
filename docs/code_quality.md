# ✅ Code Quality & Static Analysis

This section documents the use of linting, formatting, static analysis, and security tools in the **AI MLOps** project. All tools are centrally configured via `pyproject.toml` and integrated with `pre-commit` and GitHub Actions CI/CD pipelines to enforce code quality automatically before every commit and during the build process.

---

## 🧹 Linting & Formatting Tools

### 🔸 `black` — Code Formatter

- **Purpose:** Automatically format Python code according to PEP8 standards.
- **Run:**
  ```bash
  black .           # Format
  black --check .   # Check only (no changes)
  ```
- **Config:** Managed via `[tool.black]` in `pyproject.toml`.
- **Integrated:** ✅ `pre-commit`, ✅ CI/CD

---

### 🔸 `isort` — Import Sorter

- **Purpose:** Automatically sort imports using a `black`-compatible profile.
- **Run:**
  ```bash
  isort .              # Format
  isort --check-only . # Check only
  ```
- **Config:** `[tool.isort]` in `pyproject.toml`, with `profile = "black"`.
- **Integrated:** ✅ `pre-commit`, ✅ CI/CD

---

### 🔸 `flake8` — Code Style & Static Errors

- **Purpose:** Static code analysis to detect style errors or potential bugs.
- **Run:**
  ```bash
  flake8 .
  ```
- **Config:** `[tool.flake8]` in `pyproject.toml` or a `.flake8` file.
- **Integrated:** ✅ `pre-commit`, ✅ CI/CD

---

## 🔍 Type Checking

### 🔸 `mypy` — Static Typing Checker

- **Purpose:** Verifies type annotations using type hints (`Optional`, `str`, etc.).
- **Run:**
  ```bash
  mypy .
  ```
- **Config:** Recommended via `pyproject.toml`:
  ```toml
  [tool.mypy]
  python_version = "3.11"
  ignore_missing_imports = true
  exclude = 'venv|\.venv|mlruns|\.mypy_cache'
  disallow_untyped_defs = true
  check_untyped_defs = true
  strict_optional = true
  warn_unused_ignores = true
  warn_return_any = true
  warn_unused_configs = true
  ```
- **Tip:** For untyped libraries, install stubs:
  ```bash
  pip install types-PyYAML types-requests ...
  ```
- **Integrated:** ✅ `pre-commit`, ✅ CI/CD

---

## 🔐 Security

### 🔸 `bandit` — Security Scanner

- **Purpose:** Scans for known vulnerabilities in Python scripts (e.g., use of `eval`, `assert`, hardcoded secrets).
- **Run:**
  ```bash
  bandit -r . --severity-level medium
  ```
- **Config file:** `.bandit` or run manually with excludes:
  ```bash
  bandit -r . -x venv,.venv,__pycache__,mlruns
  ```
- **Integrated:** ✅ `pre-commit`, ✅ CI/CD

---

## 🧪 Test Framework

### 🔸 `pytest` — Test Runner

- **Purpose:** Executes automated tests using fixture-driven structure.
- **Run:**
  ```bash
  TEST_ENV=dev pytest tests/
  TEST_ENV=prod pytest tests/
  ```
- **Recommended structure:**
  ```
  tests/
  ├── test_ingest.py
  ├── test_process.py
  └── test_train.py
  ```
- **Config:** Auto-detected if `pytest.ini` or `pyproject.toml` is present.
- **Run:**
  ```bash
  pytest .
  ```
- **Integrated:** ✅ CI/CD

---

## ⚙️ Pre-commit Integration

- **Tool:** [`pre-commit`](https://pre-commit.com)
- **Purpose:** Runs hooks like `black`, `flake8`, `mypy`, etc. before each commit to avoid pushing non-compliant code.
- **Setup:**
  ```bash
  pip install pre-commit
  pre-commit install            # Installs the Git hook
  pre-commit run --all-files    # Run on all files manually
  ```
- **Config:** `.pre-commit-config.yaml` in root directory
- **Typical hooks enabled:** `black`, `flake8`, `isort`, `mypy`, `bandit`

---

## 📦 Aggregation via `Makefile`

All tools can be invoked via:

```bash
make lint        # Runs black, isort, flake8 in check mode
make test-dev    # Runs pytest for dev environment
make test-prod   # Runs pytest for prod environment
```

---

## ✅ Recommendations

| Tool   | Uses `pyproject.toml`? | Required for CI | Pre-commit Hook | Auto-fix Available |
| ------ | ---------------------- | --------------- | --------------- | ------------------ |
| black  | ✅                      | ✅               | ✅               | ✅                  |
| isort  | ✅                      | ✅               | ✅               | ✅                  |
| flake8 | ✅                      | ✅               | ✅               | ❌                  |
| mypy   | ✅                      | ✅               | ✅               | ❌                  |
| bandit | ✅                      | ✅               | ✅               | ❌                  |
| pytest | ✅                      | ✅               | ❌               | n/a                |

---

**Maintainer:** Erick Jara — CTO & AI/Data Engineer\
📧 [erick.jara@hotmail.it](mailto\:erick.jara@hotmail.it)