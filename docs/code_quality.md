
# ✅ Code Quality & Static Analysis

This section documents the use of linting, formatting, static analysis, and security tools in the **AI MLOps** project. All tools are centrally configured via `pyproject.toml` (when supported) to maintain consistency.

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

---

### 🔸 `isort` — Import Sorter
- **Purpose:** Automatically sort imports using a `black`-compatible profile.
- **Run:**
  ```bash
  isort .              # Format
  isort --check-only . # Check only
  ```
- **Config:** `[tool.isort]` in `pyproject.toml`, with `profile = "black"`.

---

### 🔸 `flake8` — Code Style & Static Errors
- **Purpose:** Static code analysis to detect style errors or potential bugs.
- **Run:**
  ```bash
  flake8 .
  ```
- **Config:** `[tool.flake8]` in `pyproject.toml` or a `.flake8` file.

---

## 🔍 Type Checking

### 🔸 `mypy` — Static Typing Checker
- **Purpose:** Verifies type annotations using type hints (`Optional`, `str`, etc.).
- **Run:**
  ```bash
  mypy .
  ```
- **Minimal config recommendation (mypy.ini or pyproject.toml):**
  ```toml
  [tool.mypy]
  ignore_missing_imports = true
  disallow_untyped_defs = true
  ```
- **Tip:** For untyped libraries, install stubs:
  ```bash
  pip install types-PyYAML types-requests ...
  ```

---

## 🔐 Security

### 🔸 `bandit` — Security Scanner
- **Purpose:** Scans for known vulnerabilities in Python scripts (e.g., use of `eval`, `assert`, hardcoded secrets).
- **Run:**
  ```bash
  bandit -c pyproject.toml -r .
  ```
- **Config file:** `.bandit` or `[tool.bandit]` in `pyproject.toml` (not officially supported in Bandit 1.x):
  ```ini
  [bandit]
  exclude_dirs = venv,.venv,__pycache__,mlruns
  ```

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

| Tool     | Uses `pyproject.toml`? | Required for CI | Auto-fix Available |
|----------|------------------------|------------------|---------------------|
| black    | ✅                    | ✅               | ✅                  |
| isort    | ✅                    | ✅               | ✅                  |
| flake8   | ✅                    | ✅               | ❌                  |
| mypy     | ✅                    | ✅               | ❌                  |
| bandit   | ✅                    | ✅               | ❌                  |
| pytest   | ✅                    | ✅               | n/a                 |

---

**Maintainer:** Erick Jara — CTO & AI/Data Engineer  
📧 [erick.jara@hotmail.it](mailto:erick.jara@hotmail.it)