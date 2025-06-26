# 🧠 Semantic Commit Message Guide

Semantic commit messages allow for automated versioning, changelog generation, and GitHub releases using tools like `semantic-release`.

## ✅ Why Use Semantic Commits?

- Enables automatic tagging (e.g. `v1.2.3`)
- Generates CHANGELOG.md automatically
- Triggers CI/CD workflows smartly
- Improves collaboration and traceability

---

## 🔑 Commit Message Structure

```
<type>[optional scope]: <short description>

[optional body]

[optional footer(s)]
```

---

## 🧩 Allowed Commit Types

| Type         | Purpose                                                                 |
|--------------|-------------------------------------------------------------------------|
| `feat`       | New feature                                                             |
| `fix`        | Bug fix                                                                 |
| `chore`      | Maintenance, build process, config or minor tasks (no user impact)     |
| `docs`       | Documentation only changes                                              |
| `refactor`   | Code changes that neither fix a bug nor add a feature                  |
| `test`       | Adding or improving tests                                               |
| `perf`       | Performance improvement                                                 |
| `style`      | Formatting, missing semicolons, indentation...                         |
| `ci`         | CI/CD configuration or scripts                                          |
| `revert`     | Revert to a previous commit                                             |
| `BREAKING CHANGE:` | Special footer to signal major version bump                     |

---

## 🧪 Examples

### New Feature
```
feat(api): add forecasting endpoint for batch predictions
```

### Bug Fix
```
fix(train): handle empty dataset edge case during model training
```

### Documentation
```
docs(readme): update setup instructions for MLflow
```

### Refactor
```
refactor(data): clean up processing script and modularize logic
```

### Breaking Change
```
feat(train): switch from sklearn to XGBoost training

BREAKING CHANGE: the training module now outputs XGBoost models instead of sklearn ones.
```

---

## 💡 Tips

- Keep commit messages short (max 72 chars)
- Use imperative tone: `add`, `fix`, `update`, not `added`, `fixed`
- Include body only when necessary (e.g., complex changes)
- Use `BREAKING CHANGE:` footer **only** when it affects public APIs or expected usage

---

## 🔄 Automatic Release Workflow

When merged into `main`, semantic-release will:

1. Determine the next version (based on commit types)
2. Create a GitHub tag (`vX.Y.Z`)
3. Update `CHANGELOG.md`
4. Create a GitHub Release

---

## 📦 Resources

- [Semantic Release Docs](https://semantic-release.gitbook.io)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

🧠 **Be consistent. Be semantic. Let automation work for you!**