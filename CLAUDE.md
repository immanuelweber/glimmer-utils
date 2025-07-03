# My Rules

## General

- Keep it short and simple (KISS)
- Don’t develop features that are not needed
- No overengineering

## Prototyping

- When building demos, snippets, or prototypes:
  - Use the simplest approach that works
  - Focus on getting the job done quickly
  - Create a minimal working example that is easy to understand and complete
  - When in doubt, prefer step-by-step code before optimizing or refactoring

## Setup

- Initialize a Git repository with `main` as the default branch
- Add a `.gitignore` (e.g. via [toptal/gitignore generator](https://www.toptal.com/developers/gitignore))
- Create a `pyproject.toml` as the single source of truth for tooling config
- Always work inside a `conda` or `mamba` environment
  - Do not auto-create one
  - If not already in one, prompt the user to activate or select an environment
- Optionally use `.pre-commit-config.yaml` for local hooks
- Install hooks with `pre-commit install` (optional but recommended)
- Manage dependencies via `pyproject.toml` and optionally `requirements.txt` (e.g. via `pip-compile` or `uv pip compile`)
- Optionally use a `Makefile` or `justfile` for repeatable tasks

## Python

- Use Python ≥ 3.10 (type hints, match-case, structural pattern matching, etc.)
- Use f-strings, type hints, and `match-case` where appropriate
- Place all `import` statements at the top of the file
- Prefer `from ... import ...` over `import ...` for granularity
- No relative imports (only absolute)
- Use docstrings in Google style with type hints
- Use the `logging` module instead of `print()` for anything beyond quick debugging

## Formatting & Linting (via `pyproject.toml`)

- Use [ruff](https://docs.astral.sh/ruff/) as the unified tool for:
  - Formatting (replaces `black`)
  - Import sorting (replaces `isort`)
  - Linting (replaces `flake8`)
  - Type checking (partial `mypy` replacement)
- Configure `ruff` in `pyproject.toml`
- Run `ruff check . --fix` regularly
- If needed, use `mypy` for full static type checking
- Enforce checks via `pre-commit` if possible

## Testing

- Use `pytest` with test discovery via `tests/` (prefix test files with `test_`)
- Keep tests fast, focused, and minimal
- Run tests locally before committing, if available and appropriate
