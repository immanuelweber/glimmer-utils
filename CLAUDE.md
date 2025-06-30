# my rules

## general

- keep it short and simple (KISS)
- don't develop features that are not needed
- no overengineering

## setup

- initialize a git repository with `main` as the default branch
- add a `.gitignore` (use [https://www.toptal.com/developers/gitignore](https://www.toptal.com/developers/gitignore))
- create a `pyproject.toml` as the single source of truth for tooling config
- optionally use `.pre-commit-config.yaml` for local hooks

## python

- use Python >= 3.10 (type hints, match-case, structural pattern matching, etc.)
- use f-strings, type hints, and `match-case` where appropriate
- use `import` statements at the top of the file
- prefer `from ... import ...` over `import ...` for granularity
- no relative imports (only absolute)
- docstrings must follow Google style with type hints

## formatting & linting (via `pyproject.toml`)

- use [ruff](https://docs.astral.sh/ruff/) as the unified tool for:
  - formatting (replaces `black`)
  - import sorting (replaces `isort`)
  - linting (replaces `flake8`)
  - type checking (partial mypy replacement)
- configure ruff in `pyproject.toml`
- run `ruff check . --fix` regularly
- if needed, use `mypy` for full static type checking
- enforce checks via `pre-commit` if possible

## testing

- use `pytest` with test discovery via `tests/`
- keep tests fast, focused, and minimal
- run tests locally before committing, if available and appropriate