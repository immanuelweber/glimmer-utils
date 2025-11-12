# Justfile for glimmer-utils
# Install just: cargo install just OR brew install just
# Run: just <command>
# List all commands: just --list

# Default recipe (runs when you type 'just')
default:
    @just --list

# Install package in editable mode with dev dependencies
install:
    pip install -e '.[dev,test]'

# Install pre-commit hooks
install-hooks:
    pre-commit install

# Setup project (install package + hooks)
setup: install install-hooks

# Run all pre-commit hooks on all files
lint:
    pre-commit run --all-files

# Run ruff linter with auto-fix
ruff:
    ruff check . --fix

# Run ruff formatter
format:
    ruff format .

# Run mypy type checker
typecheck:
    mypy glimmer

# Run all checks (lint + typecheck)
check: lint typecheck

# Run pytest with verbose output
test:
    pytest -v

# Run tests with coverage report
test-cov:
    pytest -v --cov=glimmer --cov-report=html --cov-report=term

# Clean up build artifacts and cache files
clean:
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf .ruff_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Build package
build: clean
    python -m build

# Update pre-commit hooks to latest versions
update-hooks:
    pre-commit autoupdate

# Run a development shell with the package installed
shell:
    python -c "import glimmer; print(f'Glimmer {glimmer.__version__} loaded')"
    python

# Show current version
version:
    python -c "import glimmer; print(glimmer.__version__)"

# Full CI workflow (what runs in GitHub Actions)
ci: lint typecheck test
