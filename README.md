# Glimmer Utils

[![Python versions](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/immanuelweber/glimmer-utils/workflows/CI/badge.svg)](https://github.com/immanuelweber/glimmer-utils/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Utilities for working with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/).

## Features

- Data collators for padding and stacking tensors
- Lightning callbacks for progress bars, printing and plotting
- A patched `LightningDataModule` for quick dataset usage

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/immanuelweber/glimmer-utils.git
cd glimmer-utils
pip install .
```

For development and testing, install the optional extras in editable mode:

```bash
pip install -e '.[dev,test]'
```

The package uses `setuptools_scm` to automatically derive the version from git tags.

## Development

### Quick Setup with justfile

This project uses [just](https://github.com/casey/just) for task automation.

First, install just if you haven't already:

```bash
pip install just-bin
```

After cloning the repository, run:

```bash
just setup
```

This will install the package in editable mode with all dev dependencies and set up pre-commit hooks.

Alternatively, you can run the steps manually:

```bash
pip install -e '.[dev,test]'
pre-commit install
```

### Available justfile commands

Run `just --list` to see all available commands:

- `just setup` - Install package and pre-commit hooks
- `just install` - Install package in editable mode with dev dependencies
- `just install-hooks` - Install pre-commit hooks
- `just lint` - Run all pre-commit hooks on all files
- `just test` - Run pytest with verbose output
- `just test-cov` - Run tests with coverage report
- `just check` - Run all checks (lint + typecheck)
- `just ci` - Full CI workflow

## Testing

Run the unit tests:

```bash
just test
```

Or directly with [pytest](https://docs.pytest.org/):

```bash
pytest -v
```

For tests with coverage report:

```bash
just test-cov
```
