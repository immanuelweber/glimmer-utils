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

The `pyproject.toml` specifies all build requirements and uses
`setuptools_scm` to derive the version from git tags. The repository is
currently tagged at **v0.3.0**.



## Requirements

Glimmer Utils requires Python 3.10 or later. The runtime dependencies are:

- numpy
- pandas
- matplotlib
- ipython
- torch
- pytorch-lightning

These packages are listed in `pyproject.toml` and will be installed
automatically when you install the project (see the Installation section
above).

## Development

Install additional tools for linting and testing:

```bash
pip install -e '.[dev,test]'
```

## Testing

Run the unit tests with [pytest](https://docs.pytest.org/):

```bash
pytest -v
```

To verify the package compiles, run:

```bash
python -m compileall -q glimmer
```
