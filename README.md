# Glimmer Utils

[![PyPI version](https://img.shields.io/pypi/v/glimmer.svg)](https://pypi.org/project/glimmer/) [![Python versions](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://pypi.org/project/glimmer/) [![License](https://img.shields.io/github/license/immanuelweber/glimmer-utils.svg)](LICENSE)

Utilities for working with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/).

## Features

- Data collators for padding and stacking tensors
- Lightning callbacks for progress bars, printing and plotting
- A patched `LightningDataModule` for quick dataset usage

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/immanuelweber/glimmer-utils.git
cd glimmer-utils
pip install .
```

For development use the editable flag:

```bash
pip install -e .
```

The `pyproject.toml` in this repository specifies `setuptools` and `wheel`
as build dependencies so that `pip install` works without additional setup.



## Requirements

Glimmer Utils requires Python 3.8 or later and a small set of third-party packages:

- numpy
- pandas
- matplotlib
- ipython
- torch
- pytorch-lightning

You can install them all with:

```bash
pip install -r requirements.txt
```

## Testing

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

To verify the package compiles, run:

```bash
python -m compileall -q glimmer
```
