# glimmer

[![CI](https://github.com/immanuelweber/glimmer-utils/workflows/CI/badge.svg)](https://github.com/immanuelweber/glimmer-utils/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Utilities for working with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/).

## Features

### Data
- **Collators** for padding and stacking tensors with variable-length inputs

### Lightning
- **PatchedDataModule** for quick dataset usage with PyTorch Lightning
- **LightProgressBar** — minimal progress bar callback
- **ProgressPlotter** — live training curve plotting
- **ProgressPrinter** — formatted metric printing
- **Utilities** — helper functions for Lightning modules

## Quick Start

```bash
git clone https://github.com/immanuelweber/glimmer-utils.git
cd glimmer-utils
uv sync
```

```python
from glimmer.data.collators import PadCollator
from glimmer.lightning.patcheddatamodule import PatchedDataModule
```

## Project Structure

```
glimmer/
    data/
        collators.py          # Padding and stacking collators
    lightning/
        lightning_derived.py  # Extended Lightning module
        lightprogressbar.py   # Minimal progress bar
        patcheddatamodule.py  # Patched LightningDataModule
        progressplotter.py    # Live training curves
        progressprinter.py    # Formatted metric output
        utils.py              # Lightning helpers
```

## Development

```bash
uv sync --group dev
pre-commit install
```

```bash
ruff check .            # Lint
ruff format .           # Format
uv run pytest           # Test
uv run mypy glimmer     # Type check
```

## License

MIT - see [LICENSE](LICENSE) for details.
