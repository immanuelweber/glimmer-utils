# Copyright (c) 2018 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).

from typing import Any


def get_num_training_batches(trainer: Any) -> int:
    if (
        isinstance(trainer.limit_train_batches, int)
        or trainer.limit_train_batches == 0.0
    ):
        num_training_batches = min(
            trainer.num_training_batches, int(trainer.limit_train_batches)
        )
    elif trainer.num_training_batches != float("inf"):
        num_training_batches = int(
            trainer.num_training_batches * trainer.limit_train_batches
        )
    return num_training_batches


def get_max_epochs(trainer: Any) -> float:
    max_epochs = trainer.max_epochs
    if trainer.max_steps is not None:
        num_training_batches = get_num_training_batches(trainer)
        max_epochs_from_steps = trainer.max_steps / num_training_batches
        max_epochs = min(max_epochs, max_epochs_from_steps)
    return max_epochs


def is_console() -> bool:
    """
    Detect whether we're running in a console/terminal environment or Jupyter.

    Returns:
        bool: True if running in console/terminal, False if in Jupyter/IPython.
    """
    try:
        # Check if we're in a Jupyter environment
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        if ipython is not None and ipython.__class__.__name__ in [
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        ]:
            # We're in Jupyter/IPython
            return False
        else:
            # We're in a regular terminal
            return True
    except ImportError:
        # IPython not available, assume console
        return True
