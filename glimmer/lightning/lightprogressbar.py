# Copyright (c) 2021 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import sys
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.trainer.states import TrainerFn


class LightProgressBar(TQDMProgressBar):
    """A custom Lightning progress bar with enhanced control over display behavior."""

    def init_validation_tqdm(self) -> Tqdm:
        """Initialize and return a TQDM progress bar for validation.

        Creates a validation progress bar with appropriate positioning and
        formatting based on the current trainer state and progress bar settings.

        Returns:
            Tqdm: Configured TQDM progress bar instance for validation tracking.
        """
        has_main_bar: bool = self.trainer.state.fn != TrainerFn.VALIDATING

        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def on_train_start(self, *_: Any) -> None:
        """Called when training starts.

        Resets the validation progress bar to None and calls the parent method.

        Args:
            *_: Variable arguments passed from Lightning (unused).
        """
        self.val_progress_bar = None
        super().on_train_start(*_)

    def on_train_epoch_start(self, trainer: pl.Trainer, *_: Any) -> None:
        """Called when a training epoch starts.

        Resets the validation progress bar if it exists and calls the parent method.

        Args:
            trainer: The Lightning trainer instance.
            *_: Additional variable arguments passed from Lightning (unused).
        """
        if self._val_progress_bar is not None:
            self.val_progress_bar.reset()  # type: ignore[attr-defined]
            self.val_progress_bar.initial = 0  # type: ignore[attr-defined]
        super().on_train_epoch_start(trainer, *_)

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when validation starts.

        Initializes the validation progress bar if not in sanity checking mode
        and no validation progress bar exists yet.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module being validated.
        """
        if not trainer.sanity_checking and self._val_progress_bar is None:
            self.val_progress_bar = self.init_validation_tqdm()

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when validation ends.

        Closes the validation progress bar if not in fitting mode, resets dataloader
        tracking, and updates training progress bar metrics if in fitting mode.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module that was validated.
        """
        if trainer.state.fn != TrainerFn.FITTING:
            self.val_progress_bar.close()  # type: ignore[attr-defined]
        self.reset_dataloader_idx_tracker()
        if (
            self._train_progress_bar is not None
            and trainer.state.fn == TrainerFn.FITTING
        ):
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, Any]:
        """Get metrics to display in the progress bar.

        Filters the metrics to show only the main loss value, removing individual
        loss components for a cleaner display.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module being trained.

        Returns:
            dict[str, Any]: Filtered metrics containing only the main loss.
        """
        # Get all metrics from parent implementation
        metrics = super().get_metrics(trainer, pl_module)

        # Filter to keep only the main "loss" metric
        filtered_metrics = {}
        for key, value in metrics.items():
            if key == "loss":
                filtered_metrics[key] = value

        return filtered_metrics
