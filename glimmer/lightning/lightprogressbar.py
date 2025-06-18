# Copyright (c) 2021 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import os
import sys
from typing import Any, Literal

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.trainer.states import TrainerFn


class LightProgressBar(TQDMProgressBar):
    """A custom Lightning progress bar with enhanced control over display behavior.

    This progress bar extends the standard TQDM progress bar with additional
    logic to automatically disable progress bars in non-interactive environments
    like CI/CD pipelines, Azure ML, SLURM clusters, and Kubernetes.
    """

    def __init__(self, progress_mode: bool | Literal["auto"] = "auto", **kwargs: Any) -> None:
        """Initialize the LightProgressBar.

        Args:
            progress_mode: Controls progress bar display behavior.
                - True: Force enable progress bar regardless of environment
                - False: Force disable progress bar
                - "auto": Automatically enable only in interactive terminals
            **kwargs: Additional keyword arguments passed to parent TQDMProgressBar.
        """
        super().__init__(**kwargs)
        self._progress_mode: bool | Literal["auto"] = progress_mode
        self._disable_bar: bool | None = None  # will be determined once

    def init_validation_tqdm(self) -> Tqdm:
        """Initialize and return a TQDM progress bar for validation.

        Creates a validation progress bar with appropriate positioning and
        formatting based on the current trainer state and progress bar settings.

        Returns:
            Tqdm: Configured TQDM progress bar instance for validation tracking.
        """
        has_main_bar: bool = self.trainer.state.fn != TrainerFn.VALIDATING
        disable_bar: bool = self.disable_progress_bar()

        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=disable_bar,
            leave=True,
            dynamic_ncols=not disable_bar,
            file=sys.stdout,
            bar_format=None if disable_bar else self.BAR_FORMAT,
        )

    def disable_progress_bar(self) -> bool:
        """Determine whether the progress bar should be disabled.

        Uses cached result if available, otherwise evaluates based on progress_mode
        and environment variables. Progress bars are automatically disabled in
        non-interactive environments like CI/CD, Azure ML, SLURM, and Kubernetes.

        Returns:
            bool: True if progress bar should be disabled, False otherwise.
        """
        if self._disable_bar is not None:
            return self._disable_bar

        if isinstance(self._progress_mode, bool):
            self._disable_bar = not self._progress_mode
        else:  # "auto"
            env = os.environ
            self._disable_bar = (
                env.get("DISABLE_PROGRESS", "0") == "1" or
                not sys.stdout.isatty() or
                env.get("CI") == "true" or
                "AZUREML_RUN_TOKEN" in env or
                "SLURM_JOB_ID" in env or
                "KUBERNETES_SERVICE_HOST" in env
            )

        return self._disable_bar

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
            self._val_progress_bar.reset()
            self._val_progress_bar.initial = 0
        super().on_train_epoch_start(trainer, *_)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when validation starts.

        Initializes the validation progress bar if not in sanity checking mode
        and no validation progress bar exists yet.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module being validated.
        """
        if not trainer.sanity_checking and self._val_progress_bar is None:
            self.val_progress_bar = self.init_validation_tqdm()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when validation ends.

        Closes the validation progress bar if not in fitting mode, resets dataloader
        tracking, and updates training progress bar metrics if in fitting mode.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module that was validated.
        """
        if trainer.state.fn != TrainerFn.FITTING:
            self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
        if (
            self._train_progress_bar is not None
            and trainer.state.fn == TrainerFn.FITTING
        ):
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
