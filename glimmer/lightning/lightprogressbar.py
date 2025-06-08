# Copyright (c) 2021 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import sys
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.trainer.states import TrainerFn


class LightProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        has_main_bar = self.trainer.state.fn != TrainerFn.VALIDATING
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
        self.val_progress_bar = None
        super().on_train_start(*_)

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if self._val_progress_bar is not None:
            self.val_progress_bar.reset()
            self.val_progress_bar.initial = 0
        super().on_train_epoch_start(trainer, *_)

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.sanity_checking and self._val_progress_bar is None:
            self.val_progress_bar = self.init_validation_tqdm()

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.state.fn != TrainerFn.FITTING:
            self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
        if (
            self._train_progress_bar is not None
            and trainer.state.fn == TrainerFn.FITTING
        ):
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
