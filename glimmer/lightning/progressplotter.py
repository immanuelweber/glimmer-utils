# Copyright (c) 2021 - 2022 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from .lightning_derived import get_lrs, get_scheduler_names

# for multiple y axis see
# https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales


class ProgressPlotter(Callback):
    def __init__(
        self,
        highlight_best: bool = True,
        show_extra_losses: bool = True,
        show_steps: bool = True,
        show_lr: bool = True,
        silent: bool = False,
    ):
        self.highlight_best = highlight_best
        self.best_of = "val"  # not implemented
        self.show_extra_losses = show_extra_losses
        self.metrics = []
        self.train_loss = []
        self.val_loss = []
        self.extra_metrics = defaultdict(list)
        self.extra_style = "--"
        self.steps = []
        self.did = None
        self.show_lr = show_lr
        self.lrs = defaultdict(list)
        self.lr_color = plt.cm.viridis(0.5)
        self.show_steps = show_steps
        self.silent = silent

    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        self.scheduler_names = get_scheduler_names(trainer.lr_schedulers)
        self.steps_per_epoch = trainer.num_training_batches

    def on_train_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.train_loss.append(float(trainer.callback_metrics["loss"]))
        lrs = get_lrs(trainer.lr_schedulers, self.scheduler_names, "step")
        for k, v in lrs.items():
            self.lrs[k].append(v)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        self.collect_metrics(trainer)
        if not self.silent:
            self.update_plot(
                trainer, self.highlight_best, self.show_lr, self.show_steps
            )

    def collect_metrics(self, trainer):
        val_loss = None
        raw_metrics = trainer.logged_metrics.copy()
        ignored_metrics = ["loss", "epoch"]
        for m in ignored_metrics:
            if m in raw_metrics:
                raw_metrics.pop(m)
        if "val_loss" in raw_metrics:
            val_loss = float(raw_metrics.pop("val_loss"))
        elif "val_loss_epoch" in raw_metrics:
            val_loss = float(raw_metrics.pop("val_loss_epoch"))
        if "val_loss_step" in raw_metrics:
            raw_metrics.pop("val_loss_step")
        if val_loss is not None:
            self.val_loss.append(val_loss)
        self.steps.append(trainer.global_step)
        for key, value in raw_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            self.extra_metrics[key].append(value)

    def update_plot(self, trainer, highlight_best, show_lr, show_steps):
        fig, ax = plt.subplots()
        plt.close(fig)
        if trainer.max_steps:
            max_steps_from_epochs = trainer.max_epochs * trainer.num_training_batches
            max_steps = min(trainer.max_steps, max_steps_from_epochs)
        else:
            max_steps = trainer.max_epochs * trainer.num_training_batches
        self.static_plot(ax, show_lr, highlight_best, show_steps, max_steps=max_steps)

        if self.did:
            self.did.update(fig)
        else:
            rand_id = random.randint(0, 1e6)
            self.did = display(fig, display_id=23 + rand_id)

    def static_plot(
        self,
        ax=None,
        show_lr=True,
        highlight_best=False,
        show_steps=True,
        max_steps=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        max_steps = max_steps if max_steps else len(self.train_loss)
        step_ax = ax.twiny()
        step_ax.set_xlabel("step")
        step_ax.set_xlim(0, max_steps)
        if not show_steps:
            step_ax.set_xticks([])
            step_ax.set_xlabel("")

        step_ax.plot(self.train_loss, label="loss")
        ax.set_xlabel("epoch")
        ax.set_xlim([0, max_steps / self.steps_per_epoch])

        if self.val_loss:
            ph = step_ax.plot(self.steps, self.val_loss, label="val_loss")
            if highlight_best:
                best_epoch = np.argmin(self.val_loss)
                best_step = (best_epoch + 1) * self.steps_per_epoch
                best_loss = self.val_loss[best_epoch]
                step_ax.plot(best_step, best_loss, "o", c=ph[0].get_color())
        lines, labels = step_ax.get_legend_handles_labels()

        if len(self.extra_metrics) and self.show_extra_losses:
            extra_ax = step_ax.twinx()
            extra_ax.set_ylabel("extra metrics")
            for key in sorted(self.extra_metrics.keys()):
                extra_ax.plot(
                    self.steps, self.extra_metrics[key], self.extra_style, label=key
                )
            extra_lines, extra_labels = extra_ax.get_legend_handles_labels()
            lines += extra_lines
            labels += extra_labels
        if show_lr and len(self.lrs):
            lr_ax = step_ax.twinx()
            lr_ax.set_ylabel("lr")
            lr_ax.spines["right"].set_position(("outward", 60))
            for key, lrs in self.lrs.items():
                lr_ax.plot(lrs, c=self.lr_color, label=key)
            lr_ax.yaxis.label.set_color(self.lr_color)

        step_ax.legend(lines, labels, loc=0)
