# Copyright (c) 2021 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import uuid
from collections import defaultdict
from typing import Any

import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from glimmer.lightning.lightning_derived import get_lrs, get_scheduler_names

# for multiple y axis see
# https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales


def ld_to_dl(lst: list[dict]) -> dict:
    # list of dicts to dict of lists
    return {key: [dic[key] for dic in lst] for key in lst[0]}


class ProgressPlotter(Callback):
    def __init__(
        self,
        highlight_best: bool = True,
        show_sub_losses: bool = True,
        show_extra_metrics: bool = True,
        show_epochs: bool = True,
        show_lr: bool = True,
        silent: bool = False,
    ):
        self.highlight_best = highlight_best
        self.best_of = "val"  # not implemented
        self.show_sub_losses = show_sub_losses
        self.show_extra_metrics = show_extra_metrics
        self.extra_style = "--"
        self.plot_display = None
        self.plot_id = str(uuid.uuid4())
        self.show_lr = show_lr
        self.lrs = defaultdict(list)
        self.lr_color = "black"
        self.show_epochs = show_epochs
        self.silent = silent

        self.train_metrics = []
        self.validation_metrics = []
        self.extra_metrics = []
        self.has_been_trained = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.scheduler_names = get_scheduler_names(trainer.lr_scheduler_configs)
        self.num_training_batches = trainer.num_training_batches

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        current_train_metrics = {
            name: [trainer.global_step, float(value)]
            for name, value in trainer.callback_metrics.items()
            if "loss" in name and "val/" not in name
        }
        self.train_metrics.append(current_train_metrics)

        current_extra_metrics = {
            name: [trainer.global_step, float(value)]
            for name, value in trainer.callback_metrics.items()
            if "loss" not in name and "val/" not in name
        }
        self.extra_metrics.append(current_extra_metrics)

        lrs = get_lrs(trainer.lr_scheduler_configs, self.scheduler_names, "step")
        for name, value in lrs.items():
            self.lrs[name].append([trainer.global_step, float(value)])

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        if not self.silent:
            self.update_plot(
                trainer, self.highlight_best, self.show_lr, self.show_epochs
            )
        self.has_been_trained = True

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # NOTE: on_train_epoch_end would be better, since this is the real end of an epoch
        # however, on incomplete epochs this is also called, but no validation has been done
        # so we should work with on_validation_end which only gets called after validation
        # this prevents logging of data twice in case of incomplete epochs
        callback_metrics = trainer.callback_metrics
        current_validation_metrics = {
            name: [trainer.global_step, float(value)]
            for name, value in callback_metrics.items()
            if "val/" in name and "loss" in name
        }
        self.validation_metrics.append(current_validation_metrics)
        current_extra_metrics = {
            name: [trainer.global_step, float(value)]
            for name, value in trainer.callback_metrics.items()
            if "val/" in name and "loss" not in name
        }
        self.extra_metrics.append(current_extra_metrics)
        if not trainer.training and self.has_been_trained and not self.silent:
            self.update_plot(
                trainer, self.highlight_best, self.show_lr, self.show_epochs
            )

    def update_plot(self, trainer, highlight_best, show_lr, show_epochs):
        fig, ax = plt.subplots()
        plt.close(fig)
        if trainer.max_steps:
            max_steps_from_epochs = trainer.max_epochs * trainer.num_training_batches
            max_steps = min(trainer.max_steps, max_steps_from_epochs)
        else:
            max_steps = trainer.max_epochs * trainer.num_training_batches
        self.static_plot(ax, show_lr, highlight_best, show_epochs, max_steps=max_steps)

        if self.plot_display:
            self.plot_display.update(fig)
        else:
            self.plot_display = display(
                fig, display_id="progressplotter-" + self.plot_id
            )

    def static_plot(
        self,
        ax=None,
        show_lr=True,
        highlight_best=False,
        show_epochs=True,
        max_steps=None,
    ):
        if ax is None:
            _, ax = plt.subplots()

        train_metrics, validation_metrics, extra_metrics = self.get_logged_metrics()
        max_steps = max_steps if max_steps else len(train_metrics["loss"])
        for name, values in train_metrics.items():
            if "loss" not in name:
                continue
            ax.plot(values[:, 0], values[:, 1], label=name)
        # FIXME: better use a dict for that
        train_colors = [line.get_color() for line in ax.get_lines()]

        for (name, values), color in zip(validation_metrics.items(), train_colors):
            if "loss" not in name:
                continue
            ax.plot(
                values[:, 0],
                values[:, 1],
                label=name,
                color=color,
                linestyle="--",
                linewidth=1,
            )

        ax.set_xlabel("step")
        ax.set_xlim(0, max_steps)
        ax.legend()

        if highlight_best:
            # TODO: make this nicer
            if "val/loss" in validation_metrics:
                val_loss = validation_metrics["val/loss"]
                best_id = np.argmin(val_loss[:, 1])
                best_step = val_loss[best_id, 0]
                val_loss_color = "black"
                for line in ax.get_lines():
                    if line.get_label() == "val/loss":
                        val_loss_color = line.get_color()
                        break
                ax.plot(
                    best_step,
                    val_loss[best_id, 1],
                    "*",
                    color=val_loss_color,
                    markersize=20,
                )

        if show_lr and len(self.lrs):
            lr_ax = ax.twinx()
            lr_ax.set_ylabel("lr")
            # lr_ax.spines["right"].set_position(("outward", 10))
            for key, lrs in self.lrs.items():
                lrs = np.array(lrs)
                lr_ax.plot(
                    lrs[:, 0],
                    lrs[:, 1],
                    color=self.lr_color,
                    label=key,
                    linestyle="-",
                    linewidth=1,
                )
            lr_ax.yaxis.label.set_color(self.lr_color)

            # ensure that the lr_ax is behind the other axes
            ax.set_zorder(lr_ax.get_zorder() + 1)
            ax.patch.set_visible(False)

        # lines, labels = step_ax.get_legend_handles_labels()
        # if len(self.extra_metrics) and self.show_extra_metrics:
        #     extra_ax = step_ax.twinx()
        #     extra_ax.set_ylabel("extra metrics")
        #     for key in sorted(self.extra_metrics.keys()):
        #         extra_ax.plot(
        #             self.steps, self.extra_metrics[key], self.extra_style, label=key
        #         )
        #     extra_lines, extra_labels = extra_ax.get_legend_handles_labels()
        #     lines += extra_lines
        #     labels += extra_labels

        if show_epochs:
            epoch_ax = ax.twiny()
            epoch_ax.set_xlabel("epoch")
            epoch_ax.set_xlim([0, max_steps / self.num_training_batches])
            epoch_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    def get_logged_metrics(self):
        def _fuse_metrics(metrics: list[dict]):
            if len(metrics) == 0:
                return {}
            metrics_values = defaultdict(list)
            for submetrics in metrics:
                if len(submetrics) == 0:
                    continue
                for name, value in submetrics.items():
                    metrics_values[name].append(value)

            metrics = {
                name: np.array(values) for name, values in metrics_values.items()
            }
            return metrics

        train_metrics = _fuse_metrics(self.train_metrics)
        validation_metrics = _fuse_metrics(self.validation_metrics)
        extra_metrics = _fuse_metrics(self.extra_metrics)

        for key, lrs in self.lrs.items():
            extra_metrics[key] = np.array(lrs)

        return train_metrics, validation_metrics, extra_metrics
