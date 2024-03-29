# Copyright (c) 2021 - 2023 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import random
import time
import uuid
from functools import partial

import numpy as np
import pandas as pd
from IPython.display import display
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}"


def improvement_styler(df, metric="loss"):
    # https://stackoverflow.com/questions/50220200/conditional-styling-in-pandas-using-other-columns
    worse_color = "color: red"
    better_color = "color: black"
    mask = df[metric].diff() > 0

    # DataFrame with same index and columns names as original filled empty strings
    styled_df = pd.DataFrame(better_color, index=df.index, columns=df.columns)
    styled_df.loc[mask] = worse_color

    min_loss = df[metric].min()
    mask = df[metric] == min_loss
    styled_df.loc[mask] = "font-weight: bold"
    return styled_df


class ProgressPrinter(Callback):
    def __init__(
        self,
        highlight_improvements: bool = True,
        improvement_metric: str = "loss",
        console: bool = False,
        python_logger=None,
        silent: bool = False,
    ):
        self.highlight_improvements = highlight_improvements
        self.improvement_metric = improvement_metric
        self.console = console
        self.python_logger = python_logger
        self.metrics = []
        self.best_epoch = {"loss": np.inf, "val/loss": np.inf, "epoch": -1}
        self.last_time = 0
        self.table_display = None
        self.table_id = "progressprinter-" + str(uuid.uuid4())
        self.silent = silent

    def on_train_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self.last_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        self.collect_metrics(trainer)
        if not self.silent:
            self.print(trainer)

    def collect_metrics(self, trainer):
        metrics = {
            "epoch": trainer.current_epoch,
            # TODO: no mean loss available for in logged metrics, better way?!
            "loss": float(trainer.callback_metrics["loss"]),
        }

        raw_metrics = trainer.logged_metrics.copy()
        ignored_metrics = ["loss", "epoch"]
        for m in ignored_metrics:
            if m in raw_metrics:
                raw_metrics.pop(m)
        if "val/loss" in raw_metrics:
            metrics["val/loss"] = float(raw_metrics.pop("val/loss"))
        elif "val/loss_epoch" in raw_metrics:
            metrics["val/loss"] = float(raw_metrics.pop("val/loss_epoch"))
        if "val/loss_step" in raw_metrics:
            raw_metrics.pop("val/loss_step")
        for key, value in raw_metrics.items():
            if key.endswith("_epoch"):
                metrics[key[6:]] = float(value)
            elif not key.endswith("_step"):
                metrics[key] = float(value)

        if "val/loss" in metrics:
            if metrics["val/loss"] < self.best_epoch["val/loss"]:
                self.best_epoch = metrics
        else:
            if metrics["loss"] < self.best_epoch["loss"]:
                self.best_epoch = metrics

        now = time.time()
        elapsed_time = now - self.last_time
        metrics["time"] = format_time(elapsed_time)
        self.metrics.append(metrics)

    def __print_jupyter(self, trainer) -> None:
        metrics_df = pd.DataFrame.from_records(self.metrics)
        # https://stackoverflow.com/questions/49239476/hide-a-pandas-column-while-using-style-apply
        if self.highlight_improvements:
            partial_styler = partial(improvement_styler, metric=self.improvement_metric)
            metrics_df = metrics_df.style.apply(partial_styler, axis=None)
            metrics_df = metrics_df.hide(axis="index")

        if not self.table_display:
            self.table_display = display(metrics_df, display_id=self.table_id)
        else:
            self.table_display.update(metrics_df)

    def __print_console(self, trainer) -> None:
        metrics_df = pd.DataFrame.from_records(self.metrics)
        last_row = metrics_df.iloc[-1]
        metrics = {index: last_row[index] for index in last_row.index}

        def __format(val):
            return f"{val:.4f}" if isinstance(val, float) else val

        metrics = {key: __format(val) for key, val in metrics.items()}
        metrics = ", ".join(
            [f"{key}: {val}" for key, val in metrics.items() if key != "epoch"]
        )
        pad = len(str(trainer.max_epochs))
        if self.python_logger:
            self.python_logger.info(
                f"{last_row.name:>{pad}}/{trainer.max_epochs}: {metrics}"
            )
        else:
            print(f"{last_row.name:>{pad}}/{trainer.max_epochs}: {metrics}")

    def print(self, trainer) -> None:
        if not self.console:
            self.__print_jupyter(trainer)
        else:
            self.__print_console(trainer)

    def static_print(self, verbose: bool = True) -> pd.DataFrame:
        metrics_df = pd.DataFrame.from_records(
            [
                self.best_epoch,
                self.metrics[-1] if len(self.metrics) else self.best_epoch,
            ]
        )
        metrics_df.index = ["best", "last"]
        if verbose:
            display(metrics_df, display_id=43 + random.randint(0, 1e6))
        return metrics_df
