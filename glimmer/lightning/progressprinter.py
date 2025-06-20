# Copyright (c) 2021 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).

import random
import time
import uuid
from collections import defaultdict
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from IPython.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from glimmer.lightning.utils import is_console


def fuse_samples(samples: np.ndarray):
    """
    Fuse samples with the same timestamp by using the latest entry.

    Args:
        samples (np.ndarray): A NumPy array with shape (n_samples, n_columns).

    Returns:
        np.ndarray: A NumPy array with shape (n_unique_samples, n_columns).
    """
    fused_samples_dict = {row[0]: row for row in samples}
    fused_samples = np.array(list(fused_samples_dict.values()))
    return fused_samples


def format_time(t):
    """
    Format `t` (in seconds) to (h):mm:ss.

    Args:
        t: Time in seconds.

    Returns:
        str: Formatted time string in (h):mm:ss format.
    """
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}"


def improvement_styler(df, metric="loss"):
    """
    Style DataFrame to highlight improvements in the specified metric.

    Args:
        df: The DataFrame to style.
        metric (str): The metric column to use for highlighting improvements.

    Returns:
        DataFrame: A styled DataFrame with colors indicating improvements.
    """
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
        use_console: bool | Literal["auto"] = "auto",
        python_logger=None,
        silent: bool = False,
    ):
        """
        Initialize the ProgressPrinter callback.

        Args:
            highlight_improvements (bool): Whether to highlight improvements in metrics.
            improvement_metric (str): The metric to use for highlighting improvements.
            use_console (bool | Literal["auto"]): Whether to use console output instead of Jupyter.
                - True: Always use console output
                - False: Always use Jupyter output
                - "auto": Automatically detect the environment
            python_logger: Optional Python logger to use for output.
            silent (bool): Whether to suppress all output.
        """
        self.highlight_improvements = highlight_improvements
        self.improvement_metric = improvement_metric
        self.use_console = use_console
        self.python_logger = python_logger
        self.metrics = []
        self.best_epoch = {"loss": np.inf, "val/loss": np.inf, "epoch": -1}
        self.last_time = 0
        self.table_display = None
        self.table_id = "progressprinter-" + str(uuid.uuid4())
        self.silent = silent

        self.epochs = []
        self.epochs_times = []
        self.train_metrics = []
        self.validation_metrics = []
        self.extra_metrics = []
        self.has_been_trained = False

    def on_train_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self.last_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        now = time.time()
        elapsed_time = now - self.last_time

        current_train_metrics = {}
        current_validation_metrics = {}
        current_extra_metrics = {}

        for name, value in trainer.callback_metrics.items():
            float_value = float(value)
            if "loss" in name and "val/" not in name:
                current_train_metrics[name] = [trainer.global_step, float_value]
            elif "loss" in name and "val/" in name:
                current_validation_metrics[name] = [trainer.global_step, float_value]
            elif "loss" not in name:
                current_extra_metrics[name] = [trainer.global_step, float_value]

        self.train_metrics.append(current_train_metrics)
        self.validation_metrics.append(current_validation_metrics)
        self.extra_metrics.append(current_extra_metrics)

        self.epochs.append([trainer.global_step, trainer.current_epoch])
        self.epochs_times.append([trainer.global_step, elapsed_time])

        # self.collect_metrics(trainer)
        if not self.silent:
            self.print(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.has_been_trained = True

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # NOTE: on_train_epoch_end is the default print caller
        # however, in case of incomplete epochs, this is also needs to be called
        # since we call validation explicitly after training
        # GIST: this prints a second time after an incomplete epoch, but only then
        # and we need to collect the validation metrics actively here
        if not trainer.training and self.has_been_trained and not self.silent:
            current_validation_metrics = {}
            current_extra_metrics = {}

            for name, value in trainer.callback_metrics.items():
                float_value = float(value)
                if "loss" in name and "val/" not in name:
                    pass
                elif "loss" in name and "val/" in name:
                    current_validation_metrics[name] = [
                        trainer.global_step,
                        float_value,
                    ]
                elif "loss" not in name:
                    current_extra_metrics[name] = [trainer.global_step, float_value]

            self.validation_metrics.append(current_validation_metrics)
            self.extra_metrics.append(current_extra_metrics)
            self.print(trainer)

    def _print_jupyter(self, trainer) -> None:
        metrics = self.static_print(verbose=False)

        if not self.table_display:
            self.table_display = display(metrics, display_id=self.table_id)
        else:
            self.table_display.update(metrics)


    def _print_console(self, trainer) -> None:
        # Simple progress print without clearing screen
        train_metrics, val_metrics, extra_metrics = self.get_logged_metrics()

        # Get latest metrics for current epoch
        latest_step = trainer.global_step
        latest_epoch = trainer.current_epoch

        # Build progress line
        progress_parts = [f"Epoch {latest_epoch:2d}", f"Step {latest_step:4d}"]

        # Add loss metrics
        for name, values in train_metrics.items():
            if "loss" in name and len(values) > 0:
                latest_value = values[values[:, 0] == latest_step]
                if len(latest_value) > 0:
                    short_name = name.replace("box_l1_loss", "L1").replace("box_giou_loss", "GIoU").replace("class_focal_loss", "Focal")
                    progress_parts.append(f"{short_name}: {latest_value[-1, 1]:.4f}")

        # Add validation loss if available
        for name, values in val_metrics.items():
            if "loss" in name and len(values) > 0:
                latest_value = values[values[:, 0] == latest_step]
                if len(latest_value) > 0:
                    short_name = name.replace("val/box_l1_loss", "val_L1").replace("val/box_giou_loss", "val_GIoU").replace("val/class_focal_loss", "val_Focal").replace("val/loss", "val_loss")
                    progress_parts.append(f"{short_name}: {latest_value[-1, 1]:.4f}")

        # Print as a single line with carriage return to overwrite
        progress_line = " | ".join(progress_parts)
        print(f"\r🚀 {progress_line}                    ", end="", flush=True)


    def print(self, trainer) -> None:
        """
        Print training progress based on the use_console setting.
        
        Args:
            trainer: The PyTorch Lightning trainer instance.
        """
        should_use_console = self._should_use_console()
        if not should_use_console:
            self._print_jupyter(trainer)
        else:
            self._print_console(trainer)

    def _should_use_console(self) -> bool:
        """
        Determine whether to use console output based on the use_console setting.

        Returns:
            bool: True if console output should be used, False for Jupyter output.
        """
        if self.use_console == "auto":
            # Auto-detect environment using utility function
            return is_console()
        else:
            # Explicit boolean value
            return bool(self.use_console)

    def static_print(self, verbose: bool = True) -> pd.DataFrame:
        def metrics_to_dataframe(metrics: dict):
            metrics_df = pd.DataFrame()
            for metric_name, data in metrics.items():
                temp_df = pd.DataFrame(data, columns=["global_step", metric_name])
                if metrics_df.empty:
                    metrics_df = temp_df
                else:
                    metrics_df = pd.merge(
                        metrics_df, temp_df, on="global_step", how="outer"
                    )
            return metrics_df

        train_metrics, validation_metrics, extra_metrics = self.get_logged_metrics()
        train_metrics = metrics_to_dataframe(train_metrics)
        validation_metrics = metrics_to_dataframe(validation_metrics)
        extra_metrics = metrics_to_dataframe(extra_metrics)
        metrics = train_metrics
        if len(validation_metrics) > 0:
            metrics = pd.merge(
                metrics, validation_metrics, on="global_step", how="outer"
            )
        if len(extra_metrics) > 0:
            metrics = pd.merge(metrics, extra_metrics, on="global_step", how="outer")
        metrics = metrics.sort_values(by="global_step")
        metrics = metrics.set_index("epoch")
        metrics = metrics.convert_dtypes()
        metrics["time"] = metrics["time"].apply(format_time)

        # https://stackoverflow.com/questions/49239476/hide-a-pandas-column-while-using-style-apply
        if self.highlight_improvements:
            if self.improvement_metric in metrics.columns:
                partial_styler = partial(
                    improvement_styler, metric=self.improvement_metric
                )
                metrics = metrics.style.apply(partial_styler, axis=None)

        if verbose:
            display(metrics, display_id=43 + random.randint(0, int(1e6)))

        return metrics

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
                name: fuse_samples(np.array(values))
                for name, values in metrics_values.items()
            }
            return metrics

        train_metrics = _fuse_metrics(self.train_metrics)
        validation_metrics = _fuse_metrics(self.validation_metrics)
        # NOTE: epochs and times are added to extra metrics for column order
        # since extra metrics will come after train and validation metrics
        extra_metrics = {
            "epoch": np.array(self.epochs),
            "time": np.array(self.epochs_times),
            **_fuse_metrics(self.extra_metrics),
        }

        return train_metrics, validation_metrics, extra_metrics
