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
        table_format: bool = True,
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
            table_format (bool): Whether to use table-style formatting for console output.
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
        self.table_format = table_format
        self.header_printed = False
        self.column_widths = {}
        self.last_printed_step = None
        self.last_printed_fractional_epoch = None

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

    def _calculate_fractional_epoch(self, trainer) -> float:
        """Calculate fractional epoch progress based on steps within current epoch."""
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step

        # Try to determine steps per epoch
        if hasattr(trainer, "num_training_batches") and trainer.num_training_batches:
            steps_per_epoch = trainer.num_training_batches
        elif (
            hasattr(trainer.datamodule, "train_dataloader")
            and trainer.datamodule.train_dataloader()
        ):
            steps_per_epoch = len(trainer.datamodule.train_dataloader())
        elif (
            hasattr(trainer, "estimated_stepping_batches")
            and trainer.estimated_stepping_batches
        ):
            # Estimate based on total steps and max epochs
            if trainer.max_epochs and trainer.max_epochs > 0:
                steps_per_epoch = (
                    trainer.estimated_stepping_batches / trainer.max_epochs
                )
            else:
                steps_per_epoch = None
        else:
            steps_per_epoch = None

        if steps_per_epoch and steps_per_epoch > 0:
            # Calculate steps completed in current epoch
            steps_in_current_epoch = global_step - (current_epoch * steps_per_epoch)
            fraction = steps_in_current_epoch / steps_per_epoch
            return current_epoch + fraction
        else:
            # Fallback to integer epochs if we can't calculate steps per epoch
            return float(current_epoch + 1)

    def _calculate_column_widths(self, trainer) -> dict[str, int]:
        """Calculate optimal column widths for table formatting."""
        train_metrics, val_metrics, extra_metrics = self.get_logged_metrics()

        # Get total epochs and steps for width calculation
        max_epochs = trainer.max_epochs if trainer.max_epochs else 999
        total_steps = (
            trainer.max_steps
            if trainer.max_steps != -1
            else trainer.estimated_stepping_batches
            if hasattr(trainer, "estimated_stepping_batches")
            else 99999
        )

        # Calculate width for epoch and step columns based on max possible values
        # Account for fractional epochs (e.g., "99.9/100")
        if str(max_epochs).isdigit():
            max_epoch_str = f"{max_epochs - 0.1:.1f}/{max_epochs}"  # "99.9/100"
        else:
            max_epoch_str = f"{max_epochs}/{max_epochs}"
        epoch_width = max(len("Epoch"), len(max_epoch_str))
        step_width = max(len("Step"), len(f"{total_steps}/{total_steps}"))

        widths = {
            "Epoch": epoch_width,
            "Step": step_width,
        }

        # Calculate widths for loss metrics
        for name, values in train_metrics.items():
            if "loss" in name and len(values) > 0:
                short_name = (
                    name.replace("box_l1_loss", "L1")
                    .replace("box_giou_loss", "GIoU")
                    .replace("class_focal_loss", "Focal")
                )
                # Width is max of header name and typical value format (e.g., "2.4199")
                widths[short_name] = max(len(short_name), 7)  # 7 chars for "2.4199 "

        # Calculate widths for validation metrics
        for name, values in val_metrics.items():
            if "loss" in name and len(values) > 0:
                short_name = (
                    name.replace("val/box_l1_loss", "val_L1")
                    .replace("val/box_giou_loss", "val_GIoU")
                    .replace("val/class_focal_loss", "val_Focal")
                    .replace("val/loss", "val_loss")
                )
                widths[short_name] = max(len(short_name), 7)  # 7 chars for "2.4199 "

        return widths

    def _print_table_header(self, column_widths: dict[str, int]) -> None:
        """Print the table header with proper column alignment."""
        header_parts = []
        for column_name, width in column_widths.items():
            header_parts.append(f"{column_name:>{width}}")

        header_line = "  ".join(header_parts)  # 2 spaces between columns
        print(f"🚀 {header_line}")

    def _print_table_row(self, trainer, column_widths: dict[str, int]) -> None:
        """Print a data row with proper column alignment."""
        train_metrics, val_metrics, extra_metrics = self.get_logged_metrics()

        # Get latest metrics for current epoch
        latest_step = trainer.global_step

        # Calculate fractional epoch first (needed for comparison)
        fractional_epoch = self._calculate_fractional_epoch(trainer)

        # Check if this is a validation update (same step and fractional epoch as last printed)
        is_validation_update = (
            self.last_printed_step is not None
            and self.last_printed_fractional_epoch is not None
            and latest_step == self.last_printed_step
            and abs(fractional_epoch - self.last_printed_fractional_epoch)
            < 0.01  # Small epsilon for floating point comparison
        )

        # Get total epochs and steps if available
        max_epochs = trainer.max_epochs if trainer.max_epochs else "?"
        total_steps = (
            trainer.max_steps
            if trainer.max_steps != -1
            else trainer.estimated_stepping_batches
            if hasattr(trainer, "estimated_stepping_batches")
            else "?"
        )

        # Fractional epoch already calculated above for comparison

        # Calculate padding based on total digits (account for decimal)
        if str(max_epochs).isdigit():
            # Padding should account for fractional display possibility
            epoch_padding = len(f"{max_epochs - 0.1:.1f}/{max_epochs}")
        else:
            epoch_padding = len(f"99.9/{max_epochs}")

        # Determine epoch display format based on whether it's fractional
        if abs(fractional_epoch - round(fractional_epoch)) < 0.01:
            # Complete epoch - display as integer
            epoch_display = f"{int(round(fractional_epoch))}/{max_epochs}"
        else:
            # Fractional epoch - display with decimal
            epoch_display = f"{fractional_epoch:.1f}/{max_epochs}"

        step_padding = len(str(total_steps)) if str(total_steps).isdigit() else 4

        # Build row data
        row_data = {
            "Epoch": f"{epoch_display:>{epoch_padding}}",
            "Step": f"{latest_step:>{step_padding}}/{total_steps}",
        }

        # Add loss metrics
        for name, values in train_metrics.items():
            if "loss" in name and len(values) > 0:
                latest_value = values[values[:, 0] == latest_step]
                if len(latest_value) > 0:
                    short_name = (
                        name.replace("box_l1_loss", "L1")
                        .replace("box_giou_loss", "GIoU")
                        .replace("class_focal_loss", "Focal")
                    )
                    row_data[short_name] = f"{latest_value[-1, 1]:.4f}"

        # Add validation loss if available
        for name, values in val_metrics.items():
            if "loss" in name and len(values) > 0:
                latest_value = values[values[:, 0] == latest_step]
                if len(latest_value) > 0:
                    short_name = (
                        name.replace("val/box_l1_loss", "val_L1")
                        .replace("val/box_giou_loss", "val_GIoU")
                        .replace("val/class_focal_loss", "val_Focal")
                        .replace("val/loss", "val_loss")
                    )
                    row_data[short_name] = f"{latest_value[-1, 1]:.4f}"

        # Format row with proper alignment
        row_parts = []
        for column_name, width in column_widths.items():
            value = row_data.get(column_name, "")
            row_parts.append(f"{value:>{width}}")

        row_line = "  ".join(row_parts)  # 2 spaces between columns

        # Add validation update indicator if this is a validation-only update
        if is_validation_update:
            print(f" ↑ {row_line}")  # Use arrow to indicate validation update
        else:
            print(f"   {row_line}")  # 3 spaces to align with rocket emoji

        # Update tracking for validation detection
        self.last_printed_step = latest_step
        self.last_printed_fractional_epoch = fractional_epoch

    def _print_console(self, trainer) -> None:
        """Print progress in either table format or single-line format."""
        if self.table_format:
            self._print_console_table(trainer)
        else:
            self._print_console_single_line(trainer)

    def _print_console_table(self, trainer) -> None:
        """Print progress in table format with aligned columns."""
        # Calculate column widths on first call
        if not self.column_widths:
            self.column_widths = self._calculate_column_widths(trainer)

        # Print header on first call
        if not self.header_printed:
            self._print_table_header(self.column_widths)
            self.header_printed = True

        # Print data row
        self._print_table_row(trainer, self.column_widths)

    def _print_console_single_line(self, trainer) -> None:
        """Print progress in single-line format (original behavior)."""
        train_metrics, val_metrics, extra_metrics = self.get_logged_metrics()

        # Get latest metrics for current epoch
        latest_step = trainer.global_step
        latest_epoch = trainer.current_epoch

        # Get total epochs and steps if available
        max_epochs = trainer.max_epochs if trainer.max_epochs else "?"
        total_steps = (
            trainer.max_steps
            if trainer.max_steps != -1
            else trainer.estimated_stepping_batches
            if hasattr(trainer, "estimated_stepping_batches")
            else "?"
        )

        # Calculate padding based on total digits
        epoch_padding = len(str(max_epochs)) if str(max_epochs).isdigit() else 1
        step_padding = len(str(total_steps)) if str(total_steps).isdigit() else 4

        # Build progress line with proper padding
        progress_parts = [
            f"Epoch {latest_epoch + 1:>{epoch_padding}}/{max_epochs}",
            f"Step {latest_step:>{step_padding}}/{total_steps}",
        ]

        # Add loss metrics
        for name, values in train_metrics.items():
            if "loss" in name and len(values) > 0:
                latest_value = values[values[:, 0] == latest_step]
                if len(latest_value) > 0:
                    short_name = (
                        name.replace("box_l1_loss", "L1")
                        .replace("box_giou_loss", "GIoU")
                        .replace("class_focal_loss", "Focal")
                    )
                    progress_parts.append(f"{short_name}: {latest_value[-1, 1]:.4f}")

        # Add validation loss if available
        for name, values in val_metrics.items():
            if "loss" in name and len(values) > 0:
                latest_value = values[values[:, 0] == latest_step]
                if len(latest_value) > 0:
                    short_name = (
                        name.replace("val/box_l1_loss", "val_L1")
                        .replace("val/box_giou_loss", "val_GIoU")
                        .replace("val/class_focal_loss", "val_Focal")
                        .replace("val/loss", "val_loss")
                    )
                    progress_parts.append(f"{short_name}: {latest_value[-1, 1]:.4f}")

        # Print each epoch on a new line
        progress_line = " | ".join(progress_parts)
        print(f"🚀 {progress_line}")

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
