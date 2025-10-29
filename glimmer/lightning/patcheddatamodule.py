# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright (c) 2018 - 2025 Immanuel Weber


from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PatchedDataModule(pl.LightningDataModule):  # type: ignore[misc]
    @classmethod
    def from_datasets(
        cls,
        train_dataset: Dataset
        | Sequence[Dataset]
        | Mapping[str, Dataset]
        | None = None,
        val_dataset: Dataset | Sequence[Dataset] | None = None,
        test_dataset: Dataset | Sequence[Dataset] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Callable[..., Any] | None = None,
        persistent_workers: bool = False,
        test_batch_size: int | None = None,
        prefetch_factor: int | None = None,
        pin_memory: bool = True,
    ) -> "PatchedDataModule":
        r"""
        Create an instance from torch.utils.data.Dataset.

        Args:
            train_dataset: (optional) Dataset to be used for train_dataloader()
            val_dataset: (optional) Dataset or list of Dataset to be used for val_dataloader()
            test_dataset: (optional) Dataset or list of Dataset to be used for test_dataloader()
            batch_size: Batch size to use for each dataloader. Default is 1.
            num_workers: Number of subprocesses to use for data loading. 0 means that the
                data will be loaded in the main process. Number of CPUs available.
            collate_fn: Optional function to collate data samples into batches.
            persistent_workers: If True, dataloader workers will not be shut down after each epoch.
            test_batch_size: Optional different batch size for test dataloader.
            prefetch_factor: Number of batches loaded in advance by each worker.
                Default is None (which uses PyTorch's default of 2 when num_workers > 0).
            pin_memory: If True, the data loader will copy Tensors into pinned memory
                before returning them. Default is True for better GPU performance.

        """
        test_batch_size = test_batch_size if test_batch_size else batch_size

        def dataloader(
            ds: Dataset,
            batch_size: int,
            shuffle: bool = False,
            collate_fn: Callable[..., Any] | None = None,
            persistent_workers: bool = False,
        ) -> DataLoader[Any]:
            # Build dataloader kwargs
            dl_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "collate_fn": collate_fn,
                "persistent_workers": persistent_workers,
            }

            # Add prefetch_factor only if specified and num_workers > 0
            if prefetch_factor is not None and num_workers > 0:
                dl_kwargs["prefetch_factor"] = prefetch_factor

            return DataLoader(ds, **dl_kwargs)

        def train_dataloader() -> (
            DataLoader[Any] | dict[str, DataLoader[Any]] | list[DataLoader[Any]]
        ):
            if isinstance(train_dataset, Mapping):
                return {
                    key: dataloader(
                        ds,
                        batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                    )
                    for key, ds in train_dataset.items()
                }
            if isinstance(train_dataset, Sequence):
                return [
                    dataloader(
                        ds,
                        batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                    )
                    for ds in train_dataset
                ]
            return dataloader(
                train_dataset,
                batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
            )

        def val_dataloader() -> DataLoader[Any] | list[DataLoader[Any]]:
            if isinstance(val_dataset, Sequence):
                return [
                    dataloader(
                        ds,
                        batch_size,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                    )
                    for ds in val_dataset
                ]
            return dataloader(
                val_dataset,
                batch_size,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
            )

        def test_dataloader() -> DataLoader[Any] | list[DataLoader[Any]]:
            if isinstance(test_dataset, Sequence):
                return [
                    dataloader(
                        ds,
                        test_batch_size,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                    )
                    for ds in test_dataset
                ]
            return dataloader(
                test_dataset,
                test_batch_size,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
            )

        datamodule = cls()
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader
        return datamodule
