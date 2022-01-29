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

# Modifications copyright (C) 2021-2022 Immanuel Weber


import pytorch_lightning as pl
from typing import Optional, Union, Sequence, Mapping
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PatchedDataModule(pl.LightningDataModule):
    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[
            Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]
        ] = None,
        val_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn=None,
        persistent_workers: bool = False,
        test_batch_size: Optional[int] = None,
    ):
        r"""
        Create an instance from torch.utils.data.Dataset.

        Args:
            train_dataset: (optional) Dataset to be used for train_dataloader()
            val_dataset: (optional) Dataset or list of Dataset to be used for val_dataloader()
            test_dataset: (optional) Dataset or list of Dataset to be used for test_dataloader()
            batch_size: Batch size to use for each dataloader. Default is 1.
            num_workers: Number of subprocesses to use for data loading. 0 means that the
                data will be loaded in the main process. Number of CPUs available.

        """
        test_batch_size = test_batch_size if test_batch_size else batch_size

        def dataloader(ds, batch_size, shuffle=False, collate_fn=None, persistent_workers=False):
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
            )

        def train_dataloader():
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

        def val_dataloader():
            if isinstance(val_dataset, Sequence):
                return [
                    dataloader(
                        ds, batch_size, collate_fn=collate_fn, persistent_workers=persistent_workers
                    )
                    for ds in val_dataset
                ]
            return dataloader(
                val_dataset,
                batch_size,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
            )

        def test_dataloader():
            if isinstance(test_dataset, Sequence):
                return [
                    dataloader(
                        ds, test_batch_size, collate_fn=collate_fn, persistent_workers=persistent_workers
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
