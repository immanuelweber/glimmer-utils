# Copyright (c) 2018 - 2025 Immanuel Weber. Licensed under the MIT license (see LICENSE).
from typing import Any

import torch as th


def ld_to_dl(lst: list[dict[str, Any]]) -> dict[str, list[Any]]:
    # list of dicts to dict of lists
    return {key: [dic[key] for dic in lst] for key in lst[0]}


def stack_n_pad_tensors(tensors: list[th.Tensor]) -> th.Tensor:
    max_shape = th.as_tensor([list(tensor.shape) for tensor in tensors]).max(0)[0]
    stack_shape = [len(tensors)] + max_shape.tolist()
    stack = th.zeros(stack_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    for tn, sl in zip(tensors, stack):
        sl[: tn.shape[0], : tn.shape[1], : tn.shape[2]].copy_(tn)
    return stack


def dict_collate_fn(
    batch: Any,
    stackable_inputs: list[str] | None = None,
    stackable_targets: list[str] | None = None,
    tensorable_inputs: list[str] | None = None,
    tensorable_targets: list[str] | None = None,
) -> list[dict[str, th.Tensor | list[Any]]]:
    """
    Custom collate function for batching dictionary-based datasets.

    Args:
        batch: Batch of data samples to collate.
        stackable_inputs (list[str] | None): Keys for inputs that should be stacked and padded.
            Defaults to ["image", "mask", "segmentation"].
        stackable_targets (list[str] | None): Keys for targets that should be stacked and padded.
            Defaults to empty list.
        tensorable_inputs (list[str] | None): Keys for inputs that should be converted to tensors.
            Defaults to ["image_id"].
        tensorable_targets (list[str] | None): Keys for targets that should be converted to tensors.
            Defaults to empty list.

    Returns:
        list[dict[str, th.Tensor | list]]: A list containing [inputs, targets] dictionaries.
    """
    if stackable_inputs is None:
        stackable_inputs = ["image", "mask", "segmentation"]
    if stackable_targets is None:
        stackable_targets = []
    if tensorable_inputs is None:
        tensorable_inputs = ["image_id"]
    if tensorable_targets is None:
        tensorable_targets = []

    batch = list(zip(*batch))
    inputs = ld_to_dl(batch[0])
    for k in stackable_inputs:
        if k in inputs:
            inputs[k] = stack_n_pad_tensors(inputs[k])
    for k in tensorable_inputs:
        if k in inputs:
            inputs[k] = th.as_tensor(inputs[k])
    targets = ld_to_dl(batch[1])
    for k in stackable_targets:
        if k in targets:
            targets[k] = stack_n_pad_tensors(targets[k])
    for k in tensorable_targets:
        if k in targets:
            targets[k] = th.as_tensor(targets[k])
    return [inputs, targets]
