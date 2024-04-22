# Copyright (c) 2018 - 2024 Immanuel Weber. Licensed under the MIT license (see LICENSE).

def get_num_training_batches(trainer):
    if (
        isinstance(trainer.limit_train_batches, int)
        or trainer.limit_train_batches == 0.0
    ):
        num_training_batches = min(
            trainer.num_training_batches, int(trainer.limit_train_batches)
        )
    elif trainer.num_training_batches != float("inf"):
        num_training_batches = int(
            trainer.num_training_batches * trainer.limit_train_batches
        )
    return num_training_batches


def get_max_epochs(trainer):
    max_epochs = trainer.max_epochs
    if trainer.max_steps is not None:
        num_training_batches = get_num_training_batches(trainer)
        max_epochs_from_steps = trainer.max_steps / num_training_batches
        max_epochs = min(max_epochs, max_epochs_from_steps)
    return max_epochs
