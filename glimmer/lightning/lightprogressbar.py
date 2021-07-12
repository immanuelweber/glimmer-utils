from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks.progress import reset, convert_inf
from tqdm.auto import tqdm

import math
from .utils import get_max_epochs


class LightProgressBar(ProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)

    def init_predict_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = tqdm(disable=True)
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for validation."""
        bar = tqdm(disable=True)
        return bar

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.max_epochs = math.ceil(get_max_epochs(trainer))

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(
            f"Epoch {trainer.current_epoch + 1}/{self.max_epochs}"
        )

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        reset(self.test_progress_bar, convert_inf(self.total_test_batches))
