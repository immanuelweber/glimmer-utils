import sys
import types
import unittest

np = types.ModuleType("numpy")
np.ndarray = list
sys.modules["numpy"] = np
pd = types.ModuleType("pandas")
pd.DataFrame = type("DataFrame", (), {})
sys.modules["pandas"] = pd
sys.modules.setdefault("glimmer.data", types.ModuleType("glimmer.data"))
pl = types.ModuleType("pytorch_lightning")
pl.LightningModule = type("LightningModule", (), {})
pl.LightningDataModule = type("LightningDataModule", (), {})
pl.Trainer = type("Trainer", (), {})
pl.callbacks = types.ModuleType("callbacks")
pl.callbacks.Callback = type("Callback", (), {})
pl.callbacks.TQDMProgressBar = type("TQDMProgressBar", (), {})
pl.callbacks.progress = types.ModuleType("progress")
pl.callbacks.progress.tqdm_progress = types.ModuleType("tqdm_progress")
pl.callbacks.progress.tqdm_progress.Tqdm = type("Tqdm", (), {})
pl.trainer = types.ModuleType("trainer")
pl.trainer.states = types.ModuleType("states")
pl.trainer.states.TrainerFn = type("TrainerFn", (), {"VALIDATING": "validating"})
sys.modules["pytorch_lightning"] = pl
sys.modules["pytorch_lightning.callbacks"] = pl.callbacks
sys.modules["pytorch_lightning.callbacks.progress"] = pl.callbacks.progress
sys.modules["pytorch_lightning.callbacks.progress.tqdm_progress"] = (
    pl.callbacks.progress.tqdm_progress
)
sys.modules["pytorch_lightning.trainer"] = pl.trainer
sys.modules["pytorch_lightning.trainer.states"] = pl.trainer.states
sys.modules.setdefault("IPython", types.ModuleType("IPython"))
mod = types.ModuleType("IPython.display")
mod.display = lambda *args, **kwargs: None
sys.modules["IPython.display"] = mod
torch = types.ModuleType("torch")
torch.utils = types.ModuleType("utils")
torch.utils.data = types.ModuleType("data")
torch.utils.data.DataLoader = type("DataLoader", (), {})
torch.utils.data.Dataset = type("Dataset", (), {})
sys.modules["torch"] = torch
sys.modules["torch.utils"] = torch.utils
sys.modules["torch.utils.data"] = torch.utils.data
mpl = types.ModuleType("matplotlib")
mpl.pyplot = types.ModuleType("pyplot")
sys.modules["matplotlib"] = mpl
sys.modules["matplotlib.pyplot"] = mpl.pyplot
from glimmer.lightning import utils


class DummyTrainer:
    def __init__(self, limit_train_batches, num_training_batches, max_epochs, max_steps=None):
        self.limit_train_batches = limit_train_batches
        self.num_training_batches = num_training_batches
        self.max_epochs = max_epochs
        self.max_steps = max_steps


class UtilsTestCase(unittest.TestCase):
    def test_get_num_training_batches_int_limit(self):
        trainer = DummyTrainer(limit_train_batches=5, num_training_batches=10, max_epochs=1)
        self.assertEqual(utils.get_num_training_batches(trainer), 5)

    def test_get_num_training_batches_float_limit(self):
        trainer = DummyTrainer(limit_train_batches=0.5, num_training_batches=20, max_epochs=1)
        self.assertEqual(utils.get_num_training_batches(trainer), 10)

    def test_get_max_epochs_with_steps(self):
        trainer = DummyTrainer(
            limit_train_batches=1.0, num_training_batches=4, max_epochs=10, max_steps=6
        )
        self.assertEqual(utils.get_max_epochs(trainer), 1.5)

    def test_get_max_epochs_no_steps(self):
        trainer = DummyTrainer(limit_train_batches=1.0, num_training_batches=4, max_epochs=3)
        self.assertEqual(utils.get_max_epochs(trainer), 3)


if __name__ == "__main__":
    unittest.main()
