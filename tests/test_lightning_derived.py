import sys
import types
import unittest
np = types.ModuleType("numpy"); np.ndarray = list; sys.modules["numpy"] = np
pd = types.ModuleType("pandas"); pd.DataFrame = type("DataFrame", (), {}); sys.modules["pandas"] = pd
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
sys.modules["pytorch_lightning.callbacks.progress.tqdm_progress"] = pl.callbacks.progress.tqdm_progress
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
from glimmer.lightning import lightning_derived as ld


class DummyOptimizer:
    def __init__(self, lr_values):
        self.param_groups = lr_values

class DummyScheduler:
    def __init__(self, optimizer, interval="epoch"):
        self.optimizer = optimizer
        self.__class__.__name__ = "DummyScheduler"
        self.interval = interval

class DummyLRSchedulerConfig:
    def __init__(self, optimizer, name=None, interval="epoch"):
        self.scheduler = DummyScheduler(optimizer, interval)
        self.name = name
        self.interval = interval

class LightningDerivedTestCase(unittest.TestCase):
    def test_get_scheduler_names_unique(self):
        optim = DummyOptimizer([{"lr": 0.1}])
        sched = DummyLRSchedulerConfig(optim, name="sched1")
        self.assertEqual(ld.get_scheduler_names([sched]), ["sched1"])

    def test_get_scheduler_names_multiple_param_groups(self):
        optim = DummyOptimizer([{"lr": 0.1}, {"lr": 0.2}])
        sched = DummyLRSchedulerConfig(optim)
        names = ld.get_scheduler_names([sched])
        self.assertEqual(names, ["lr-DummyOptimizer-DummyScheduler/pg1", "lr-DummyOptimizer-DummyScheduler/pg2"])

    def test_get_lrs_interval_filter(self):
        optim1 = DummyOptimizer([{"lr": 0.1}])
        sched1 = DummyLRSchedulerConfig(optim1, name="s1", interval="step")
        optim2 = DummyOptimizer([{"lr": 0.2}])
        sched2 = DummyLRSchedulerConfig(optim2, name="s2", interval="epoch")
        names = ld.get_scheduler_names([sched1, sched2])
        lrs = ld.get_lrs([sched1, sched2], names, interval="step")
        self.assertEqual(lrs, {"s1": 0.1})

if __name__ == "__main__":
    unittest.main()
