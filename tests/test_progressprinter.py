import sys
import types
import unittest

np = types.ModuleType("numpy"); np.ndarray = list; sys.modules["numpy"] = np
pd = types.ModuleType("pandas"); pd.DataFrame = type("DataFrame", (), {}); sys.modules["pandas"] = pd
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
torch = types.ModuleType("torch")
torch.utils = types.ModuleType("utils")
torch.utils.data = types.ModuleType("data")
torch.utils.data.DataLoader = type("DataLoader", (), {})
torch.utils.data.Dataset = type("Dataset", (), {})
sys.modules["pytorch_lightning"] = pl
sys.modules["pytorch_lightning.callbacks"] = pl.callbacks
sys.modules["pytorch_lightning.callbacks.progress"] = pl.callbacks.progress
sys.modules["pytorch_lightning.callbacks.progress.tqdm_progress"] = pl.callbacks.progress.tqdm_progress
sys.modules["pytorch_lightning.trainer"] = pl.trainer
sys.modules["pytorch_lightning.trainer.states"] = pl.trainer.states
sys.modules["torch"] = torch
sys.modules["torch.utils"] = torch.utils
sys.modules["torch.utils.data"] = torch.utils.data
mpl = types.ModuleType("matplotlib")
mpl.pyplot = types.ModuleType("pyplot")
sys.modules["matplotlib"] = mpl
sys.modules["matplotlib.pyplot"] = mpl.pyplot
sys.modules.setdefault("IPython", types.ModuleType("IPython"))
mod = types.ModuleType("IPython.display")
mod.display = lambda *args, **kwargs: None
sys.modules["IPython.display"] = mod
sys.modules.setdefault("glimmer.data", types.ModuleType("glimmer.data"))

from glimmer.lightning import progressprinter as pp


class ProgressPrinterTestCase(unittest.TestCase):
    def test_format_time_hours_minutes_seconds(self):
        self.assertEqual(pp.format_time(3661), "1:01:01")

    def test_format_time_zero(self):
        self.assertEqual(pp.format_time(0), "0:00:00")

if __name__ == "__main__":
    unittest.main()
