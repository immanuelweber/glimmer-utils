import sys, types
import unittest

np = types.ModuleType("numpy"); np.ndarray = list; sys.modules["numpy"] = np
pd = types.ModuleType("pandas"); pd.DataFrame = type("DataFrame", (), {}); sys.modules["pandas"] = pd
pl = types.ModuleType("pytorch_lightning")
pl.LightningModule = type("LightningModule", (), {})
pl.Trainer = type("Trainer", (), {})
pl.callbacks = types.ModuleType("callbacks")
pl.callbacks.Callback = type("Callback", (), {})
sys.modules["pytorch_lightning"] = pl
sys.modules["pytorch_lightning.callbacks"] = pl.callbacks
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
