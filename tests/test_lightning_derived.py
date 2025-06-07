import sys, types
import unittest

sys.modules.setdefault("glimmer.data", types.ModuleType("glimmer.data"))
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
