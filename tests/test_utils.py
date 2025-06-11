import sys
import types
import unittest

sys.modules.setdefault("glimmer.data", types.ModuleType("glimmer.data"))
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
        trainer = DummyTrainer(limit_train_batches=1.0, num_training_batches=4, max_epochs=10, max_steps=6)
        self.assertEqual(utils.get_max_epochs(trainer), 1.5)

    def test_get_max_epochs_no_steps(self):
        trainer = DummyTrainer(limit_train_batches=1.0, num_training_batches=4, max_epochs=3)
        self.assertEqual(utils.get_max_epochs(trainer), 3)

if __name__ == "__main__":
    unittest.main()
