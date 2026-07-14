import unittest
from unittest.mock import patch

import torch

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


class Krea2TurboCacheGuardTests(unittest.TestCase):
    def test_rejects_ram_cache_when_two_weight_copies_do_not_fit(self):
        trainer = Krea2NetworkTrainer()
        model = torch.nn.Linear(16, 16, bias=False)
        memory = type("Memory", (), {"available": 1})()
        with patch("psutil.virtual_memory", return_value=memory):
            self.assertFalse(trainer._turbo_cache_has_headroom(model))


if __name__ == "__main__":
    unittest.main()
