import unittest

import torch

from musubi_tuner.krea2.krea2_utils import _make_krea2_prequant_lora_merge_hook


class Krea2PrequantLoraMergeTests(unittest.TestCase):
    def test_scaled_fp8_weight_is_restored_before_lora_merge(self):
        key = "blocks.0.attn.wq.weight"
        hook = _make_krea2_prequant_lora_merge_hook(
            torch.float32,
            {key + "_scale": torch.tensor(0.25)},
        )
        names, values = hook(key, torch.tensor([[4.0]], dtype=torch.float8_e4m3fn))
        self.assertEqual(names, [key])
        torch.testing.assert_close(values[0], torch.tensor([[1.0]]))

        names, values = hook(key + "_scale", torch.tensor(0.25))
        self.assertEqual(names, [])
        self.assertIsNone(values)


if __name__ == "__main__":
    unittest.main()
