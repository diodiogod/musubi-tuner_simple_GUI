import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from musubi_tuner import krea2_generate_image


class Krea2PromptPreencodingTests(unittest.TestCase):
    def test_unique_prompts_are_encoded_once_and_cached_on_cpu(self):
        prompts = [
            SimpleNamespace(prompt="front portrait", negative_prompt="", guidance_scale=1.0),
            SimpleNamespace(prompt="front portrait", negative_prompt="", guidance_scale=1.0),
            SimpleNamespace(prompt="side portrait", negative_prompt="", guidance_scale=1.0),
        ]
        encoded = (
            torch.arange(12).reshape(2, 2, 3),
            torch.ones((2, 2), dtype=torch.bool),
            None,
            None,
        )
        with patch.object(krea2_generate_image, "encode", return_value=encoded) as encode:
            cache = krea2_generate_image.preencode_prompt_args(object(), prompts, "cuda")

        encode.assert_called_once()
        self.assertEqual(encode.call_args.args[1], ["front portrait", "side portrait"])
        self.assertEqual(len(cache), 2)
        front = cache[krea2_generate_image._prompt_embedding_key(prompts[0])]
        self.assertEqual(front[0].shape, (1, 2, 3))
        self.assertEqual(front[0].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
