import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from musubi_tuner.face_refinement.lora_validation import render_trigger_prompts, validate_krea2_lora
from musubi_tuner_gui import MusubiTunerGUI


class FaceRefinementLoraTests(unittest.TestCase):
    def test_recognizes_musubi_krea_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "subject.safetensors"
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.zeros(2, 2),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.zeros(2, 2),
            }, path)
            report = validate_krea2_lora(path)
            self.assertEqual(report["modules"], 1)

    def test_rejects_non_krea_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "other.safetensors"
            save_file({"model.weight": torch.zeros(2, 2)}, path)
            with self.assertRaisesRegex(ValueError, "Krea 2 LoRA"):
                validate_krea2_lora(path)

    def test_trigger_is_substituted_or_prefixed(self):
        self.assertEqual(
            render_trigger_prompts(["portrait of {trigger}", "cinematic portrait"], "subject_token"),
            ["portrait of subject_token", "subject_token, cinematic portrait"],
        )
        self.assertEqual(
            render_trigger_prompts(["[profile_left] cinematic portrait"], "subject_token"),
            ["[profile_left] subject_token, cinematic portrait"],
        )

    def test_job_resolver_accepts_state_folder_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "subject-000001-state"
            state.mkdir()
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.zeros(2, 2),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.zeros(2, 2),
            }, state / "model.safetensors")
            gui = object.__new__(MusubiTunerGUI)
            gui._continuation_state_candidates = lambda _job: [state]
            resolved = gui._resolve_job_face_lora({"settings_snapshot": {}})
            self.assertEqual(resolved, (state / "model.safetensors").resolve())


if __name__ == "__main__":
    unittest.main()
