import unittest

from musubi_tuner_gui import MusubiTunerGUI


class GuiTrainingProgressTests(unittest.TestCase):
    def test_ignores_model_loading_progress(self):
        line = "Loading krea2_turbo_fp8_scaled.safetensors: 100%|##########| 686/686 [00:23<00:00, 29.13key/s]"
        self.assertIsNone(MusubiTunerGUI._parse_main_training_progress(line))

    def test_parses_training_and_depth_metrics(self):
        line = "steps:  14%|#4| 47/329 [01:47<10:43, 2.28s/it, avr_loss=0.119, loss/diffusion=0.117, loss/depth_anchor=0.238]"
        parsed = MusubiTunerGUI._parse_main_training_progress(line)
        self.assertEqual((parsed["step"], parsed["total"]), (47, 329))
        self.assertAlmostEqual(parsed["loss"], 0.119)
        self.assertAlmostEqual(parsed["depth_loss"], 0.238)


if __name__ == "__main__":
    unittest.main()
