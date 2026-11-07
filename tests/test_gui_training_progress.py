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

    def test_parses_and_formats_repa_metrics(self):
        parsed = MusubiTunerGUI._parse_main_training_progress(
            "steps: 1%|#| 1/100 [00:03<05:00, 3.0s/it, avr_loss=0.9, "
            "loss/repa=0.54, loss/repa_weighted=0.27, repa/cosine_similarity=0.46]"
        )
        self.assertAlmostEqual(parsed["repa_loss"], 0.54)
        self.assertAlmostEqual(parsed["repa_weighted"], 0.27)
        self.assertAlmostEqual(parsed["repa_similarity"], 0.46)
        self.assertEqual(MusubiTunerGUI._repa_monitor_text(parsed),
                         "REPA: active · loss 0.5400 · added loss 0.27000\nVisual teacher similarity: 0.460")

    def test_repa_monitor_preserves_zero_and_does_not_invent_missing_values(self):
        self.assertIn("loss 0.0000 · added loss 0.00000", MusubiTunerGUI._repa_monitor_text({
            "repa_loss": 0.0, "repa_weighted": 0.0,
        }))
        self.assertIn("similarity: —", MusubiTunerGUI._repa_monitor_text({"repa_loss": 0.0}))

    def test_epoch_markers_only_include_reached_boundaries(self):
        markers = MusubiTunerGUI._epoch_marker_positions(1320, 3, 467)

        self.assertEqual(markers, [(440, 2)])

    def test_epoch_markers_include_each_reached_boundary(self):
        markers = MusubiTunerGUI._epoch_marker_positions(1320, 3, 900)

        self.assertEqual(markers, [(440, 2), (880, 3)])

    def test_epoch_markers_are_capped_for_long_runs(self):
        markers = MusubiTunerGUI._epoch_marker_positions(1000, 1000, 1000)

        self.assertLessEqual(len(markers), 8)
        self.assertEqual(markers[0], (1, 2))
        self.assertEqual(markers[-1], (999, 1000))


if __name__ == "__main__":
    unittest.main()
