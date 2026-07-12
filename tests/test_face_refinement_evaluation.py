import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from musubi_tuner.face_refinement.evaluation import evaluate_suite


class _DummyReward:
    def __init__(self, **_kwargs):
        self.pose_prototypes = {"frontal": torch.tensor([[1.0, 0.0]])}

    def detect_faces(self, _image):
        return [{"bbox": np.array([0, 0, 64, 64]), "score": 1.0,
                 "kps": np.array([[20, 20], [44, 20], [32, 32], [23, 44], [41, 44]], dtype=np.float32)}]

    @staticmethod
    def _face_area(_face): return 4096

    @staticmethod
    def crop_tensor(image, _face): return image

    @staticmethod
    def encode_faces(_crop): return torch.tensor([[1.0, 0.0]])

    @staticmethod
    def identity_scores(_embedding): return torch.tensor([0.8])


class FaceRefinementEvaluationTests(unittest.TestCase):
    def test_scores_fixed_suite_and_computes_baseline_delta(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); image = root / "sample.png"
            Image.new("RGB", (64, 64), "gray").save(image)
            suite = root / "suite.json"; refs = root / "refs.json"; baseline = root / "baseline.json"; output = root / "result.json"
            suite.write_text(json.dumps({"cases": [{"pose": "frontal", "seed": 1, "prompt": "portrait", "image": str(image)}]}), encoding="utf-8")
            refs.write_text(json.dumps({"reference_images": [{"path": "reference.png", "pose": "frontal", "enabled": True}]}), encoding="utf-8")
            baseline.write_text(json.dumps({"poses": {"frontal": {"overall_similarity": 0.7, "pose_similarity": 0.9, "pose_success_rate": 1.0, "detection_rate": 1.0}}}), encoding="utf-8")
            with patch("musubi_tuner.face_refinement.evaluation.FaceSimilarityReward", _DummyReward):
                result = evaluate_suite(suite, root, refs, "models", output, baseline)
            self.assertAlmostEqual(result["poses"]["frontal"]["overall_similarity"], 0.8)
            self.assertAlmostEqual(result["deltas"]["frontal"]["overall_similarity"], 0.1)
            self.assertTrue(result["cases"][0]["pose_success"])


if __name__ == "__main__": unittest.main()
