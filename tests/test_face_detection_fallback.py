import numpy as np

from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward


class _TightCropDetector:
    def __init__(self):
        self.shapes = []

    def detect(self, image, **_kwargs):
        self.shapes.append(image.shape)
        if len(self.shapes) == 1:
            return np.empty((0, 5), dtype=np.float32), None
        bbox = np.array([[35, 30, 85, 90, 0.9]], dtype=np.float32)
        kps = np.array([[[45, 45], [70, 45], [58, 58], [48, 72], [68, 72]]], dtype=np.float32)
        return bbox, kps


def test_tight_portrait_retries_with_padding_and_maps_detection_back():
    reward = object.__new__(FaceSimilarityReward)
    reward.detector = _TightCropDetector()
    reward.det_size = (640, 640)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    faces = reward.detect_faces(image)

    assert reward.detector.shapes == [(80, 100, 3), (120, 150, 3)]
    assert len(faces) == 1
    np.testing.assert_allclose(faces[0]["bbox"], [10, 10, 60, 70])
    np.testing.assert_allclose(faces[0]["kps"][0], [20, 25])
    assert np.isclose(faces[0]["score"], 0.9)
