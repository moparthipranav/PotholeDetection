
import numpy as np

from feature_extractor import CameraConfig, EgoRoadFeatureExtractor
from fuzzy_system import EgoRoadFuzzySystem


def make_test_scene(h=720, w=1280):
    lane = np.zeros((h, w), dtype=np.uint8)
    drive = np.zeros((h, w), dtype=np.uint8)

    # Synthetic left/right lane boundaries.
    pts_l = np.array([[520, h], [550, 520], [590, 400]], np.int32)
    pts_r = np.array([[760, h], [730, 520], [690, 400]], np.int32)
    cv = np.zeros_like(lane)
    import cv2
    cv2.polylines(lane, [pts_l], False, 1, 8)
    cv2.polylines(lane, [pts_r], False, 1, 8)

    # Drivable corridor.
    poly = np.array([[520,h], [760,h], [690,400], [590,400]], np.int32)
    cv2.fillPoly(drive, [poly], 1)

    traffic = [
        {
            "bbox": [610, 455, 690, 555],
            "conf": 0.92,
            "class_id": 2,
            "class_name": "car",
        }
    ]

    potholes = [
        {
            "bbox": [570, 620, 650, 675],
            "conf": 0.88,
            "class_id": 0,
            "class_name": "pothole",
        }
    ]
    return lane, drive, traffic, potholes


if __name__ == "__main__":
    lane, drive, traffic, potholes = make_test_scene()

    extractor = EgoRoadFeatureExtractor(CameraConfig())
    fuzzy = EgoRoadFuzzySystem()

    f = extractor.extract(lane, drive, traffic, potholes, dt=1/15)
    y = fuzzy.infer(f)

    print("\n=== EgoRoad feature extraction ===")
    print(f)
    print("\n=== EgoRoad fuzzy decision ===")
    print(y)
