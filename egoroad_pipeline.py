
"""
Drop-in orchestration layer.

Expected YOLOP adapter output:
{
    "lane_mask": HxW numpy array,
    "drive_mask": HxW numpy array,
    "traffic": [
        {"bbox":[x1,y1,x2,y2], "conf":0.91, "class_id":2, "class_name":"car"}
    ]
}

Expected YOLOv8 pothole output:
[
    {"bbox":[x1,y1,x2,y2], "conf":0.88, "class_name":"pothole"}
]

Use:
    from egoroad_pipeline import EgoRoadPipeline
    pipe = EgoRoadPipeline(...)
    features, decision = pipe.process(yolop, potholes)
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, List, Optional

from feature_extractor import CameraConfig, EgoRoadFeatureExtractor
from fuzzy_system import EgoRoadFuzzySystem


class EgoRoadPipeline:
    def __init__(self, camera_config: Optional[CameraConfig] = None):
        self.camera = camera_config or CameraConfig()
        self.extractor = EgoRoadFeatureExtractor(self.camera)
        self.fuzzy = EgoRoadFuzzySystem(gamma=0.7)

    def process(
        self,
        yolop_output: Dict[str, Any],
        pothole_detections: Optional[List[Dict[str, Any]]] = None,
        dt: Optional[float] = None,
    ):
        features = self.extractor.extract(
            lane_mask=yolop_output["lane_mask"],
            drive_mask=yolop_output["drive_mask"],
            traffic_detections=yolop_output.get("traffic", []),
            pothole_detections=pothole_detections or [],
            dt=dt,
        )
        decision = self.fuzzy.infer(features)
        return features, decision

    @staticmethod
    def to_dict(features, decision):
        return {
            "features": asdict(features),
            "decision": asdict(decision),
        }
