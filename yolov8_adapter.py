
"""
YOLOv8 -> EgoRoad adapter.

Usage with Ultralytics:
    result = pothole_model(frame, verbose=False)[0]
    potholes = ultralytics_result_to_detections(result, names=pothole_model.names)

For a pothole-only model, every detection can be retained.
"""

from __future__ import annotations
from typing import Any, Dict, List


def ultralytics_result_to_detections(result, names=None) -> List[Dict[str, Any]]:
    detections = []
    if result is None or result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    for box, c, k in zip(xyxy, conf, cls):
        class_name = str(names[k]) if names is not None else str(k)
        detections.append({
            "bbox": [float(x) for x in box],
            "conf": float(c),
            "class_id": int(k),
            "class_name": class_name,
        })
    return detections
