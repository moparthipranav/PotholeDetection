
"""
YOLOP adapter template.

YOLOP implementations differ in their return format. Keep the model
inference code you already have and convert only its outputs into this
canonical dictionary:

{
    "lane_mask": HxW binary mask,
    "drive_mask": HxW binary mask,
    "traffic": [
        {"bbox":[x1,y1,x2,y2], "conf":0.9,
         "class_id":2, "class_name":"car"}
    ]
}

The adapter below accepts already-decoded outputs. This avoids changing
your existing YOLOP implementation.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def make_yolop_output(
    lane_mask: np.ndarray,
    drive_mask: np.ndarray,
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "lane_mask": lane_mask,
        "drive_mask": drive_mask,
        "traffic": detections,
    }


def normalize_detection(
    bbox, conf, class_id, class_name=None
):
    return {
        "bbox": [float(v) for v in bbox],
        "conf": float(conf),
        "class_id": int(class_id),
        "class_name": str(class_name if class_name is not None else class_id),
    }
