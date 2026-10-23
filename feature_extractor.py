
"""
EgoRoad feature extraction.

Inputs are deliberately model-agnostic:
    lane_mask: HxW binary/0..1 lane segmentation
    drive_mask: HxW binary/0..1 drivable-area segmentation
    traffic_detections: list of dicts:
        {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int,
         "class_name": str}
    pothole_detections: same bbox/conf format

The geometry follows the EgoRoad paper:
- lane deviation from the lane boundaries near the lower image horizon
- drivable corridor ratio
- pinhole ground-plane distance and TTC
- pothole severity from confidence, distance and bbox area

The extractor is CPU-only and suitable for running after YOLOP/YOLOv8 inference.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import time

import cv2
import numpy as np


@dataclass
class CameraConfig:
    # Replace these with your calibrated values.
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 640.0
    cy: float = 360.0

    # Camera geometry.
    height_m: float = 1.50
    pitch_deg: float = 2.0

    # Forward hazard horizon.
    z_max_m: float = 30.0

    # Used only when lane boundaries cannot be recovered.
    fallback_lane_half_width_m: float = 1.75


@dataclass
class ExtractedFeatures:
    lane_deviation_m: float
    lane_state: str
    drivable_ratio: float

    nearest_object_distance_m: float
    nearest_object_class: str
    nearest_object_conf: float
    ttc_s: float

    pothole_severity: float
    nearest_pothole_distance_m: float
    pothole_conf: float

    traffic_count: int
    ego_corridor_occupancy: float

    lane_confidence: float
    feature_valid: bool
    timestamp: float


class SimpleObjectTracker:
    """Very small centroid tracker. No deep tracker dependency."""

    def __init__(self, max_distance_px: float = 100.0, max_age: int = 8):
        self.max_distance_px = max_distance_px
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    def update(self, detections: List[Dict[str, Any]], dt: float):
        now_ids = set()
        det_centers = [self._center(d["bbox"]) for d in detections]

        # Greedy nearest-neighbour association.
        used = set()
        for tid, tr in list(self.tracks.items()):
            if not det_centers:
                tr["age"] += 1
                continue

            distances = [
                float(np.linalg.norm(c - tr["center"])) if i not in used else 1e9
                for i, c in enumerate(det_centers)
            ]
            j = int(np.argmin(distances))
            if distances[j] <= self.max_distance_px:
                new_center = det_centers[j]
                velocity = (new_center - tr["center"]) / max(dt, 1e-3)
                tr["velocity_px_s"] = 0.7 * tr["velocity_px_s"] + 0.3 * velocity
                tr["center"] = new_center
                tr["bbox"] = detections[j]["bbox"]
                tr["class_name"] = detections[j].get("class_name", "object")
                tr["age"] = 0
                tr["seen"] += 1
                detections[j]["track_id"] = tid
                used.add(j)
                now_ids.add(tid)
            else:
                tr["age"] += 1

        for i, d in enumerate(detections):
            if i in used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "center": det_centers[i],
                "bbox": d["bbox"],
                "velocity_px_s": np.zeros(2, dtype=np.float32),
                "age": 0,
                "seen": 1,
                "class_name": d.get("class_name", "object"),
            }
            d["track_id"] = tid
            now_ids.add(tid)

        for tid in list(self.tracks):
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        return detections


class EgoRoadFeatureExtractor:
    def __init__(
        self,
        camera: CameraConfig,
        min_lane_pixels: int = 30,
        corridor_half_width_px: float = 0.18,
    ):
        self.cam = camera
        self.min_lane_pixels = min_lane_pixels
        self.corridor_half_width_px = corridor_half_width_px
        self.tracker = SimpleObjectTracker()
        self.prev_time = None

    @staticmethod
    def _binary(mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m

    @staticmethod
    def _lane_boundaries(lane_mask: np.ndarray, ego_x: float):
        """
        Recover the ego-lane left/right boundaries from a lane mask.
        At several lower-image scanlines, choose the nearest lane pixels
        to the ego center on each side, then fit x(y) polynomials.
        """
        h, w = lane_mask.shape
        ys = np.linspace(int(h * 0.55), h - 8, 10).astype(int)

        left_pts, right_pts = [], []
        for y in ys:
            xs = np.flatnonzero(lane_mask[y] > 0)
            if xs.size == 0:
                continue
            left = xs[xs < ego_x]
            right = xs[xs > ego_x]
            if left.size:
                left_pts.append((y, int(left.max())))
            if right.size:
                right_pts.append((y, int(right.min())))

        return left_pts, right_pts

    @staticmethod
    def _fit_line(points: List[Tuple[int, int]]):
        if len(points) < 2:
            return None
        y = np.array([p[0] for p in points], dtype=np.float32)
        x = np.array([p[1] for p in points], dtype=np.float32)
        deg = 2 if len(points) >= 4 else 1
        coeff = np.polyfit(y, x, deg)
        return coeff

    @staticmethod
    def _eval_poly(coeff, y):
        return float(np.polyval(coeff, y))

    def _lane_geometry(self, lane_mask: np.ndarray):
        h, w = lane_mask.shape
        ego_x = self.cam.cx if 0 <= self.cam.cx < w else w * 0.5
        left, right = self._lane_boundaries(lane_mask, ego_x)

        if len(left) < 2 or len(right) < 2:
            return None

        lc = self._fit_line(left)
        rc = self._fit_line(right)
        y0 = int(h * 0.88)
        xl = self._eval_poly(lc, y0)
        xr = self._eval_poly(rc, y0)

        if xr <= xl + 5:
            return None

        lane_center_px = 0.5 * (xl + xr)
        width_px = xr - xl

        # Approximate local lateral ground resolution from lane width.
        # Assumes a nominal 3.5 m lane when no explicit homography is supplied.
        sx = 3.5 / max(width_px, 1.0)
        deviation_m = (ego_x - lane_center_px) * sx

        confidence = min(1.0, 0.5 * (len(left) / 10.0) + 0.5 * (len(right) / 10.0))
        return {
            "left_x": xl,
            "right_x": xr,
            "center_x": lane_center_px,
            "deviation_m": float(abs(deviation_m)),
            "signed_deviation_m": float(deviation_m),
            "confidence": float(confidence),
            "y_scan": y0,
        }

    def _corridor_ratio(self, drive_mask, lane_geom):
        h, w = drive_mask.shape
        y1, y2 = int(h * 0.55), int(h * 0.92)
        if y2 <= y1:
            return 0.0

        corridor = np.zeros_like(drive_mask, dtype=np.uint8)
        if lane_geom is None:
            cx = int(self.cam.cx if 0 <= self.cam.cx < w else w / 2)
            half = int(w * self.corridor_half_width_px)
            corridor[y1:y2, max(0, cx-half):min(w, cx+half)] = 1
        else:
            # Use the fitted lower lane corridor, interpolating linearly
            # between scanline anchors at the lower half of the image.
            # A conservative fixed-width corridor is used around the lane center.
            cx = int(round(lane_geom["center_x"]))
            half = max(10, int(round((lane_geom["right_x"] - lane_geom["left_x"]) * 0.42)))
            corridor[y1:y2, max(0, cx-half):min(w, cx+half)] = 1

        n = int(corridor.sum())
        if n == 0:
            return 0.0
        return float((drive_mask * corridor).sum() / n)

    def _distance_from_bottom_pixel(self, v_bottom: float) -> float:
        """
        Equation (3) from the supplied EgoRoad paper.
        """
        theta = math.radians(self.cam.pitch_deg)
        den = ((v_bottom - self.cam.cy) * math.cos(theta)
               + self.cam.fy * math.sin(theta))
        if abs(den) < 1e-6:
            return float("inf")
        z = (self.cam.fy * self.cam.height_m) / den
        return float(z) if z > 0 else float("inf")

    @staticmethod
    def _inside_ego_corridor(box, lane_geom, w):
        x1, _, x2, _ = box
        xc = 0.5 * (x1 + x2)
        if lane_geom is None:
            return abs(xc - w * 0.5) < w * 0.22
        return lane_geom["left_x"] <= xc <= lane_geom["right_x"]

    def _ttc(self, detections, lane_geom, image_w, dt):
        if not detections:
            return float("inf"), "none", 0.0, float("inf")

        candidates = []
        for d in detections:
            box = d["bbox"]
            if not self._inside_ego_corridor(box, lane_geom, image_w):
                continue
            x1, y1, x2, y2 = box
            z = self._distance_from_bottom_pixel(y2)
            if not np.isfinite(z) or z > self.cam.z_max_m:
                continue
            candidates.append((z, d))

        if not candidates:
            return float("inf"), "none", 0.0, float("inf")

        candidates.sort(key=lambda x: x[0])
        z, d = candidates[0]

        # Estimate closing speed from tracked pixel velocity using
        # finite difference in metric depth. This is deliberately
        # conservative; a better system should use calibrated tracking.
        tid = d.get("track_id")
        zdot = 0.0
        if tid is not None and tid in self.tracker.tracks:
            vy = float(self.tracker.tracks[tid]["velocity_px_s"][1])
            # Local numerical derivative of Z(v_bottom).
            vb = float(d["bbox"][3])
            eps = max(1.0, abs(vy) * dt)
            z2 = self._distance_from_bottom_pixel(vb + eps)
            if np.isfinite(z2):
                zdot = (z2 - z) / max(dt, 1e-3)

        ttc = z / (-zdot) if zdot < 0 else float("inf")
        return float(ttc), str(d.get("class_name", "object")), float(d.get("conf", 0.0)), float(z)

    def _pothole_severity(self, potholes, image_w, image_h):
        best_h = 0.0
        best_z = float("inf")
        best_conf = 0.0

        for d in potholes or []:
            x1, y1, x2, y2 = d["bbox"]
            conf = float(d.get("conf", 0.0))
            z = self._distance_from_bottom_pixel(y2)
            if not np.isfinite(z) or z > self.cam.z_max_m:
                continue

            area_ratio = max(0.0, (x2-x1) * (y2-y1) / max(image_w * image_h, 1))
            beta = 8.0
            h = min(
                1.0,
                conf
                * max(0.0, 1.0 - z / self.cam.z_max_m)
                * (1.0 + beta * area_ratio),
            )
            if h > best_h:
                best_h, best_z, best_conf = h, z, conf

        return float(best_h), float(best_z), float(best_conf)

    def extract(
        self,
        lane_mask: np.ndarray,
        drive_mask: np.ndarray,
        traffic_detections: Optional[List[Dict[str, Any]]] = None,
        pothole_detections: Optional[List[Dict[str, Any]]] = None,
        dt: Optional[float] = None,
    ) -> ExtractedFeatures:

        lane_mask = self._binary(lane_mask)
        drive_mask = self._binary(drive_mask)

        if lane_mask is None or drive_mask is None:
            raise ValueError("lane_mask and drive_mask are required")

        h, w = lane_mask.shape
        if drive_mask.shape != lane_mask.shape:
            drive_mask = cv2.resize(
                drive_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )

        traffic = [dict(x) for x in (traffic_detections or [])]
        if dt is None:
            now = time.time()
            dt = 1/30 if self.prev_time is None else max(1e-3, now - self.prev_time)
            self.prev_time = now

        traffic = self.tracker.update(traffic, dt)

        lane_geom = self._lane_geometry(lane_mask)
        if lane_geom is None:
            lane_dev = 1.25
            lane_state = "unknown"
            lane_conf = 0.0
        else:
            lane_dev = abs(lane_geom["signed_deviation_m"])
            lane_conf = lane_geom["confidence"]
            lane_state = (
                "centered" if lane_dev <= 0.25
                else "drifting" if lane_dev <= 0.75
                else "departed"
            )

        drivable_ratio = self._corridor_ratio(drive_mask, lane_geom)

        ttc, obj_class, obj_conf, obj_z = self._ttc(
            traffic, lane_geom, w, dt
        )

        pothole_h, pothole_z, pothole_conf = self._pothole_severity(
            pothole_detections or [], w, h
        )

        # Approximate occupancy in the ego corridor.
        ego_boxes = [
            d["bbox"] for d in traffic
            if self._inside_ego_corridor(d["bbox"], lane_geom, w)
        ]
        corridor_occupancy = 0.0
        if ego_boxes:
            occ_area = sum(
                max(0.0, (b[2]-b[0]) * (b[3]-b[1])) for b in ego_boxes
            )
            corridor_occupancy = min(1.0, occ_area / max((w*0.30)*(h*0.35), 1.0))

        return ExtractedFeatures(
            lane_deviation_m=lane_dev,
            lane_state=lane_state,
            drivable_ratio=drivable_ratio,
            nearest_object_distance_m=obj_z,
            nearest_object_class=obj_class,
            nearest_object_conf=obj_conf,
            ttc_s=ttc,
            pothole_severity=pothole_h,
            nearest_pothole_distance_m=pothole_z,
            pothole_conf=pothole_conf,
            traffic_count=len(traffic),
            ego_corridor_occupancy=corridor_occupancy,
            lane_confidence=lane_conf,
            feature_valid=(lane_conf > 0.0 or drivable_ratio > 0.0),
            timestamp=time.time(),
        )
