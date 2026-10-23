
"""
Lightweight two-stage Mamdani fuzzy inference for EgoRoad.

No scikit-fuzzy dependency: this implementation uses NumPy only,
which makes deployment easier on Jetson Nano.

The linguistic partitions and the cascaded structure follow the
supplied EgoRoad paper:
- lane deviation: Centered / Drifting / Departed
- drivable ratio: Constrained / Adequate / Clear
- TTC: Critical / Caution / Safe
- pothole severity: Low / Medium / High
- Stage 1: Surface Hazard + Clearance Risk
- Stage 2: SAS + Braking + Steering
- centroid defuzzification
- EMA smoothing, gamma = 0.7
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import math
import numpy as np


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        y = np.maximum(y, np.minimum((x-a)/(b-a), 1.0))
    else:
        y = np.maximum(y, (x >= a).astype(float))
    if d > c:
        y = np.minimum(y, np.maximum((d-x)/(d-c), 0.0))
    else:
        y = np.minimum(y, (x <= c).astype(float))
    return y


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        y = np.maximum(y, np.minimum((x-a)/(b-a), 1.0))
    else:
        y = np.maximum(y, (x == b).astype(float))
    if c > b:
        y = np.minimum(y, np.maximum((c-x)/(c-b), 0.0))
    else:
        y = np.minimum(y, (x == b).astype(float))
    return y


@dataclass
class FuzzyOutput:
    surface_hazard: float
    clearance_risk: float
    sas_raw: float
    sas: float
    awareness: str
    braking_intensity: float
    steering: str
    lane_deviation: float
    drivable_ratio: float
    ttc: float
    pothole_severity: float


class EgoRoadFuzzySystem:
    def __init__(self, gamma: float = 0.7):
        self.gamma = gamma
        self.prev_sas = None

        # Discrete output universes keep computation deterministic and cheap.
        self.surface_u = np.linspace(0, 10, 501)
        self.clearance_u = np.linspace(0, 10, 501)
        self.sas_u = np.linspace(1, 10, 451)
        self.brake_u = np.linspace(0, 1, 201)

    @staticmethod
    def _one(v):
        return float(np.asarray(v).reshape(-1)[0])

    def fuzzify(self, lane_dev, drive_ratio, ttc, pothole):
        # Lane deviation [0, 2.5] m
        lane = {
            "centered": self._one(trapmf([lane_dev], 0.0, 0.0, 0.18, 0.35)),
            "drifting": self._one(trimf([lane_dev], 0.20, 0.55, 0.90)),
            "departed": self._one(trapmf([lane_dev], 0.70, 1.00, 2.50, 2.50)),
        }

        # Drivable ratio [0, 1]
        drive = {
            "constrained": self._one(trapmf([drive_ratio], 0.0, 0.0, 0.35, 0.55)),
            "adequate": self._one(trimf([drive_ratio], 0.35, 0.65, 0.85)),
            "clear": self._one(trapmf([drive_ratio], 0.70, 0.85, 1.0, 1.0)),
        }

        # TTC [0, 15+] s. Infinity is treated as 15 s.
        t = min(15.0, float(ttc) if np.isfinite(ttc) else 15.0)
        ttc_m = {
            "critical": self._one(trapmf([t], 0.0, 0.0, 1.8, 2.8)),
            "caution": self._one(trimf([t], 2.0, 4.5, 6.5)),
            "safe": self._one(trapmf([t], 5.5, 7.0, 15.0, 15.0)),
        }

        poth = {
            "low": self._one(trapmf([pothole], 0.0, 0.0, 0.15, 0.35)),
            "medium": self._one(trimf([pothole], 0.20, 0.50, 0.75)),
            "high": self._one(trapmf([pothole], 0.60, 0.80, 1.0, 1.0)),
        }
        return lane, drive, ttc_m, poth

    @staticmethod
    def _rule_surface(drive, poth):
        # 9-rule matrix: drive state x pothole severity.
        # Values are centroids for output Surface Hazard.
        table = {
            ("constrained", "low"): 5.0,
            ("constrained", "medium"): 7.0,
            ("constrained", "high"): 9.0,
            ("adequate", "low"): 2.5,
            ("adequate", "medium"): 5.5,
            ("adequate", "high"): 8.0,
            ("clear", "low"): 1.0,
            ("clear", "medium"): 4.0,
            ("clear", "high"): 7.0,
        }
        return table

    def _surface_stage(self, drive, poth):
        # Consequent membership functions.
        out = np.zeros_like(self.surface_u)
        table = self._rule_surface(drive, poth)

        for d_name, d_mu in drive.items():
            for p_name, p_mu in poth.items():
                alpha = min(d_mu, p_mu)
                if alpha <= 0:
                    continue
                center = table[(d_name, p_name)]
                consequent = trimf(
                    self.surface_u,
                    max(0, center-2.0),
                    center,
                    min(10, center+2.0),
                )
                out = np.maximum(out, np.minimum(alpha, consequent))

        return self._centroid(self.surface_u, out)

    def _clearance_stage(self, lane, ttc):
        # Nine lane/TTC rules. IMU jerk, when available, can be used as a
        # post-rule risk multiplier in an extension without changing this base.
        table = {
            ("centered", "safe"): 1.0,
            ("centered", "caution"): 4.0,
            ("centered", "critical"): 8.5,
            ("drifting", "safe"): 3.5,
            ("drifting", "caution"): 6.0,
            ("drifting", "critical"): 9.0,
            ("departed", "safe"): 6.5,
            ("departed", "caution"): 8.5,
            ("departed", "critical"): 10.0,
        }

        out = np.zeros_like(self.clearance_u)
        for l_name, l_mu in lane.items():
            for t_name, t_mu in ttc.items():
                alpha = min(l_mu, t_mu)
                if alpha <= 0:
                    continue
                center = table[(l_name, t_name)]
                consequent = trimf(
                    self.clearance_u,
                    max(0, center-2.0),
                    center,
                    min(10, center+2.0),
                )
                out = np.maximum(out, np.minimum(alpha, consequent))

        return self._centroid(self.clearance_u, out)

    def _master_stage(self, surface, clearance):
        # Fuzzify stage-1 crisp risks into Nominal/Caution/Critical.
        s = {
            "nominal": self._one(trapmf([surface], 0.0, 0.0, 2.5, 4.0)),
            "caution": self._one(trimf([surface], 2.5, 5.0, 7.5)),
            "critical": self._one(trapmf([surface], 6.0, 7.5, 10.0, 10.0)),
        }
        c = {
            "low": self._one(trapmf([clearance], 0.0, 0.0, 2.5, 4.0)),
            "medium": self._one(trimf([clearance], 2.5, 5.0, 7.5)),
            "high": self._one(trapmf([clearance], 6.0, 7.5, 10.0, 10.0)),
        }

        # Master rules. Higher risk -> lower SAS and stronger braking.
        rules = [
            ("critical", "high", 2.5, 1.0, "maintain"),
            ("critical", "medium", 2.5, 1.0, "maintain"),
            ("critical", "low", 4.5, 0.65, "evasive"),
            ("caution", "high", 4.0, 0.75, "maintain"),
            ("caution", "medium", 6.0, 0.45, "evasive"),
            ("caution", "low", 7.0, 0.20, "evasive"),
            ("nominal", "high", 6.5, 0.20, "maintain"),
            ("nominal", "medium", 8.0, 0.0, "maintain"),
            ("nominal", "low", 9.0, 0.0, "maintain"),
        ]

        agg_sas = np.zeros_like(self.sas_u)
        agg_brake = np.zeros_like(self.brake_u)
        steer_scores = {"maintain": 0.0, "evasive": 0.0}

        for s_name, s_mu in s.items():
            for c_name, c_mu in c.items():
                alpha = min(s_mu, c_mu)
                if alpha <= 0:
                    continue

                matched = [
                    r for r in rules
                    if r[0] == s_name and r[1] == c_name
                ][0]
                _, _, sas_center, brake_center, steer = matched

                sas_mf = trimf(
                    self.sas_u,
                    max(1, sas_center-2.0),
                    sas_center,
                    min(10, sas_center+2.0),
                )
                brake_mf = trimf(
                    self.brake_u,
                    max(0, brake_center-0.25),
                    brake_center,
                    min(1, brake_center+0.25),
                )
                agg_sas = np.maximum(agg_sas, np.minimum(alpha, sas_mf))
                agg_brake = np.maximum(agg_brake, np.minimum(alpha, brake_mf))
                steer_scores[steer] = max(steer_scores[steer], alpha)

        sas_raw = self._centroid(self.sas_u, agg_sas)
        brake = self._centroid(self.brake_u, agg_brake)
        steer = "evasive" if steer_scores["evasive"] > steer_scores["maintain"] else "maintain"
        return sas_raw, brake, steer

    @staticmethod
    def _centroid(u, mf):
        den = float(np.sum(mf))
        if den <= 1e-9:
            return float(np.mean(u))
        return float(np.sum(u * mf) / den)

    def infer(self, features) -> FuzzyOutput:
        lane, drive, ttc, poth = self.fuzzify(
            features.lane_deviation_m,
            features.drivable_ratio,
            features.ttc_s,
            features.pothole_severity,
        )

        surface = self._surface_stage(drive, poth)
        clearance = self._clearance_stage(lane, ttc)
        sas_raw, brake, steer = self._master_stage(surface, clearance)

        if self.prev_sas is None:
            sas = sas_raw
        else:
            sas = self.gamma * sas_raw + (1.0 - self.gamma) * self.prev_sas
        self.prev_sas = sas

        awareness = (
            "LOW" if sas <= 4.0
            else "MODERATE" if sas < 8.0
            else "HIGH"
        )

        return FuzzyOutput(
            surface_hazard=float(surface),
            clearance_risk=float(clearance),
            sas_raw=float(sas_raw),
            sas=float(sas),
            awareness=awareness,
            braking_intensity=float(brake),
            steering=steer,
            lane_deviation=float(features.lane_deviation_m),
            drivable_ratio=float(features.drivable_ratio),
            ttc=float(features.ttc_s),
            pothole_severity=float(features.pothole_severity),
        )
