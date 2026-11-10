"""
Laplacian Variance Scorer & Progressive Best-Shot Selection Module.
Theoretical Basis:
- Discrete 2nd derivative operator: \nabla^2 I = d^2 I / dx^2 + d^2 I / dy^2
- Sharpness Metric: S = Var(\nabla^2 I)
  High variance corresponds to sharp high-frequency edges; low variance indicates motion blur.
- Quality Score: Q = S * (conf ** 1.5)
- Progressive Update:
  Immediately snapshot at frame 4 (temporary evidence), then overwrite if subsequent
  frames yield higher quality score (e.g., weapon held steady or closer to camera).
"""

import cv2
import numpy as np


def compute_laplacian_variance(image_roi: np.ndarray, kernel_size: int = 3) -> float:
    if image_roi is None or image_roi.size == 0:
        return 0.0

    if len(image_roi.shape) == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi

    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    variance = float(lap.var())
    return variance


def crop_weapon_roi(frame: np.ndarray, bbox: list, padding: int = 15) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return frame[y1:y2, x1:x2].copy()


def calculate_quality_score(frame: np.ndarray, bbox: list, conf: float) -> tuple[float, float, np.ndarray]:
    roi = crop_weapon_roi(frame, bbox)
    if roi.size == 0:
        return 0.0, 0.0, roi

    area = roi.shape[0] * roi.shape[1]
    if area < 100:
        return 0.0, 0.0, roi

    lap_var = compute_laplacian_variance(roi)
    area_factor = min(2.0, np.sqrt(area) / 50.0)

    quality_score = lap_var * (conf ** 1.5) * area_factor
    return quality_score, lap_var, roi


class ProgressiveBestShotManager:
    def __init__(self, improvement_threshold: float = 1.10):
        self.improvement_threshold = improvement_threshold
        self.best_shots = {}

    def evaluate_and_update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: list,
        conf: float,
        is_first_confirmation: bool = False
    ) -> tuple[bool, dict]:
        quality_score, lap_var, roi = calculate_quality_score(frame, bbox, conf)
        curr = self.best_shots.get(track_id)

        if curr is None or is_first_confirmation:
            info = {
                "track_id": track_id,
                "quality_score": quality_score,
                "laplacian_var": lap_var,
                "conf": conf,
                "frame": frame.copy(),
                "crop": roi,
                "bbox": bbox,
                "is_temp": is_first_confirmation,
                "update_count": 1
            }
            self.best_shots[track_id] = info
            return True, info

        if quality_score > curr["quality_score"] * self.improvement_threshold:
            info = {
                "track_id": track_id,
                "quality_score": quality_score,
                "laplacian_var": lap_var,
                "conf": conf,
                "frame": frame.copy(),
                "crop": roi,
                "bbox": bbox,
                "is_temp": False,
                "update_count": curr.get("update_count", 1) + 1
            }
            self.best_shots[track_id] = info
            return True, info

        return False, curr

    def clear_track(self, track_id: int):
        self.best_shots.pop(track_id, None)
