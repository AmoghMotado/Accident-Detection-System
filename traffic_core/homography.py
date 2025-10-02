from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

Point = Tuple[float, float]

def compute_h(image_points: List[Point], world_points_m: List[Point]) -> np.ndarray:
    if len(image_points) != 4 or len(world_points_m) != 4:
        raise ValueError("Provide exactly 4 image_points and 4 world_points_m")
    img = np.array(image_points, dtype=np.float32)
    wrd = np.array(world_points_m, dtype=np.float32)
    H, _ = cv2.findHomography(img, wrd, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    return H

def parse_h_from_cfg(calib: Dict[str, Any]) -> Optional[np.ndarray]:
    H = calib.get("H")
    if H and len(H) == 9:
        return np.array(H, dtype=np.float64).reshape(3, 3)
    ip = calib.get("image_points")
    wp = calib.get("world_points_m")
    if ip and wp:
        return compute_h(ip, wp)
    return None

def image_to_world_m(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    pt = np.array([x, y, 1.0], dtype=np.float64)
    v = H @ pt
    if abs(v[2]) < 1e-9:
        return (0.0, 0.0)
    return (float(v[0]/v[2]), float(v[1]/v[2]))

def speed_kmh_from_tracks(prev_xy: Tuple[float,float], cur_xy: Tuple[float,float], fps: float) -> float:
    import math
    if fps <= 0: return 0.0
    dx = cur_xy[0] - prev_xy[0]
    dy = cur_xy[1] - prev_xy[1]
    dist_m = math.hypot(dx, dy)
    mps = dist_m * fps
    return mps * 3.6
