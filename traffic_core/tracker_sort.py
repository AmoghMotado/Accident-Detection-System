from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import itertools
import numpy as np
from filterpy.kalman import KalmanFilter

Box = Tuple[int, int, int, int]  # x, y, w, h


def _center(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


class _KF2D:
    """
    Simple constant-velocity Kalman filter on (cx, cy).
    State: [cx, cy, vx, vy]
    Meas:  [cx, cy]
    """
    def __init__(self, cx: float, cy: float, dt: float = 1.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1,  0],
                              [0, 0, 0,  1]], dtype=float)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=float)
        self.kf.P *= 50.0            # state covariance
        self.kf.R *= 2.0             # measurement noise
        self.kf.Q = np.eye(4) * 0.01 # process noise
        self.kf.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)

    def predict(self) -> Tuple[float, float]:
        self.kf.predict()
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

    def update(self, cx: float, cy: float) -> None:
        self.kf.update(np.array([[cx], [cy]], dtype=float))

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.kf.x[2, 0]), float(self.kf.x[3, 0])


class Track:
    def __init__(self, track_id: int, det: Dict, fps: float):
        self.id = track_id
        self.label: str = det["label"]
        self.class_id: int = det["class_id"]
        self.conf: float = det.get("conf", 0.0)
        self.bbox: Box = det["bbox"]
        cx, cy = _center(self.bbox)
        self.fps = fps
        self.kf = _KF2D(cx, cy, dt=1.0 / max(fps, 1.0))
        self.age = 0          # total frames since born
        self.missed = 0       # consecutive frames without match
        self.history: List[Tuple[float, float]] = [(cx, cy)]

    def predict(self):
        self.age += 1
        cx, cy = self.kf.predict()
        # keep previous box size, move by predicted center
        x, y, w, h = self.bbox
        nx = int(round(cx - w / 2.0))
        ny = int(round(cy - h / 2.0))
        self.bbox = (nx, ny, w, h)
        return cx, cy

    def update(self, det: Dict):
        self.label = det["label"]
        self.class_id = det["class_id"]
        self.conf = det.get("conf", self.conf)
        self.bbox = det["bbox"]
        cx, cy = _center(self.bbox)
        self.kf.update(cx, cy)
        self.missed = 0
        self.history.append((cx, cy))

    def mark_missed(self):
        self.missed += 1

    def speed_pixels_per_sec(self) -> float:
        """Approx pixel speed using last two centers."""
        if len(self.history) < 2 or self.fps <= 0:
            return 0.0
        (x2, y2) = self.history[-1]
        (x1, y1) = self.history[-2]
        dist_px = math.hypot(x2 - x1, y2 - y1)
        return dist_px * self.fps  # pixels/second


class SortTracker:
    """
    Lightweight SORT-style tracker:
      - One Kalman filter per track (cx,cy,vx,vy)
      - Greedy data association by centroid distance (pixels)
      - Track birth/death via max_age and distance threshold

    NOTE: This avoids SciPy/Hungarian to keep dependencies light.
    """
    def __init__(self, max_age: int = 30, match_thresh: float = 80.0, fps: float = 30.0):
        self.max_age = max_age
        self.match_thresh = match_thresh
        self.fps = fps
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _greedy_match(self, detections: List[Dict]) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Return list of (det_index, track_id) matches. Greedy by nearest center.
        Unmatched dets -> (i, None); unmatched tracks -> (None, tid)
        """
        det_centers = [ _center(d["bbox"]) for d in detections ]
        trk_ids = list(self.tracks.keys())
        trk_centers = [ _center(self.tracks[tid].bbox) for tid in trk_ids ]

        # Build distance matrix
        D = []
        for dc in det_centers:
            row = [ math.hypot(dc[0]-tc[0], dc[1]-tc[1]) for tc in trk_centers ]
            D.append(row)

        matched_det = set()
        matched_trk = set()
        pairs: List[Tuple[Optional[int], Optional[int]]] = []

        # Greedy: repeatedly pick smallest distance under threshold
        while True:
            best = None
            best_val = float("inf")
            for i in range(len(D)):
                if i in matched_det: continue
                for j in range(len(D[i])):
                    if j in matched_trk: continue
                    if D[i][j] < best_val:
                        best_val = D[i][j]; best = (i, j)
            if best is None or best_val > self.match_thresh:
                break
            i, j = best
            matched_det.add(i); matched_trk.add(j)
            pairs.append((i, trk_ids[j]))

        # Unmatched detections
        for i in range(len(detections)):
            if i not in matched_det:
                pairs.append((i, None))
        # Unmatched tracks
        for j, tid in enumerate(trk_ids):
            if j not in matched_trk:
                pairs.append((None, tid))

        return pairs

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with current-frame detections.
        Each detection must have keys: bbox, label, class_id, (optional) conf.
        Returns list of track dicts: {bbox,label,class_id,conf,track_id,speed_kmh}
        """
        # 1) Predict all tracks forward
        for trk in self.tracks.values():
            trk.predict()

        # 2) Associate detections to tracks
        matches = self._greedy_match(detections)

        # 3) Update matched tracks, create new for unmatched detections,
        #    age and remove stale tracks
        # Matched
        for di, tid in matches:
            if di is not None and tid is not None:
                self.tracks[tid].update(detections[di])

        # New tracks
        for di, tid in matches:
            if di is not None and tid is None:
                det = detections[di]
                t = Track(self._next_id, det, fps=self.fps)
                self.tracks[self._next_id] = t
                self._next_id += 1

        # Mark missed & cleanup
        dead = []
        for tid, trk in self.tracks.items():
            # If not updated this cycle, mark missed
            if len(trk.history) == 0 or trk.age == 0 or trk.history[-1] != _center(trk.bbox):
                # We can't directly tell if updated; use 'missed' via pairing:
                # A track that did not appear in any match with det_index is unmatched.
                pass
        # More robust: recompute which tids were matched
        matched_tids = { tid for di, tid in matches if di is not None and tid is not None }
        for tid, trk in self.tracks.items():
            if tid not in matched_tids:
                trk.mark_missed()
            if trk.missed > self.max_age:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

        # 4) Prepare outputs with speeds (km/h requires calibration elsewhere)
        out: List[Dict] = []
        for tid, trk in self.tracks.items():
            px_per_sec = trk.speed_pixels_per_sec()
            out.append({
                "bbox": trk.bbox,
                "label": trk.label,
                "class_id": trk.class_id,
                "conf": trk.conf,
                "track_id": tid,
                "speed_px_per_sec": px_per_sec,  # convert to km/h in the main loop using calibration
            })
        return out
