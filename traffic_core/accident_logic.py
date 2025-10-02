from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict, deque
import math, time

from .roi import inside_roi

Box = Tuple[int, int, int, int]

def _center(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)

def _iou_xywh(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = aw * ah; area_b = bw * bh
    return inter / (area_a + area_b - inter + 1e-6)

class AccidentEngine:
    """
    Heuristic accident / hazard detector built on tracked objects.
    Emits events like: stopped_vehicle, sudden_deceleration, near_miss,
    and the new decisive 'collision_confirmed'.
    """
    def __init__(
        self,
        polygon: Iterable[Tuple[int, int]],
        allowed_labels: Iterable[str],
        *,
        min_speed_kmh: float = 3.0,
        stopped_frames: int = 45,
        sudden_decel_kmh: float = 25.0,
        collision_iou: float = 0.25,
        near_miss_dist_px: float = 40.0,
        fps: float = 30.0,
        # NEW:
        rel_speed_kmh: float = 15.0,
        iou_spike: float = 0.08,
        window: int = 6,
    ):
        self.poly = list(polygon)
        self.allowed = set(allowed_labels)
        self.min_speed = float(min_speed_kmh)
        self.stop_frames = int(stopped_frames)
        self.decel_thr = float(sudden_decel_kmh)
        self.iou_thr = float(collision_iou)
        self.near_thr = float(near_miss_dist_px)
        self.fps = float(fps)

        # new decisive crash tuning
        self.rel_speed_kmh = float(rel_speed_kmh)
        self.iou_spike = float(iou_spike)
        self.window = int(window)

        # histories
        self.speed_hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max(2, int(2*self.fps))))
        self.bbox_hist: Dict[int, deque]  = defaultdict(lambda: deque(maxlen=max(2, self.window)))

        # debounce
        self.last_stop_emit: Dict[int, float] = {}
        self.last_decel_emit: Dict[int, float] = {}
        self.pair_emit_ts: Dict[Tuple[int, int, str], float] = {}
        self.debounce_sec = 2.0

    @staticmethod
    def _pxps_to_kmh(px_per_sec: float, pixels_per_meter: float) -> float:
        if pixels_per_meter <= 0:
            return 0.0
        mps = (px_per_sec / pixels_per_meter)
        return mps * 3.6

    def step(self, tracks: List[Dict], pixels_per_meter: float) -> List[Dict]:
        now = time.time()
        events: List[Dict] = []

        # update histories for visible allowed tracks
        for d in tracks:
            if d["label"] not in self.allowed:
                continue
            if not inside_roi(d["bbox"], self.poly):
                continue

            tid = int(d.get("track_id", -1))
            if tid < 0:
                continue

            # speed in km/h
            kmh = self._pxps_to_kmh(d.get("speed_px_per_sec", 0.0), pixels_per_meter)
            d["speed_kmh"] = kmh
            self.speed_hist[tid].append((now, kmh))
            self.bbox_hist[tid].append(tuple(d["bbox"]))

        # stopped vehicle
        for tid, q in self.speed_hist.items():
            if len(q) >= self.stop_frames:
                avg = sum(v for _, v in list(q)[-self.stop_frames:]) / self.stop_frames
                if avg < self.min_speed and now - self.last_stop_emit.get(tid, 0.0) >= self.debounce_sec:
                    events.append({
                        "type": "stopped_vehicle",
                        "severity": "medium",
                        "time": now,
                        "track_id": tid,
                        "avg_speed_kmh": round(avg, 1),
                    })
                    self.last_stop_emit[tid] = now

        # sudden deceleration (~1s window)
        win = int(self.fps) if self.fps > 0 else 30
        for tid, q in self.speed_hist.items():
            if len(q) > win:
                spd_now = q[-1][1]
                spd_prev = q[-win][1]
                drop = spd_prev - spd_now
                if drop >= self.decel_thr and spd_now < self.min_speed * 2 \
                   and now - self.last_decel_emit.get(tid, 0.0) >= self.debounce_sec:
                    events.append({
                        "type": "sudden_deceleration",
                        "severity": "high",
                        "time": now,
                        "track_id": tid,
                        "delta_kmh": round(drop, 1),
                        "from_kmh": round(spd_prev, 1),
                        "to_kmh": round(spd_now, 1),
                    })
                    self.last_decel_emit[tid] = now

        # pairwise: collision / near-miss / collision_confirmed
        n = len(tracks)
        for i in range(n):
            a = tracks[i]
            if a["label"] not in self.allowed or not inside_roi(a["bbox"], self.poly):
                continue
            tid_a = int(a.get("track_id", -1))
            if tid_a < 0:
                continue
            for j in range(i + 1, n):
                b = tracks[j]
                if b["label"] not in self.allowed or not inside_roi(b["bbox"], self.poly):
                    continue
                tid_b = int(b.get("track_id", -1))
                if tid_b < 0:
                    continue

                iou_now = _iou_xywh(a["bbox"], b["bbox"])
                # relative speed
                rel_speed = abs(a.get("speed_kmh", 0.0) - b.get("speed_kmh", 0.0))

                # IOU spike vs previous frame
                prev_iou = 0.0
                if len(self.bbox_hist[tid_a]) >= 2 and len(self.bbox_hist[tid_b]) >= 2:
                    prev_a = self.bbox_hist[tid_a][-2]
                    prev_b = self.bbox_hist[tid_b][-2]
                    prev_iou = _iou_xywh(prev_a, prev_b)
                diou = iou_now - prev_iou

                # decisive collision
                if iou_now >= self.iou_thr and diou >= self.iou_spike and rel_speed >= self.rel_speed_kmh:
                    key = (min(tid_a, tid_b), max(tid_a, tid_b), "collision_confirmed")
                    if now - self.pair_emit_ts.get(key, 0.0) >= self.debounce_sec:
                        events.append({
                            "type": "collision_confirmed",
                            "severity": "critical",
                            "time": now,
                            "a": tid_a,
                            "b": tid_b,
                            "extra": {
                                "iou": round(iou_now, 3),
                                "diou": round(diou, 3),
                                "rel_speed_kmh": round(rel_speed, 1),
                            }
                        })
                        self.pair_emit_ts[key] = now
                    continue

                # legacy overlap (kept for compatibility)
                if iou_now >= self.iou_thr:
                    key = (min(tid_a, tid_b), max(tid_a, tid_b), "collision_overlap")
                    if now - self.pair_emit_ts.get(key, 0.0) >= self.debounce_sec:
                        events.append({
                            "type": "collision_overlap",
                            "severity": "critical",
                            "time": now,
                            "a": tid_a,
                            "b": tid_b,
                            "iou": round(iou_now, 3)
                        })
                        self.pair_emit_ts[key] = now
                    continue

                # near-miss: close centers + noticeable relative speed
                ax, ay = _center(a["bbox"]); bx, by = _center(b["bbox"])
                dist = math.hypot(ax - bx, ay - by)
                if dist < self.near_thr and rel_speed >= self.decel_thr / 2.0:
                    key = (min(tid_a, tid_b), max(tid_a, tid_b), "near_miss")
                    if now - self.pair_emit_ts.get(key, 0.0) >= self.debounce_sec:
                        events.append({
                            "type": "near_miss",
                            "severity": "high",
                            "time": now,
                            "a": tid_a,
                            "b": tid_b,
                            "dist_px": int(dist),
                            "rel_kmh": round(rel_speed, 1)
                        })
                        self.pair_emit_ts[key] = now

        return events
