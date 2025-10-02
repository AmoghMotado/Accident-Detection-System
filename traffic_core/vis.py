from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
import cv2
import numpy as np

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]

PALETTE = {
    "box": (0, 255, 0),      # green
    "text": (255, 255, 255), # white
    "event": (0, 0, 255),    # red
    "roi": (0, 0, 255),      # red
}

def draw_roi(frame_bgr: np.ndarray, polygon: Iterable[Point], color=PALETTE["roi"], thickness: int = 2):
    poly = np.array(list(polygon), dtype=np.int32)
    cv2.polylines(frame_bgr, [poly], isClosed=True, color=color, thickness=thickness)

def put_label(frame_bgr, text, org, scale=0.5, color=PALETTE["text"], thickness=1, bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(frame_bgr, (x-2, y-h-2), (x+w+2, y+2), (0,0,0), -1)
    cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_track(frame_bgr: np.ndarray, track: Dict, show_speed=True):
    x, y, w, h = map(int, track["bbox"])
    tid = track.get("track_id", -1)
    label = track.get("label", "obj")
    kmh = track.get("speed_kmh", None)

    # exact detector box (no padding)
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), PALETTE["box"], 2)

    if show_speed and kmh is not None:
        txt = f"ID{tid} {label} {kmh:.1f} km/h"
    else:
        txt = f"ID{tid} {label}"
    put_label(frame_bgr, txt, (x, max(15, y - 6)))

def draw_events_toast(frame_bgr: np.ndarray, events: List[Dict], max_lines: int = 5, y0: int = 24):
    y = y0
    for e in events[:max_lines]:
        txt = e["type"] + (f" [{e.get('severity')}]" if e.get("severity") else "")
        put_label(frame_bgr, txt, (10, y), scale=0.6, color=PALETTE["event"], thickness=2)
        y += 22

def draw_fps(frame_bgr: np.ndarray, fps: float):
    put_label(frame_bgr, f"FPS: {fps:.1f}", (10, frame_bgr.shape[0]-10), scale=0.6)
