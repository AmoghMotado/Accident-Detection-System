from typing import Sequence, Tuple, Iterable
import numpy as np
import cv2

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]  # x, y, w, h


def polygon_to_np(polygon: Iterable[Point]) -> np.ndarray:
    """Convert a list of (x,y) to an np.int32 Nx1x2 polygon suitable for cv2."""
    poly = np.array(list(polygon), dtype=np.int32)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be an iterable of (x,y) points")
    return poly.reshape((-1, 1, 2))


def inside_roi(box_xywh: Box, polygon: Iterable[Point]) -> bool:
    """Return True if the bbox center lies inside the polygon."""
    x, y, w, h = box_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    poly = np.array(list(polygon), dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def draw_roi(frame_bgr: np.ndarray, polygon: Iterable[Point], color=(0, 0, 255), thickness: int = 2) -> None:
    """Draw the ROI polygon on the frame (in-place)."""
    poly = np.array(list(polygon), dtype=np.int32)
    cv2.polylines(frame_bgr, [poly], isClosed=True, color=color, thickness=thickness)


def polygon_area(polygon: Iterable[Point]) -> float:
    """Polygon area using the shoelace formula (pixels^2)."""
    poly = np.array(list(polygon), dtype=np.float64)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def bbox_iou(a: Box, b: Box) -> float:
    """IoU for xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 0 else 0.0
