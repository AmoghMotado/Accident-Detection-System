from __future__ import annotations
from typing import List, Dict

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
    raise ImportError("Install deep-sort-realtime: pip install deep-sort-realtime lapx scikit-learn") from e


class DeepSortTracker:
    """
    Thin wrapper over deep-sort-realtime.
    Input detections: [{bbox:(x,y,w,h), conf:float, class_id:int, label:str}, ...]
    Output tracks:    [{bbox, conf, class_id, label, track_id, speed_px_per_sec}, ...]
    """
    def __init__(self, max_age=30, n_init=3, max_iou_distance=0.7, nn_budget=100, embedder="mobilenet"):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            nn_budget=nn_budget,
            embedder=embedder,
        )
        self.prev_centers = {}  # tid -> (cx, cy)

    def update(self, frame_bgr, detections: List[Dict], fps: float):
        # Convert to DeepSort input format: [ [x1,y1,x2,y2,conf,cls], ...]
        ds_dets = []
        for d in detections:
            x, y, w, h = d["bbox"]
            ds_dets.append(([x, y, x + w, y + h], float(d.get("conf", 0.0)), int(d.get("class_id", 0))))

        tracks = self.tracker.update_tracks(ds_dets, frame=frame_bgr)

        out: List[Dict] = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            # bbox
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            # ----- robust class / confidence extraction across lib versions -----
            # Some versions expose det_class / det_confidence; others have get_*; some expose .class_name
            cls_attr = getattr(t, "det_class", None)
            if cls_attr is None and hasattr(t, "get_det_class"):
                try:
                    cls_attr = t.get_det_class()
                except Exception:
                    cls_attr = None

            conf_attr = getattr(t, "det_confidence", None)
            if conf_attr is None and hasattr(t, "get_det_confidence"):
                try:
                    conf_attr = t.get_det_confidence()
                except Exception:
                    conf_attr = None
            if conf_attr is None:
                # other fallbacks used by some forks
                conf_attr = getattr(t, "confidence", 0.0)

            # normalize class id and label
            try:
                cls_id = int(cls_attr) if cls_attr is not None else 0
            except Exception:
                cls_id = 0

            # If the tracker doesn't carry a semantic label, map common COCO ids; else fallback to generic
            id2name = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
            if isinstance(cls_attr, str) and cls_attr:
                label = cls_attr
            else:
                label = id2name.get(cls_id, "vehicle")

            conf = float(conf_attr or 0.0)

            # speed in pixels/sec using previous center
            pxps = 0.0
            if t.track_id in self.prev_centers:
                pcx, pcy = self.prev_centers[t.track_id]
                dx = cx - pcx
                dy = cy - pcy
                pxps = (dx * dx + dy * dy) ** 0.5 * fps
            self.prev_centers[t.track_id] = (cx, cy)

            out.append({
                "bbox": (x1, y1, w, h),
                "label": label,
                "class_id": cls_id,
                "conf": conf,
                "track_id": int(t.track_id),
                "speed_px_per_sec": float(pxps),
            })

        return out
