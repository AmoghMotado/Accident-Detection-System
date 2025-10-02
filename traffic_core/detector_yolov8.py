from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    # Ultralytics (yolov8)
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "Ultralytics is required. Install with `pip install ultralytics`.\n"
        f"Original error: {e}"
    )


def _normalize_target_classes(
    target_classes: Optional[List[Union[str, int]]],
    id2name: Dict[int, str]
) -> Optional[List[int]]:
    """
    Accepts a list of class names or indices and returns a list of indices.
    If None is passed, returns None (= keep all classes).
    """
    if target_classes is None:
        return None
    name2id = {v: k for k, v in id2name.items()}
    norm: List[int] = []
    for c in target_classes:
        if isinstance(c, int):
            if c in id2name:
                norm.append(c)
        elif isinstance(c, str):
            if c in name2id:
                norm.append(name2id[c])
        else:
            continue
    # dedupe while preserving order
    seen = set()
    out = []
    for i in norm:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


class Yolov8Detector:
    """
    Thin wrapper around Ultralytics YOLOv8 to produce consistent dictionaries:

    Output per detection:
    {
        "bbox": (x, y, w, h),          # integers, xywh in pixels
        "conf": float,                 # confidence score
        "class_id": int,               # numeric class id
        "label": str                   # class name
    }
    """

    def __init__(
        self,
        weights: str,
        device: Union[str, int] = 0,
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        target_classes: Optional[List[Union[str, int]]] = None,
    ):
        """
        Args:
            weights: Path to YOLOv8 weights (.pt).
            device: GPU index (e.g., 0) or "cpu".
            conf_thresh: confidence threshold.
            iou_thresh: NMS IoU threshold.
            imgsz: inference image size (short side).
            target_classes: list of class names or ids to keep (None = keep all).
        """
        self.model = YOLO(weights)
        self.model.to(device)
        # Ultralytics class map (id -> name)
        # In most models, self.model.names is a dict {id: name}
        self.id2name: Dict[int, str] = dict(self.model.names)
        self.imgsz = imgsz
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.device = device

        self.keep_ids = _normalize_target_classes(target_classes, self.id2name)

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run a forward pass on a single BGR image (H,W,3).

        Returns:
            List of detection dicts (see class docstring).
        """
        # Ultralytics expects RGB by default; it will convert internally if needed,
        # but passing BGR is fine since the library handles cv2 images too.
        # We’ll rely on the high-level API which accepts numpy arrays directly.
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        dets: List[Dict[str, Any]] = []
        if not results:
            return dets

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        # xyxy boxes: (x1, y1, x2, y2)
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
            if self.keep_ids is not None and cid not in self.keep_ids:
                continue
            w = max(0, int(round(x2 - x1)))
            h = max(0, int(round(y2 - y1)))
            x = int(round(x1))
            y = int(round(y1))
            label = self.id2name.get(int(cid), str(cid))
            dets.append({
                "bbox": (x, y, w, h),
                "conf": float(conf),
                "class_id": int(cid),
                "label": label
            })

        return dets
