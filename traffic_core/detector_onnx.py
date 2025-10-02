from typing import List, Dict, Any, Optional, Union
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception as e:
    raise ImportError("Install onnxruntime or onnxruntime-gpu") from e

class YOLOv8ONNX:
    """Minimal ONNXRuntime inference for YOLOv8 export (ultralytics layout)."""
    def __init__(self, onnx_path: str, providers: Optional[list] = None, conf_thresh=0.45, iou_thresh=0.45, imgsz=640, id2name: Optional[dict]=None, keep_ids: Optional[list]=None):
        self.session = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        self.conf = conf_thresh; self.iou = iou_thresh; self.imgsz = imgsz
        self.id2name = id2name or {}
        self.keep_ids = keep_ids

        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    def _pre(self, img_bgr):
        h0,w0 = img_bgr.shape[:2]
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None]  # NCHW
        return img, (w0, h0)

    def _post(self, pred, orig_size):
        # Expect (1, N, 84): [cx,cy,w,h,conf,cls...]
        boxes = []
        p = pred[0]  # (N,84)
        w0,h0 = orig_size
        for row in p:
            conf = row[4]
            if conf < self.conf: continue
            cls_id = int(np.argmax(row[5:]))
            if self.keep_ids is not None and cls_id not in self.keep_ids: continue
            cx,cy,w,h = row[:4]
            # scale back to original size
            cx *= w0/self.imgsz; cy *= h0/self.imgsz
            w  *= w0/self.imgsz; h  *= h0/self.imgsz
            x = int(cx - w/2); y = int(cy - h/2)
            label = self.id2name.get(cls_id, str(cls_id))
            boxes.append({"bbox": (x,y,int(w),int(h)), "conf": float(conf), "class_id": cls_id, "label": label})
        # naive NMS (optional): rely on conf threshold for now, or implement cv2.dnn.NMSBoxes
        return boxes

    def infer(self, frame_bgr):
        blob, orig = self._pre(frame_bgr)
        out = self.session.run([self.out_name], {self.inp_name: blob})[0]
        return self._post(out, orig)
