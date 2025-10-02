#!/usr/bin/env python3
import os, csv, time, argparse, sys
from typing import Dict, Any, List
import cv2, yaml, numpy as np

from traffic_core.detector_yolov8 import Yolov8Detector
from traffic_core.detector_onnx import YOLOv8ONNX
from traffic_core.tracker_sort import SortTracker
from traffic_core.tracker_deepsort import DeepSortTracker
from traffic_core.accident_logic import AccidentEngine
from traffic_core.roi import draw_roi, inside_roi
from traffic_core.vis import draw_track, draw_events_toast, draw_fps
from traffic_core.homography import parse_h_from_cfg, image_to_world_m, speed_kmh_from_tracks

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_cfg(p): return yaml.safe_load(open(p))

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    # --- detector selection ---
    det_cfg = cfg["model"]
    detector_type = det_cfg.get("backend", "ultralytics")  # "ultralytics" | "onnx"
    if detector_type == "onnx":
        # optional id2name map (from Ultralytics if you know it)
        id2name = {2:"car",3:"motorbike",5:"bus",7:"truck"}
        keep = det_cfg.get("target_classes", None)
        keep_ids = None
        if keep:
            name2id = {v:k for k,v in id2name.items()}
            keep_ids = [ (c if isinstance(c,int) else name2id.get(c,c)) for c in keep ]
        detector = YOLOv8ONNX(
            onnx_path=det_cfg["weights"].replace(".pt",".onnx"),
            providers=None,
            conf_thresh=det_cfg.get("conf_thresh",0.45),
            iou_thresh=det_cfg.get("iou_thresh",0.45),
            imgsz=det_cfg.get("imgsz",640),
            id2name=id2name,
            keep_ids=keep_ids
        )
    else:
        detector = Yolov8Detector(
            weights=det_cfg["weights"],
            device=det_cfg.get("device",0),
            conf_thresh=det_cfg.get("conf_thresh",0.45),
            iou_thresh=det_cfg.get("iou_thresh",0.45),
            imgsz=det_cfg.get("imgsz",640),
            target_classes=det_cfg.get("target_classes",None)
        )

    # --- input ---
    inp = cfg["input"]
    src = inp["source"]
    resize = tuple(inp["resize"]) if inp.get("resize") else None
    stride = int(inp.get("stride",1))
    fps_fallback = float(inp.get("fallback_fps",30))
    cap = cv2.VideoCapture(src); assert cap.isOpened(), f"open fail: {src}"
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if resize: out_w, out_h = resize

    # --- roi ---
    poly = cfg["roi"]["polygon"]
    allowed = det_cfg.get("target_classes", ["car","bus","truck","motorbike"])

    # --- tracker selection ---
    trk_mode = cfg.get("tracking",{}).get("mode","deepsort")  # "deepsort" | "sort"
    if trk_mode == "deepsort":
        tracker = DeepSortTracker()
        use_frame_for_tracker = True
    else:
        tracker = SortTracker(fps=fps, max_age=int(cfg.get("tracking",{}).get("max_age",30)),
                              match_thresh=float(cfg.get("tracking",{}).get("match_thresh",80)))
        use_frame_for_tracker = False

    # --- homography / calibration ---
    calib = cfg["calibration"]
    H = parse_h_from_cfg(calib)
    ppm = float(calib.get("pixels_per_meter", 10.0))

    # --- accident engine ---
    acc = cfg["accident"]
    engine = AccidentEngine(
        polygon=poly, allowed_labels=allowed,
        min_speed_kmh=float(acc.get("min_speed_kmh",3)),
        stopped_frames=int(acc.get("stopped_frames",45)),
        sudden_decel_kmh=float(acc.get("sudden_decel_kmh",25)),
        collision_iou=float(acc.get("collision_iou",0.25)),
        near_miss_dist_px=float(acc.get("near_miss_dist_px",40)),
        fps=fps
    )

    # --- output ---
    out_cfg = cfg["output"]; draw = bool(out_cfg.get("draw",True))
    save_video = bool(out_cfg.get("save_video",False))
    out_path = out_cfg.get("out_path","./outputs/accidents.mp4")
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w,out_h))
    else:
        writer = None
    csv_path = out_cfg.get("save_events_csv","./outputs/events.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_f = open(csv_path,"w",newline=""); csv_w = csv.DictWriter(csv_f, fieldnames=["time","type","severity","track_id","a","b","extra"]); csv_w.writeheader()

    frame_id = 0; t_last=time.time(); sm_fps=fps
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if resize: frame = cv2.resize(frame,(out_w,out_h))
            if stride>1 and (frame_id%stride!=0): frame_id+=1; continue

            dets = detector.infer(frame)
            dets = [d for d in dets if d["label"] in allowed and inside_roi(d["bbox"], poly)]

            if trk_mode == "deepsort":
                tracks = tracker.update(frame, dets, fps=fps)
            else:
                tracks = tracker.update(dets)

            # fill km/h using homography if available
            if H is not None:
                # derive per-track speed from centers mapped by H
                for t in tracks:
                    x,y,w,h = t["bbox"]; cx = x+w/2; cy = y+h/2
                    tid = t["track_id"]
                    # DeepSort wrapper tracks prev centers internally for pxps,
                    # but we recompute using homography for accuracy
                    # (store tiny state in t)
                    prev = t.get("_prev_world")
                    wx, wy = image_to_world_m(H, cx, cy)
                    if prev:
                        kmh = speed_kmh_from_tracks(prev, (wx,wy), fps)
                    else:
                        kmh = 0.0
                    t["speed_kmh"] = kmh
                    t["_prev_world"] = (wx,wy)
            else:
                # fallback scalar
                for t in tracks:
                    pxps = t.get("speed_px_per_sec",0.0)
                    mps = pxps / max(ppm,1e-6)
                    t["speed_kmh"] = mps*3.6

            events = engine.step(tracks, pixels_per_meter=ppm)

            if draw:
                from traffic_core.vis import draw_roi as draw_roi_vis
                draw_roi_vis(frame, poly)
                for t in tracks: 
                    from traffic_core.vis import draw_track
                    draw_track(frame, t, show_speed=True)
                from traffic_core.vis import draw_events_toast, draw_fps
                draw_events_toast(frame, events)
                now=time.time(); inst=1.0/max(1e-6, now-t_last); sm_fps=0.9*sm_fps+0.1*inst; draw_fps(frame, sm_fps); t_last=now

            if events:
                for e in events:
                    csv_w.writerow({
                        "time": int(e.get("time", time.time())),
                        "type": e.get("type",""),
                        "severity": e.get("severity",""),
                        "track_id": e.get("track_id",""),
                        "a": e.get("a",""),
                        "b": e.get("b",""),
                        "extra": {k:v for k,v in e.items() if k not in ("time","type","severity","track_id","a","b")}
                    })

            if writer: writer.write(frame)
            if draw:
                cv2.imshow("Accident Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            frame_id += 1
    finally:
        cap.release()
        if writer: writer.release()
        csv_f.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--config","-c",default="configs/infer.yaml")
    a=p.parse_args()
    main(a.config)
