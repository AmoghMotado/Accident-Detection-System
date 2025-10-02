"""
traffic_core: Core modules for the Accident Detection System.

Submodules:
- detector_yolov8: Ultralytics YOLOv8 detector wrapper.
- roi: ROI / polygon utilities.
- tracker_sort: Lightweight centroid/SORT-style tracker.
- accident_logic: Accident/incident heuristics engine.
- vis: Drawing utilities for overlays.

This package is intentionally lightweight and pure-Python so it runs
anywhere you can install the requirements.
"""

__all__ = [
    "detector_yolov8",
    "roi",
    "tracker_sort",
    "accident_logic",
    "vis",
]
