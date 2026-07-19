"""YOLO red/green detector. Drop-in for BlockDetector — returns the same Detection objects.
Load a trained model exported for the Pi (NCNN dir, .tflite, or .pt)."""
from math import atan, tan, radians, degrees
from .block_detector import Detection


class MLBlockDetector:
    def __init__(self, weights, hfov_deg=60.0, conf=0.4, imgsz=320):
        from ultralytics import YOLO
        self.model = YOLO(weights, task="detect")
        self.hfov = hfov_deg
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        fx = (W / 2.0) / tan(radians(self.hfov) / 2.0)
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bearing = degrees(atan((cx - W / 2.0) / fx))
            name = self.model.names[int(b.cls)]
            out.append(Detection(name, int(cx), int(cy), int(w), int(h),
                                 int(w * h), round(bearing, 1), 0.0))
        out.sort(key=lambda d: d.area, reverse=True)
        return out
