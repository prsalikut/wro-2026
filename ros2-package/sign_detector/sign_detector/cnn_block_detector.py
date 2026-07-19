"""CNN red/green block detector. Drop-in for BlockDetector -- returns the same
Detection objects, so sign_detector_node's fusion/smoothing is unchanged.

WHY THIS EXISTS
---------------
The user's trained CNN (block_detection_model.h5 / .tflite) is a 64x64 2-class
*classifier* (index 0 = green, index 1 = red). Measured behaviour:
  * green crop -> [0.94, 0.06]   red crop -> [0.00, 1.00]   (confident, correct)
  * BUT it has no "background" class: paper/desk crops read as a confident "red".
  * and it does NOT localize (no bounding box).

So a CNN alone can neither find the block nor reject the backdrop. The fix is to
split the job by each method's strength:

  LOCALIZE with colour  -- the ONE thing fixed thresholds do robustly is "is this
                           a saturated, block-shaped blob?" (blocks sit at LAB
                           chroma ~35-45; the grid paper/desk/walls are ~4-8, an
                           enormous gap). This finds the box AND rejects background.
  CLASSIFY with the CNN -- red-vs-green is the finicky part that fixed hue/a*
                           thresholds get wrong when the light shifts; the learned
                           model handles it far more robustly.

On top of that this adds, versus the raw .h5:
  * TFLite backend         (fast enough to run live on the Pi; .h5+TF is too heavy)
  * BGR->RGB               (the model trained on RGB; OpenCV frames are BGR -- a
                            silent channel swap would wreck accuracy)
  * test-time augmentation (average the box crop + a tightened crop -> steadier)
  * confidence gate        (low-margin frames fall back to the robust LAB a* axis
                            instead of emitting a coin-flip colour)
  * temporal label lock    (EMA the class probs per blob across frames -> kills the
                            red/green flicker the user complained about)
  * background safety net   (chroma + area + aspect + solidity gates stand in for
                            the class the model is missing)

If the model/backend can't load, the node falls back to the pure-colour
BlockDetector -- the robot never goes blind because of this file.
"""
import cv2
import numpy as np
from math import atan, tan, radians, degrees

from .block_detector import Detection


class _Classifier:
    """Thin wrapper over whatever inference backend is available.
    Accepts a .tflite (preferred: tflite_runtime, else tensorflow.lite) or a
    .h5 (keras/tensorflow). predict(batch[N,S,S,3] float32 in 0..1) -> [N,2]."""

    def __init__(self, weights, input_size):
        self.n = input_size
        self.kind, self.impl = self._load(weights)

    @staticmethod
    def _interpreter():
        """Return (InterpreterClass, backend_name), trying the light ARM wheel
        first (Pi), then LiteRT, then full TF -- TF>=2.16 dropped the old
        `from tensorflow.lite import Interpreter` spelling."""
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter, "tflite_runtime"
        except ImportError:
            pass
        try:
            from ai_edge_litert.interpreter import Interpreter
            return Interpreter, "ai_edge_litert"
        except ImportError:
            pass
        import tensorflow as tf
        return tf.lite.Interpreter, "tf.lite"

    def _load(self, w):
        wl = w.lower()
        if wl.endswith(".tflite"):
            Interpreter, src = self._interpreter()
            it = Interpreter(model_path=w)
            it.allocate_tensors()
            self.backend = src
            return "tflite", it
        try:
            import keras
        except ImportError:
            from tensorflow import keras
        self.backend = "keras"
        return "keras", keras.models.load_model(w, compile=False)

    def predict(self, batch):
        if self.kind == "keras":
            return np.asarray(self.impl.predict(batch, verbose=0))
        it = self.impl
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        res = []
        for i in range(batch.shape[0]):
            it.set_tensor(inp["index"], batch[i:i + 1].astype(inp["dtype"]))
            it.invoke()
            res.append(it.get_tensor(out["index"])[0])
        return np.asarray(res)


class CNNBlockDetector:
    L_MIN = 12
    ASPECT_LO, ASPECT_HI = 0.30, 3.5
    SOLIDITY_MIN = 0.55
    A_SPLIT = 132
    SMOOTH_ALPHA = 0.6
    SMOOTH_TTL = 5

    def __init__(self, weights, hfov_deg=60.0, sign_height_m=0.093,
                 min_area_frac=0.004, max_area_frac=0.5, chroma_min=28.0,
                 input_size=64, conf_min=0.60, tta=True, smooth=True,
                 class_order=("green", "red")):
        self.clf = _Classifier(weights, input_size)
        self.n = input_size
        self.hfov = hfov_deg
        self.sign_h = sign_height_m
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.chroma_min = chroma_min
        self.conf_min = conf_min
        self.tta = tta
        self.smooth = smooth
        self.order = class_order
        self.k_open = np.ones((5, 5), np.uint8)
        self.k_close = np.ones((11, 11), np.uint8)
        self._mem = []

    def _candidates(self, bgr):
        H, W = bgr.shape[:2]
        area_min = self.min_area_frac * H * W
        area_max = self.max_area_frac * H * W
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.int16)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)
        mask = ((chroma >= self.chroma_min) & (L >= self.L_MIN)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (area_min <= area <= area_max):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (self.ASPECT_LO < w / float(h) < self.ASPECT_HI):
                continue
            if area / float(w * h) < self.SOLIDITY_MIN:
                continue
            boxes.append((x, y, w, h, int(area)))
        return boxes, lab

    def _prep(self, bgr, x, y, w, h, inset):
        dx, dy = int(w * inset), int(h * inset)
        x0, y0 = max(0, x + dx), max(0, y + dy)
        x1, y1 = min(bgr.shape[1], x + w - dx), min(bgr.shape[0], y + h - dy)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            crop = bgr[y:y + h, x:x + w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.n, self.n), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.float32) / 255.0

    def _classify(self, bgr, box):
        x, y, w, h, _ = box
        insets = (0.0, 0.12) if self.tta else (0.0,)
        batch = np.stack([self._prep(bgr, x, y, w, h, s) for s in insets])
        return self.clf.predict(batch).mean(axis=0)

    def _lock(self, cx, cy, probs, W):
        if not self.smooth:
            return probs
        tol = 0.08 * W
        best, bd = None, tol
        for m in self._mem:
            d = abs(m["cx"] - cx) + abs(m["cy"] - cy)
            if d < bd:
                best, bd = m, d
        if best is None:
            best = {"cx": cx, "cy": cy, "p": probs.copy(), "ttl": self.SMOOTH_TTL}
            self._mem.append(best)
        else:
            a = self.SMOOTH_ALPHA
            best["p"] = a * probs + (1 - a) * best["p"]
            best["cx"], best["cy"], best["ttl"] = cx, cy, self.SMOOTH_TTL
        return best["p"]

    def _age(self):
        for m in self._mem:
            m["ttl"] -= 1
        self._mem = [m for m in self._mem if m["ttl"] > 0]

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        fx = (W / 2.0) / tan(radians(self.hfov) / 2.0)
        boxes, lab = self._candidates(bgr)
        out = []
        for box in boxes:
            x, y, w, h, area = box
            cx, cy = x + w // 2, y + h // 2
            probs = self._lock(cx, cy, self._classify(bgr, box), W)
            gi = self.order.index("green")
            ri = self.order.index("red")
            conf = float(max(probs))
            if conf >= self.conf_min:
                color = self.order[int(np.argmax(probs))]
            else:
                core = lab[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4, 1]
                a_mean = float(core.mean()) if core.size else 128.0
                color = "red" if a_mean >= self.A_SPLIT else "green"
            bearing = degrees(atan((cx - W / 2.0) / fx))
            dist = (self.sign_h * fx) / h
            out.append(Detection(color, cx, cy, w, h, area,
                                 round(bearing, 1), round(dist, 2)))
        self._age()
        out.sort(key=lambda d: d.area, reverse=True)
        return out
