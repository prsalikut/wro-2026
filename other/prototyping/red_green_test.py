"""
Simple red/green block detector for WRO Future Engineers.
Run live on a webcam:   python red_green_test.py
Run on one image:       python red_green_test.py photo.jpg
Run on a folder:        python red_green_test.py "training_images/etc-training images"
Needs only: pip install opencv-python numpy      (press q to quit the live window)

Two calibration PROFILES:
  "practice" - the warm-lit hand-made cardboard blocks in your training set.
               Measured from your 142 photos: red shows up orange-ish (hue ~4-16)
               and green is yellow-green (hue ~32-40) because of the warm indoor
               lighting. The KEY trick for red is a HIGH saturation floor (>=120):
               it separates the red block (S~180) from the floor/wood (same hue,
               but low saturation).
  "official" - the real WRO pillars: red RGB(238,39,55), green RGB(68,214,44).
               Use this once you calibrate against the real board at the venue.
"""
import cv2, numpy as np, sys, os, glob

PROFILES = {
    "practice": dict(
        RED  =[((0, 120, 50), (16, 255, 255)), ((166, 120, 50), (179, 255, 255))],
        GREEN=[((28, 70, 45), (46, 255, 255))],
    ),
    "official": dict(
        RED  =[((0, 100, 70), (10, 255, 255)), ((168, 100, 70), (179, 255, 255))],
        GREEN=[((40, 80, 70), (85, 255, 255))],
    ),
}
PROFILE = "practice"

MIN_AREA_FRAC = 0.004
KOPEN  = np.ones((5, 5), np.uint8)
KCLOSE = np.ones((11, 11), np.uint8)


def _find(mask, area_min):
    """Clean the mask and keep solid, block-ish blobs."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KOPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KCLOSE)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.3 < w / float(h) < 3.5):
            continue
        if a / float(w * h) < 0.55:
            continue
        out.append((x, y, w, h))
    return out


def detect(frame):
    p = PROFILES[PROFILE]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    area_min = MIN_AREA_FRAC * frame.shape[0] * frame.shape[1]

    def mask(ranges):
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        return m

    return {"RED": _find(mask(p["RED"]), area_min),
            "GREEN": _find(mask(p["GREEN"]), area_min)}


def annotate(frame, dets):
    for label, boxes in dets.items():
        col = (0, 0, 255) if label == "RED" else (0, 200, 0)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
            cv2.putText(frame, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return frame


def gui_ok():
    try:
        cv2.namedWindow("_t"); cv2.destroyWindow("_t"); return True
    except cv2.error:
        return False


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    HAVE_GUI = gui_ok()
    print(f"profile = {PROFILE}")

    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect(frame)
            print(f"RED: {len(dets['RED'])}   GREEN: {len(dets['GREEN'])}   ", end="\r")
            if HAVE_GUI:
                cv2.imshow("red/green  (press q to quit)", annotate(frame, dets))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()

    elif os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.*")))
        hits = 0
        for f in files:
            frame = cv2.imread(f)
            if frame is None:
                continue
            dets = detect(frame)
            n = len(dets["RED"]) + len(dets["GREEN"])
            hits += n > 0
            print(f"{os.path.basename(f):20s}  RED={len(dets['RED'])}  GREEN={len(dets['GREEN'])}")
        print(f"\n{hits}/{len(files)} images had at least one detection")

    else:
        frame = cv2.imread(src)
        dets = detect(frame)
        print("detections:", {k: len(v) for k, v in dets.items()})
        for label, boxes in dets.items():
            for (x, y, w, h) in boxes:
                print(f"  {label} at x={x} y={y} w={w} h={h}")
        cv2.imwrite("red_green_out.jpg", annotate(frame, dets))
        print("annotated image saved -> red_green_out.jpg")
        if HAVE_GUI:
            cv2.imshow("red/green  (press any key)", annotate(frame, dets))
            cv2.waitKey(0)

    if HAVE_GUI:
        cv2.destroyAllWindows()
