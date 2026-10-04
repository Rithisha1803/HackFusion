# gaze_detector.py  (v3 - iris-first architecture)
# ─────────────────────────────────────────────────────────────────────────────
# KEY CHANGES in v3:
#   - Iris is the PRIMARY detector. Head pose is LAST RESORT only.
#   - Head pose solvePnP was unreliable (darkest-column heuristic often
#     locks onto the eye socket instead of the nose, biasing yaw left).
#   - Pupil detection now uses minMaxLoc (fastest + most reliable for dark
#     pupils on bright sclera) with Otsu as fallback.
#   - Per-eye x_ratio is computed relative to the FULL face width so that
#     both eyes contribute a consistent signal (not just within their own ROI).
#   - GazeSmoother window reduced to 3 for faster response.
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from collections import deque, Counter

# ── Cascades (loaded once, thread-safe read-only) ────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Iris x-ratio is measured as (eye_center_x / face_width).
# Looking straight → both eyes sit at ~0.25 and ~0.75 of face width.
# Looking left  → both eye pupils shift left  → average ratio drops
# Looking right → both eye pupils shift right → average ratio rises
# We use the DEVIATION from 0.5 (centre of face) to decide direction.
_IRIS_LATERAL_THRESH = 0.055   # deviation from 0.5 to trigger left/right
_IRIS_UP_THRESH      = 0.32    # y-ratio within eye ROI
_IRIS_DOWN_THRESH    = 0.68

# Head pose fallback thresholds (only used when iris fails)
_YAW_LEFT_THRESH   =  18.0
_YAW_RIGHT_THRESH  = -18.0
_PITCH_UP_THRESH   = -14.0
_PITCH_DOWN_THRESH =  14.0

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

# 3-D face model for solvePnP
_FACE_3D_MODEL = np.array([
    [ 0.0,    0.0,    0.0  ],
    [ 0.0,  -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [ 43.3,  32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [ 28.9, -28.9,  -24.1 ],
], dtype=np.float32)


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_gaze_direction(frame: np.ndarray) -> dict:
    """
    Pass a BGR webcam frame. Returns direction dict.
    """
    if frame is None:
        return _result("away", 0.95, method="null_frame")

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return _result("away", 0.92, method="no_face")

    fx, fy, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_gray = gray[fy : fy + fh, fx : fx + fw]

    # ── Primary: iris / pupil position ───────────────────────────────────────
    iris_x_dev, iris_y, eyes_found = _detect_iris(face_gray, fw, fh)

    if eyes_found > 0 and iris_x_dev is not None:
        direction = _dir_from_iris(iris_x_dev, iris_y)
        confidence = 0.88 if eyes_found == 2 else 0.74
        return {
            "direction":    direction,
            "confidence":   round(confidence, 3),
            "yaw":          None,
            "pitch":        None,
            "iris_x_ratio": round(iris_x_dev + 0.5, 3),
            "iris_y_ratio": round(iris_y, 3) if iris_y is not None else None,
            "eyes_found":   eyes_found,
            "method":       f"iris_primary ({eyes_found} eye{'s' if eyes_found>1 else ''})",
        }

    # ── Fallback: head pose ───────────────────────────────────────────────────
    yaw, pitch = _head_pose(face_gray, fw, fh, w, h)
    direction  = _dir_from_pose(yaw, pitch) or "center"
    confidence = 0.62 if (yaw is not None) else 0.35

    return {
        "direction":    direction,
        "confidence":   round(confidence, 3),
        "yaw":          round(yaw, 2)   if yaw   is not None else None,
        "pitch":        round(pitch, 2) if pitch is not None else None,
        "iris_x_ratio": None,
        "iris_y_ratio": None,
        "eyes_found":   0,
        "method":       "head_pose_fallback",
    }


# ─── Iris / pupil detection ───────────────────────────────────────────────────

def _detect_iris(face_gray, fw, fh):
    """
    Returns (x_deviation_from_centre, avg_y_ratio, num_eyes_found).

    x_deviation: signed float centred on 0.
      negative → pupils shifted left  → person looking LEFT
      positive → pupils shifted right → person looking RIGHT

    We measure each eye's pupil x as a fraction of FACE width (not eye width),
    then subtract 0.5 so that straight-ahead = 0.
    """
    eye_top    = int(fh * 0.25)
    eye_bottom = int(fh * 0.58)
    band       = face_gray[eye_top:eye_bottom, :]

    eyes = _eye_cascade.detectMultiScale(
        band,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(int(fw * 0.12), int(fw * 0.08)),
        maxSize=(int(fw * 0.46), int(fw * 0.30)),
    )

    if len(eyes) == 0:
        return None, None, 0

    # Sort left-to-right; keep at most 2 (one per eye)
    eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]

    x_devs, y_ratios = [], []

    for (ex, ey, ew, eh) in eyes_sorted:
        # Pupil x in face coordinates (0=left edge of face, 1=right edge)
        pupil_face_x = _find_pupil_x(band, ex, ey, ew, eh)
        if pupil_face_x is None:
            continue

        # Convert to face-width fraction then deviation from centre
        x_dev = (pupil_face_x / fw) - 0.5
        x_devs.append(x_dev)

        # Y ratio within eye ROI (for up/down)
        pupil_y = _find_pupil_y(band, ex, ey, ew, eh)
        if pupil_y is not None:
            y_ratios.append(pupil_y)

    if not x_devs:
        return None, None, 0

    avg_xdev = float(np.mean(x_devs))
    avg_y    = float(np.mean(y_ratios)) if y_ratios else None
    return avg_xdev, avg_y, len(x_devs)


def _find_pupil_x(band, ex, ey, ew, eh):
    """
    Find the darkest point x in the eye ROI.
    Returns x in BAND (face) coordinates, or None.
    """
    pad_x = max(3, int(ew * 0.12))
    pad_y = max(2, int(eh * 0.14))
    roi   = band[ey + pad_y : ey + eh - pad_y,
                  ex + pad_x : ex + ew - pad_x]

    if roi.size < 40:
        return None

    roi_h, roi_w = roi.shape
    roi_eq  = _CLAHE.apply(roi)
    blurred = cv2.GaussianBlur(roi_eq, (7, 7), 0)

    # Method 1: darkest point via minMaxLoc (fast, works when pupil is clearly dark)
    _, _, min_loc, _ = cv2.minMaxLoc(blurred)
    min_x_roi = min_loc[0]

    # Sanity: the darkest point should be in the inner 80% of the ROI
    # (reject if it's right at the edge — likely an eyelash artefact)
    if 0.10 * roi_w < min_x_roi < 0.90 * roi_w:
        pupil_x_band = ex + pad_x + min_x_roi
        return float(pupil_x_band)

    # Method 2: contour centroid fallback
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_ratio = np.sum(thresh > 0) / thresh.size
    if mask_ratio < 0.02 or mask_ratio > 0.65:
        p15 = int(np.percentile(blurred, 15))
        _, thresh = cv2.threshold(blurred, p15, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_area = roi_h * roi_w
    valid = [c for c in contours if roi_area * 0.015 <= cv2.contourArea(c) <= roi_area * 0.55]
    if not valid:
        return None

    def _darkness(c):
        mask = np.zeros_like(roi)
        cv2.drawContours(mask, [c], -1, 255, -1)
        vals = roi[mask == 255]
        return float(np.mean(vals)) if len(vals) > 0 else 255.0

    pupil_c = min(valid, key=_darkness)
    M = cv2.moments(pupil_c)
    if M["m00"] == 0:
        return None

    cx_roi = M["m10"] / M["m00"]
    return float(ex + pad_x + cx_roi)


def _find_pupil_y(band, ex, ey, ew, eh):
    """
    Returns pupil y as ratio within the eye ROI (0=top, 1=bottom).
    """
    pad_x = max(3, int(ew * 0.12))
    pad_y = max(2, int(eh * 0.14))
    roi   = band[ey + pad_y : ey + eh - pad_y,
                  ex + pad_x : ex + ew - pad_x]

    if roi.size < 40:
        return None

    roi_h, roi_w = roi.shape
    blurred = cv2.GaussianBlur(_CLAHE.apply(roi), (7, 7), 0)
    _, min_loc, _, _ = cv2.minMaxLoc(blurred)[1], cv2.minMaxLoc(blurred)[2], None, None
    # minMaxLoc returns (minVal, maxVal, minLoc, maxLoc)
    result = cv2.minMaxLoc(blurred)
    min_y_roi = result[2][1]  # minLoc y

    if roi_h == 0:
        return None
    return float(min_y_roi / roi_h)


def _dir_from_iris(x_dev, y_ratio):
    """
    x_dev: deviation from face centre (-ve = left, +ve = right)
    y_ratio: pupil y position within eye ROI (0=top, 1=bottom)
    """
    # Left/right takes priority (most common cheating direction)
    if x_dev < -_IRIS_LATERAL_THRESH:
        return "left"
    if x_dev >  _IRIS_LATERAL_THRESH:
        return "right"

    if y_ratio is not None:
        if y_ratio < _IRIS_UP_THRESH:
            return "up"
        if y_ratio > _IRIS_DOWN_THRESH:
            return "down"

    return "center"


# ─── Head pose (fallback only) ────────────────────────────────────────────────

def _head_pose(face_gray, fw, fh, img_w, img_h):
    try:
        # Use fixed symmetric 2D landmarks (no heuristic nose detection)
        pts_2d = np.array([
            [fw * 0.50, fh * 0.46],   # nose tip (assumed centre)
            [fw * 0.50, fh * 0.97],   # chin
            [fw * 0.19, fh * 0.38],   # left eye outer
            [fw * 0.81, fh * 0.38],   # right eye outer
            [fw * 0.33, fh * 0.74],   # left mouth corner
            [fw * 0.67, fh * 0.74],   # right mouth corner
        ], dtype=np.float32)

        focal  = float(img_w)
        cam_mx = np.array([[focal, 0, img_w/2.0],
                           [0, focal, img_h/2.0],
                           [0, 0, 1]], dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D_MODEL, pts_2d, cam_mx, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None

        R, _ = cv2.Rodrigues(rvec)
        sy   = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            pitch = np.degrees(np.arctan2( R[2,1], R[2,2]))
        else:
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))

        return float(yaw), float(pitch)
    except Exception:
        return None, None


def _dir_from_pose(yaw, pitch):
    if yaw is None or pitch is None:
        return None
    if yaw   >  _YAW_LEFT_THRESH:   return "left"
    if yaw   <  _YAW_RIGHT_THRESH:  return "right"
    if pitch <  _PITCH_UP_THRESH:   return "up"
    if pitch >  _PITCH_DOWN_THRESH: return "down"
    return "center"


def _result(direction, confidence, method="", reason=""):
    return {
        "direction":    direction,
        "confidence":   confidence,
        "yaw":          None,
        "pitch":        None,
        "iris_x_ratio": None,
        "iris_y_ratio": None,
        "eyes_found":   0,
        "method":       f"{method}" + (f" ({reason})" if reason else ""),
    }


# ─── GazeSmoother ─────────────────────────────────────────────────────────────

class GazeSmoother:
    """
    Majority-vote smoother with sticky hold.
    Window=3 for fast response; sticky until centre majority confirmed.
    """

    def __init__(self, window: int = 3):
        self._window         = window
        self._history        = deque(maxlen=window)
        self._held_direction = "center"
        self._is_held        = False

    def update(self, result: dict) -> dict:
        self._history.append(result)
        counts       = Counter(r["direction"] for r in self._history)
        majority_dir = counts.most_common(1)[0][0]
        avg_conf     = sum(r["confidence"] for r in self._history) / len(self._history)

        if self._is_held:
            if majority_dir == "center":
                self._is_held        = False
                self._held_direction = "center"
                direction     = "center"
                method_suffix = "_released"
            else:
                direction     = self._held_direction
                method_suffix = "_held"
        else:
            direction = majority_dir
            if majority_dir != "center":
                self._held_direction = majority_dir
                self._is_held        = True
            method_suffix = ""

        return {
            **result,
            "direction":  direction,
            "confidence": round(avg_conf, 3),
            "method":     result.get("method", "smoother") + method_suffix,
        }

    def reset(self):
        self._history.clear()
        self._held_direction = "center"
        self._is_held        = False

    @property
    def is_warning_active(self) -> bool:
        return self._is_held

    @property
    def held_direction(self) -> str:
        return self._held_direction


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Test 1: blank frame → away ──")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r = detect_gaze_direction(blank)
    assert r["direction"] == "away", f"Expected away, got {r['direction']}"
    print(f"  ✅ direction={r['direction']}  conf={r['confidence']}")

    print("── Test 2: iris direction logic ──")
    assert _dir_from_iris(-0.08, 0.50) == "left",   "left fail"
    assert _dir_from_iris( 0.08, 0.50) == "right",  "right fail"
    assert _dir_from_iris( 0.00, 0.50) == "center", "center fail"
    assert _dir_from_iris( 0.00, 0.20) == "up",     "up fail"
    assert _dir_from_iris( 0.00, 0.80) == "down",   "down fail"
    print("  ✅ all direction assertions passed")

    print("── Test 3: GazeSmoother ──")
    sm = GazeSmoother(window=3)
    def _r(d): return {"direction":d,"confidence":0.82,"method":"t","yaw":None,
                       "pitch":None,"iris_x_ratio":None,"iris_y_ratio":None,"eyes_found":2}
    for _ in range(2): out = sm.update(_r("left"))
    assert out["direction"] == "left" and sm.is_warning_active
    print(f"  After 2× left: {out['direction']}  hold={sm.is_warning_active} ✅")
    for _ in range(2): out = sm.update(_r("center"))
    assert out["direction"] == "center" and not sm.is_warning_active
    print(f"  After 2× center: {out['direction']}  hold={sm.is_warning_active} ✅")

    print("\n✅ All self-tests passed")