"""Stage 1 — image quality gate (fail fast, ask capture system to retake)."""
from __future__ import annotations

import cv2
import numpy as np

from .config import DEFAULTS, Thresholds


def assess(img_bgr: np.ndarray, t: Thresholds = DEFAULTS) -> dict:
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # on the 1000-px normalised size
    brightness = float(gray.mean() / 255.0)
    glare = float((gray > 250).mean())  # share of near-white saturated pixels

    checks = [
        {"name": "resolution (short side, px)", "value": min(h, w), "op": ">=", "limit": t.min_side,
         "ok": min(h, w) >= t.min_side, "code": "LOW_RESOLUTION"},
        {"name": "sharpness (Laplacian variance)", "value": round(blur, 1), "op": ">=", "limit": t.min_blur,
         "ok": blur >= t.min_blur, "code": "IMAGE_BLURRY"},
        {"name": "brightness", "value": round(brightness, 3), "op": "between",
         "limit": [t.min_brightness, t.max_brightness],
         "ok": t.min_brightness <= brightness <= t.max_brightness, "code": "BAD_EXPOSURE"},
        {"name": "glare ratio", "value": round(glare, 4), "op": "<=", "limit": t.max_glare,
         "ok": glare <= t.max_glare, "code": "GLARE"},
    ]
    issues = [c["code"] for c in checks if not c["ok"]]
    return {
        "width": w, "height": h, "blur": round(blur, 1),
        "brightness": round(brightness, 3), "glare_ratio": round(glare, 4),
        "passed": not issues, "issues": issues, "checks": checks,
    }
