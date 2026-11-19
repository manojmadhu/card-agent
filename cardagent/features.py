"""Local visual features (no external model).

Prototype uses ORB keypoints + HSV histogram + body colour.
Production: swap `visual_signature`/`visual_similarity` for a local CLIP/DINOv2
embedding + vector index (Qdrant/FAISS); the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

_orb = cv2.ORB_create(nfeatures=800)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def body_colour_rgb(img_bgr: np.ndarray) -> tuple[int, int, int]:
    """Median colour of the card body, ignoring text (sample a border band)."""
    h, w = img_bgr.shape[:2]
    band = np.concatenate([
        img_bgr[int(h * 0.03):int(h * 0.12), int(w * 0.03):int(w * 0.97)].reshape(-1, 3),
        img_bgr[int(h * 0.90):int(h * 0.97), int(w * 0.03):int(w * 0.97)].reshape(-1, 3),
        img_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.90):int(w * 0.97)].reshape(-1, 3),
    ])
    b, g, r = np.median(band, axis=0)
    return int(r), int(g), int(b)


def _lab(rgb) -> np.ndarray:
    px = np.uint8([[list(rgb)[::-1]]])  # to BGR
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def colour_distance(rgb_a, rgb_b) -> float:
    """Approximate CIE76 delta-E in OpenCV's Lab scaling."""
    return float(np.linalg.norm(_lab(rgb_a) - _lab(rgb_b)))


def colour_similarity(rgb_a, rgb_b, scale: float = 35.0) -> float:
    return float(np.exp(-colour_distance(rgb_a, rgb_b) / scale))


def visual_signature(img_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = _orb.detectAndCompute(gray, None)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return {"des": des, "hist": hist, "n_kp": len(kp)}


def visual_similarity(sig_a: dict, sig_b: dict) -> float:
    """0..1 — blend of keypoint match ratio and colour-histogram correlation."""
    orb_score = 0.0
    if sig_a["des"] is not None and sig_b["des"] is not None and len(sig_b["des"]) >= 2:
        matches = _bf.knnMatch(sig_a["des"], sig_b["des"], k=2)
        good = [m for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        orb_score = min(1.0, len(good) / 60.0)
    hist_score = max(0.0, float(cv2.compareHist(sig_a["hist"], sig_b["hist"], cv2.HISTCMP_CORREL)))
    return 0.6 * orb_score + 0.4 * hist_score
