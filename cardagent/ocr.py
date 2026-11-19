"""Local OCR wrapper (Tesseract 5, eng+ara). Swap for PaddleOCR in production."""
from __future__ import annotations

import functools
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

WANTED_LANGS = [x for x in os.environ.get("CARD_AGENT_OCR_LANGS", "eng+ara").split("+") if x]
log = logging.getLogger("card-agent.ocr")

_WINDOWS_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


def _resolve_tesseract() -> None:
    """Point pytesseract at the binary: $TESSERACT_CMD, then PATH, then common Windows install dirs."""
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd and os.name == "nt":
        cmd = next((str(p) for p in _WINDOWS_CANDIDATES if p.is_file()), None)
    if not cmd:
        raise RuntimeError(
            "Tesseract OCR binary not found. Install it (Windows: `winget install UB-Mannheim.TesseractOCR`, "
            "tick the Arabic language pack; Linux: `apt install tesseract-ocr tesseract-ocr-ara`), "
            "or set TESSERACT_CMD to the full path of tesseract.exe."
        )
    pytesseract.pytesseract.tesseract_cmd = cmd


_resolve_tesseract()


@functools.lru_cache(maxsize=1)
def installed_languages() -> list[str]:
    try:
        return sorted(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001
        return []


@functools.lru_cache(maxsize=1)
def full_lang() -> str:
    """eng+ara when both are installed; otherwise whatever subset is (with a warning), never a crash."""
    have = installed_languages()
    use = [lang for lang in WANTED_LANGS if lang in have] if have else WANTED_LANGS
    missing = [lang for lang in WANTED_LANGS if have and lang not in have]
    if missing:
        log.warning("Tesseract language data missing: %s — Arabic keywords will not be read. "
                    "Install tesseract-ocr-%s.", ", ".join(missing), missing[0])
    return "+".join(use or ["eng"])


@functools.lru_cache(maxsize=1)
def engine_version() -> str:
    return f"tesseract-{pytesseract.get_tesseract_version()}"


def _prep(img_bgr: np.ndarray, invert: bool | None = None, upscale: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if invert is None:
        invert = gray.mean() < 110  # light text on dark card
    if invert:
        gray = 255 - gray
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def read_full(img_bgr: np.ndarray) -> dict:
    """OCR the whole card in both polarities and merge tokens."""
    tokens, confs = [], []
    for inv in (False, True):
        data = pytesseract.image_to_data(_prep(img_bgr, invert=inv), lang=full_lang(), config="--psm 11",
                                         output_type=pytesseract.Output.DICT)
        for txt, conf in zip(data["text"], data["conf"]):
            txt = txt.strip()
            if txt and float(conf) > 30:
                tokens.append(txt.upper())
                confs.append(round(float(conf) / 100.0, 3))
    return {"tokens": tokens, "confs": confs, "text": " ".join(tokens),
            "mean_conf": float(np.mean(confs)) if confs else 0.0}


def read_line(img_bgr: np.ndarray, whitelist: str | None = None) -> tuple[str, float]:
    cfg = "--psm 7"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    best = ("", 0.0)
    for inv in (False, True):  # try both polarities, keep the more confident read
        data = pytesseract.image_to_data(_prep(img_bgr, invert=inv, upscale=2.0), lang="eng", config=cfg,
                                         output_type=pytesseract.Output.DICT)
        words = [(t.strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]
        if words:
            cand = (" ".join(w for w, _ in words), float(np.mean([c for _, c in words])) / 100.0)
            if cand[1] > best[1]:
                best = cand
    return best
