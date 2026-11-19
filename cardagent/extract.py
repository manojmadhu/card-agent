"""Stage 6 — template-driven field extraction with layered fallbacks.

1) Variant ROI -> line OCR -> normalise -> regex
2) Regex search over full-card OCR text (handles small layout shifts)
3) Optional local VLM (vlm.py) for still-missing required fields; its output is
   re-checked by the same regex so the VLM never has the final word.
"""
from __future__ import annotations

import re

from . import vlm
from .ocr import read_line
from .registry import CARD_H, CARD_W, Family

WHITELISTS = {
    "digits": "0123456789",
    "date_mm_yy": "0123456789/EXPIRYDATE:",
    "emirates_id": "0123456789-ID:",
    "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ.-",
}
FULLTEXT_PATTERNS = {
    "digits": r"\b\d[\d ]{7,18}\d\b",
    "date_mm_yy": r"\b(0[1-9]|1[0-2])/(\d{4}|\d{2})\b",
    "emirates_id": r"\b784-?\d{4}-?\d{7}-?\d\b",
}


def normalise_value(raw: str, kind: str, anchor: str | None) -> str:
    s = raw.upper().strip()
    if anchor:
        s = re.sub(rf"^\W*{anchor}\w*\W*", "", s)
    if kind == "digits":
        return re.sub(r"\D", "", s)
    if kind == "date_mm_yy":
        m = re.search(r"(0[1-9]|1[0-2])\s*/\s*(\d{4}|\d{2})", s)
        return f"{m.group(1)}/{m.group(2)}" if m else s
    if kind == "emirates_id":
        d = re.sub(r"\D", "", s)
        return f"{d[:3]}-{d[3:7]}-{d[7:14]}-{d[14:15]}" if len(d) == 15 else s
    if kind == "upper":
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z .'-]", "", s)).strip()
    return s


def extract_fields(fam: Family, img, full_ocr: dict) -> dict:
    out = {}
    for f in fam.fields:
        name, kind, rx = f["name"], f.get("normalise", ""), re.compile(f["regex"])
        x0, y0, x1, y1 = f["roi"]
        pad = 0.01
        crop = img[int(max(0, y0 - pad) * CARD_H):int(min(1, y1 + pad) * CARD_H),
                   int(max(0, x0 - pad) * CARD_W):int(min(1, x1 + pad) * CARD_W)]
        raw, conf = read_line(crop)  # whitelists hurt LSTM accuracy; normalise() cleans instead
        val = normalise_value(raw, kind, f.get("anchor"))
        result = {"value": val or None, "raw": raw, "confidence": round(conf, 3),
                  "source": "ocr_roi", "valid_format": bool(val and rx.match(val))}

        if not result["valid_format"] and kind in FULLTEXT_PATTERNS:
            for m in re.finditer(FULLTEXT_PATTERNS[kind], full_ocr["text"]):
                cand = normalise_value(m.group(0), kind, None)
                if rx.match(cand):
                    result.update(value=cand, source="ocr_fulltext", confidence=0.7, valid_format=True)
                    break

        if not result["valid_format"] and f.get("required") and vlm.enabled():
            v = vlm.extract_field(img, name, f["regex"])
            if v and rx.match(normalise_value(v, kind, None)):
                result.update(value=normalise_value(v, kind, None), source="vlm",
                              confidence=0.6, valid_format=True)
        result["required"] = bool(f.get("required"))
        result["roi"] = [x0, y0, x1, y1]
        result["regex"] = f["regex"]
        out[name] = result
    return out
