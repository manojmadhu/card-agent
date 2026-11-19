"""Stages 4 & 5 — hierarchical classification: FAMILY first, then VARIANT within it.

Each level fuses independent signals and requires BOTH a minimum score and a
minimum margin over the runner-up, so look-alike cards go to review instead of
being confidently mislabelled.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .config import DEFAULTS, Thresholds
from .features import body_colour_rgb, colour_similarity, visual_similarity, visual_signature
from .registry import Family, Registry

VARIANT_WEIGHTS = {"colour": 0.45, "keyword": 0.30, "visual": 0.15, "prefix": 0.10}


def keyword_match(tokens: list[str], text: str, keywords: list[str]) -> tuple[float, str | None]:
    """(score, matched keyword): 1.0 for an exact keyword, partial credit for a fuzzy match (OCR errors)."""
    best, hit = 0.0, None
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return 1.0, kw
        for tok in tokens:
            if abs(len(tok) - len(kw)) <= 2:
                r = SequenceMatcher(None, tok, kw).ratio()
                if r >= 0.75 and r > best:
                    best, hit = r, kw
    return best, hit


def keyword_score(tokens: list[str], text: str, keywords: list[str]) -> float:
    return keyword_match(tokens, text, keywords)[0]


def _best_ref(sig: dict, refs: list[dict]) -> tuple[float, str | None]:
    best, ref_id = 0.0, None
    for r in refs:
        s = visual_similarity(sig, r)
        if s > best:
            best, ref_id = s, r.get("id")
    return best, ref_id


def classify_family(reg: Registry, img, ocr: dict, t: Thresholds = DEFAULTS) -> dict:
    sig = visual_signature(img)
    body = body_colour_rgb(img)
    cands = []
    for fam in reg.families.values():
        s_text, kw_hit = keyword_match(ocr["tokens"], ocr["text"], fam.keywords)
        neg, neg_hit = keyword_match(ocr["tokens"], ocr["text"], fam.negative_keywords)
        if neg >= 0.9:
            s_text *= 0.2
        s_vis, ref_id = _best_ref(sig, [r for v in fam.variants for r in v.refs])
        s_col = max(colour_similarity(body, v.body_color_rgb) for v in fam.variants)
        w = fam.weights
        score = w["text"] * s_text + w["visual"] * s_vis + w["colour"] * s_col
        cands.append({"family": fam.family, "name": fam.family_name, "score": round(score, 3),
                      "signals": {"text": round(s_text, 3), "visual": round(s_vis, 3),
                                  "colour": round(s_col, 3)},
                      "weights": w, "keyword_hit": kw_hit, "negative_hit": neg_hit if neg >= 0.9 else None,
                      "best_ref": ref_id})
    cands.sort(key=lambda c: c["score"], reverse=True)
    top = cands[0] if cands else None
    margin = top["score"] - (cands[1]["score"] if len(cands) > 1 else 0.0) if top else 0.0
    decided = bool(top and top["score"] >= t.family_min and margin >= t.family_margin)
    return {"decided": decided, "family": top["family"] if decided else None,
            "confidence": top["score"] if top else 0.0, "margin": round(margin, 3),
            "candidates": cands, "body_rgb": body, "signature": sig}


def classify_variant(fam: Family, img, ocr: dict, sig: dict, body_rgb, t: Thresholds = DEFAULTS) -> dict:
    digits = re.findall(r"\d{8,}", ocr["text"].replace(" ", ""))
    W = VARIANT_WEIGHTS
    cands = []
    for v in fam.variants:
        s_col = colour_similarity(body_rgb, v.body_color_rgb, scale=25.0)
        s_kw, kw_hit = keyword_match(ocr["tokens"], ocr["text"], v.keywords) if v.keywords else (0.0, None)
        s_vis, ref_id = _best_ref(sig, v.refs)
        s_pre = 1.0 if v.number_prefix and any(d.startswith(v.number_prefix) for d in digits) else 0.0
        score = W["colour"] * s_col + W["keyword"] * s_kw + W["visual"] * s_vis + W["prefix"] * s_pre
        cands.append({"variant": v.id, "name": v.name, "score": round(score, 3), "rgb": list(v.body_color_rgb),
                      "signals": {"colour": round(s_col, 3), "keyword": round(s_kw, 3),
                                  "visual": round(s_vis, 3), "prefix": s_pre},
                      "keyword_hit": kw_hit, "best_ref": ref_id})
    cands.sort(key=lambda c: c["score"], reverse=True)
    top = cands[0]
    margin = top["score"] - (cands[1]["score"] if len(cands) > 1 else 0.0)
    decided = top["score"] >= t.variant_min and margin >= t.variant_margin
    return {"decided": decided, "variant": top["variant"] if decided else None,
            "confidence": top["score"], "margin": round(margin, 3), "candidates": cands,
            "number_seen": digits[0] if digits else None}
