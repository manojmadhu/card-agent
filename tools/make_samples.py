"""Generate SYNTHETIC test cards (fictional designs marked SAMPLE) for the prototype.

Replace with real, consented card images for calibration before production.
Usage: python tools/make_samples.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

W, H = 1012, 638
# First existing font wins: Linux DejaVu, then Windows/macOS equivalents (Pillow searches the OS font dir).
FONT_B = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]
FONT_M = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Bold.ttf", "consolab.ttf", "courbd.ttf",
          "Courier New Bold.ttf"]

LAYOUT = {  # y positions (fraction) match registry ROIs
    "fazaa": {"number": 0.42, "name": 0.57, "exp": (0.06, 0.745), "id": None},
    "esaad": {"number": 0.46, "name": 0.64, "exp": (0.62, 0.80), "id": (0.06, 0.80)},
}
NAMES = ["AHMED ALI HASSAN", "FATIMA SAEED", "OMAR KHALID", "MARIAM YOUSEF", "SARA AL MANSOORI",
         "KHALID RASHID", "NOURA SALEM", "YOUSEF IBRAHIM", "HESSA MOHAMMED", "RASHID OBAID"]


def font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    raise OSError(f"none of these fonts are available: {paths}")


def render(family: str, variant: dict, name: str, number: str, exp: str, eid: str | None,
           title: str | None = None) -> np.ndarray:
    bg = tuple(variant["body_color_rgb"])
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg = (255, 255, 255) if lum < 120 else (20, 20, 25)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    accent = tuple(min(255, c + 35) if lum < 120 else max(0, c - 35) for c in bg)
    # family-specific decoration (gives visual matcher something to work with)
    if family == "fazaa":
        d.ellipse([W - 230, 30, W - 70, 190], outline=fg, width=10)
        d.polygon([(W - 150, 55), (W - 105, 160), (W - 195, 160)], fill=accent, outline=fg)
        d.rectangle([0, int(H * 0.30), W, int(H * 0.33)], fill=accent)
    elif family == "esaad":
        cx, cy = W - 150, 115
        pts = [(cx + 80 * np.cos(a) * (1 if i % 2 == 0 else 0.45), cy + 80 * np.sin(a) * (1 if i % 2 == 0 else 0.45))
               for i, a in enumerate(np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1])]
        d.polygon(pts, fill=accent, outline=fg)
        for k in range(4):
            d.line([(int(W * 0.55) + 40 * k, int(H * 0.34)), (int(W * 0.75) + 40 * k, int(H * 0.40))], fill=accent, width=6)
    else:
        d.rectangle([W - 220, 40, W - 80, 180], outline=fg, width=8)

    d.text((int(W * 0.06), int(H * 0.08)), title or family.upper(), font=font(FONT_B, 80), fill=fg)
    d.text((int(W * 0.06), int(H * 0.25)), variant.get("label", variant["id"].upper()), font=font(FONT_B, 36), fill=fg)
    d.text((int(W * 0.62), int(H * 0.27)), "SAMPLE", font=font(FONT_B, 22), fill=accent)
    L = LAYOUT.get(family, LAYOUT["fazaa"])
    spaced = " ".join(number[i:i + 4] for i in range(0, len(number), 4))
    d.text((int(W * 0.06), int(H * L["number"])), spaced, font=font(FONT_M, 58), fill=fg)
    d.text((int(W * 0.06), int(H * L["name"])), name, font=font(FONT_B, 46), fill=fg)
    ex, ey = L["exp"]
    d.text((int(W * ex), int(H * ey)), f"EXP {exp}", font=font(FONT_M, 40), fill=fg)
    if L["id"] and eid:
        ix, iy = L["id"]
        d.text((int(W * ix), int(H * iy)), f"ID {eid}", font=font(FONT_M, 34), fill=fg)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def photograph(card: np.ndarray, seed: int, upside_down=False, blur=0) -> np.ndarray:
    """Simulate a capture: table background, rotation, perspective, lighting, noise."""
    rnd = np.random.default_rng(seed)
    if upside_down:
        card = cv2.rotate(card, cv2.ROTATE_180)
    BW, BH = 1600, 1200
    tone = int(rnd.integers(70, 150))
    bg = np.full((BH, BW, 3), tone, np.uint8)
    bg = cv2.add(bg, rnd.normal(0, 12, bg.shape).astype(np.int16).clip(-40, 40).astype(np.uint8))
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    cx, cy, s = BW / 2 + rnd.uniform(-80, 80), BH / 2 + rnd.uniform(-60, 60), rnd.uniform(0.95, 1.15)
    ang = np.deg2rad(rnd.uniform(-12, 12))
    base = (src - [W / 2, H / 2]) * s
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (base @ rot.T + [cx, cy] + rnd.uniform(-25, 25, (4, 2))).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, M, (BW, BH))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (BW, BH))
    out = np.where(mask[..., None] > 0, warped, bg)
    gain = rnd.uniform(0.85, 1.12)
    out = np.clip(out.astype(np.float32) * gain + rnd.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out


def main():
    random.seed(7)
    fams = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in (ROOT / "registry").glob("*.yaml")}
    # reference images: 3 clean per variant
    for fam, cfg in fams.items():
        for v in cfg["variants"]:
            ref_dir = ROOT / "registry" / "refs" / fam / v["id"]
            ref_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                n_len = 10 if fam == "fazaa" else 12
                num = v["number_prefix"] + "".join(random.choice("0123456789") for _ in range(n_len - len(v["number_prefix"])))
                img = render(fam, v, random.choice(NAMES), num, "06/29", "784-1988-7654321-2")
                cv2.imwrite(str(ref_dir / f"ref_{i}.png"), img)

    sd = ROOT / "samples"
    sd.mkdir(exist_ok=True)
    V = {f: {v["id"]: v for v in c["variants"]} for f, c in fams.items()}
    cases = [  # (file, family, variant, name, number, exp, eid, kwargs, expected)
        ("fazaa_silver.jpg", "fazaa", "silver", "AHMED ALI HASSAN", "1042873391", "08/28", None, {}, "ACCEPT"),
        ("fazaa_gold.jpg", "fazaa", "gold", "FATIMA SAEED", "2210457783", "11/27", None, {}, "ACCEPT"),
        ("fazaa_platinum.jpg", "fazaa", "platinum", "OMAR KHALID", "3198822450", "02/29", None, {}, "ACCEPT"),
        ("fazaa_gold_upside_down.jpg", "fazaa", "gold", "MARIAM YOUSEF", "2877001234", "05/28", None, {"upside_down": True}, "ACCEPT"),
        ("fazaa_gold_prefix_mismatch.jpg", "fazaa", "gold", "KHALID RASHID", "1877001299", "05/28", None, {}, "REVIEW"),
        ("esaad_classic.jpg", "esaad", "classic", "SARA AL MANSOORI", "104455667788", "12/27", "784-1990-1234567-1", {}, "ACCEPT"),
        ("esaad_gold.jpg", "esaad", "gold", "NOURA SALEM", "209988776655", "07/28", "784-1985-7654321-3", {}, "ACCEPT"),
        ("esaad_platinum.jpg", "esaad", "platinum", "YOUSEF IBRAHIM", "301122334455", "09/27", "784-1979-1112223-4", {}, "ACCEPT"),
        ("esaad_privilege.jpg", "esaad", "privilege", "HESSA MOHAMMED", "405566778899", "01/29", "784-1992-3334445-5", {}, "ACCEPT"),
        ("esaad_gold_expired.jpg", "esaad", "gold", "RASHID OBAID", "201234567890", "03/25", "784-1980-9998887-6", {}, "REJECT"),
        ("fazaa_blurry.jpg", "fazaa", "silver", "AHMED ALI HASSAN", "1042873391", "08/28", None, {"blur": 9}, "REJECT"),
    ]
    manifest = []
    for i, (fn, fam, vid, name, num, exp, eid, kw, expected) in enumerate(cases):
        card = render(fam, V[fam][vid], name, num, exp, eid)
        cv2.imwrite(str(sd / fn), photograph(card, seed=100 + i, **kw), [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest.append({"file": fn, "family": fam, "variant": vid, "expected_status": expected,
                         "card_number": num, "holder_name": name})
    # unknown card type
    unk = render("other", {"id": "member", "label": "MEMBER", "body_color_rgb": [120, 30, 60]},
                 "JOHN SMITH", "5555444433", "10/28", None, title="ACME REWARDS")
    cv2.imwrite(str(sd / "unknown_card.jpg"), photograph(unk, seed=999))
    manifest.append({"file": "unknown_card.jpg", "family": None, "variant": None, "expected_status": "REJECT"})
    import json
    (sd / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest)} samples + references")


if __name__ == "__main__":
    main()
