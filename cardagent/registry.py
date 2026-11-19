"""Card Registry: loads card-family YAML definitions + reference images.

Adding a card family/variant = drop a YAML in registry/ and reference images in
registry/refs/<family>/<variant>/*.(png|jpg|jpeg), then call reload(). No code change.

A reload builds a complete new index and swaps it in atomically, so requests in flight keep
using the previous one. A family file that fails to load is reported in `errors` and skipped,
instead of taking every other family down with it.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from .features import visual_signature

CARD_W, CARD_H = 1012, 638  # ISO ID-1 aspect ratio
REF_EXTS = (".png", ".jpg", ".jpeg")

log = logging.getLogger("card-agent.registry")


@dataclass
class Variant:
    id: str
    name: str
    body_color_rgb: tuple[int, int, int]
    keywords: list[str]
    number_prefix: str | None
    refs: list[dict] = field(default_factory=list)  # visual signatures, each with an "id"


@dataclass
class Family:
    family: str
    family_name: str
    issuer: str
    keywords: list[str]
    negative_keywords: list[str]
    weights: dict
    variants: list[Variant]
    fields: list[dict]
    rules: list[dict]

    def variant(self, vid: str) -> Variant | None:
        return next((v for v in self.variants if v.id == vid), None)


def ref_files(root: Path, family: str, variant: str) -> list[Path]:
    d = root / "refs" / family / variant
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in REF_EXTS)


def load_family(root: Path, cfg: dict) -> Family:
    variants = []
    for v in cfg["variants"]:
        prefix = v.get("number_prefix")
        var = Variant(
            id=v["id"], name=v["name"],
            body_color_rgb=tuple(int(c) for c in v["body_color_rgb"]),
            keywords=[str(k).upper() for k in v.get("keywords") or []],
            number_prefix=str(prefix) if prefix not in (None, "") else None,
        )
        for img_path in ref_files(root, cfg["family"], v["id"]):
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("unreadable reference image %s", img_path)
                continue
            sig = visual_signature(cv2.resize(img, (CARD_W, CARD_H)))
            sig["id"] = f"{cfg['family']}/{v['id']}/{img_path.name}"
            var.refs.append(sig)
        variants.append(var)
    return Family(
        family=cfg["family"], family_name=cfg["family_name"],
        issuer=cfg.get("issuer", "") or "",
        keywords=[str(k).upper() for k in cfg.get("keywords") or []],
        negative_keywords=[str(k).upper() for k in cfg.get("negative_keywords") or []],
        weights=cfg.get("weights") or {"text": 0.55, "visual": 0.30, "colour": 0.15},
        variants=variants, fields=cfg.get("fields") or [], rules=cfg.get("rules") or [],
    )


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self.families: dict[str, Family] = {}
        self.errors: dict[str, str] = {}
        self.version = ""
        self.reload()

    def reload(self) -> None:
        families, errors, digest = {}, {}, hashlib.sha256()
        for path in sorted(self.root.glob("*.yaml")):
            raw = path.read_bytes()
            digest.update(path.name.encode() + b"\0" + raw)
            try:
                cfg = yaml.safe_load(raw)
                if not isinstance(cfg, dict) or "family" not in cfg:
                    raise ValueError("not a card-family mapping")
                fam = load_family(self.root, cfg)
                for v in fam.variants:  # the reference set is part of the registry version
                    for p in ref_files(self.root, fam.family, v.id):
                        st = p.stat()
                        digest.update(f"{fam.family}/{v.id}/{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
                families[fam.family] = fam
            except Exception as e:  # noqa: BLE001 — one bad file must not break the others
                errors[path.stem] = f"{type(e).__name__}: {e}"
                log.error("registry: %s failed to load: %s", path.name, e)
        with self._lock:
            self.families = families
            self.errors = errors
            self.version = digest.hexdigest()[:12]

    def summary(self) -> dict:
        return {
            "version": self.version,
            "families": [
                {"family": f.family, "name": f.family_name, "issuer": f.issuer,
                 "variants": [{"id": v.id, "name": v.name, "body_color_rgb": list(v.body_color_rgb),
                               "reference_images": len(v.refs)}
                              for v in f.variants]}
                for f in self.families.values()
            ],
            "errors": self.errors,
        }
