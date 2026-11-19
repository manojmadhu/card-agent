"""Registry administration: card-family YAML, version history and reference images.

Only ever touches files under the registry directory:
    registry/<family>.yaml                       live definition
    registry/.history/<family>/<ts>.yaml         every previous version (restore = PUT it back)
    registry/refs/<family>/<variant>/*.png|jpg   reference images
    registry/.trash/...                          deleted families / references (never hard-deleted)
Writes are validated before they land, are atomic, and trigger the registry hot reload.
"""
from __future__ import annotations

import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import normalise
from .features import visual_similarity, visual_signature
from .registry import CARD_H, CARD_W, REF_EXTS, Registry, ref_files

ID_RE = re.compile(r"^[a-z0-9_-]+$")
FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FILE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
RULE_TYPES = {"regex", "not_expired", "emirates_id_format", "luhn", "prefix_matches_variant"}
NORMALISERS = {"upper", "digits", "date_mm_yy", "emirates_id", "none", ""}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_HISTORY = 50


class AdminError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def check_id(value: str, what: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise AdminError(400, f"invalid {what} '{value}' (lowercase a-z, 0-9, _ and - only)")
    return value


def validate_family(doc, family_id: str) -> list[str]:
    """Schema checks mirroring what Registry.load_family and the pipeline rely on."""
    if not isinstance(doc, dict):
        return ["YAML root must be a mapping"]
    errs = []
    for k in ("family", "family_name", "keywords", "variants"):
        if k not in doc:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    if doc["family"] != family_id:
        errs.append(f"'family: {doc['family']}' does not match the id '{family_id}'")
    if not isinstance(doc.get("keywords"), list) or not doc["keywords"]:
        errs.append("keywords: at least one keyword is required")
    for k in ("negative_keywords",):
        if doc.get(k) is not None and not isinstance(doc[k], list):
            errs.append(f"{k} must be a list")
    w = doc.get("weights")
    if w is not None:
        try:
            total = sum(float(w.get(k, 0)) for k in ("text", "visual", "colour"))
            if abs(total - 1) > 0.011:
                errs.append(f"weights must sum to 1 (got {total:.2f})")
        except (AttributeError, TypeError, ValueError):
            errs.append("weights must be {text, visual, colour} numbers")
    variants = doc.get("variants")
    if not isinstance(variants, list) or not variants:
        errs.append("variants: at least one variant is required")
        variants = []
    seen = set()
    for i, v in enumerate(variants):
        p = f"variants[{i}]"
        if not isinstance(v, dict):
            errs.append(f"{p} must be a mapping")
            continue
        vid = v.get("id")
        if not isinstance(vid, str) or not ID_RE.match(vid):
            errs.append(f"{p}.id must be a lowercase id")
        elif vid in seen:
            errs.append(f"{p}.id '{vid}' is duplicated")
        seen.add(vid)
        if not v.get("name"):
            errs.append(f"{p}.name is required")
        rgb = v.get("body_color_rgb")
        if not (isinstance(rgb, list) and len(rgb) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in rgb)):
            errs.append(f"{p}.body_color_rgb must be [r, g, b] integers 0-255")
        pre = v.get("number_prefix")
        if pre not in (None, "") and not re.fullmatch(r"\d+", str(pre)):
            errs.append(f"{p}.number_prefix must be digits")
    names = set()
    for i, f in enumerate(doc.get("fields") or []):
        p = f"fields[{i}]"
        if not isinstance(f, dict):
            errs.append(f"{p} must be a mapping")
            continue
        name = f.get("name")
        if not isinstance(name, str) or not FIELD_RE.match(name):
            errs.append(f"{p}.name must be snake_case")
        names.add(name)
        roi = f.get("roi")
        if not (isinstance(roi, list) and len(roi) == 4 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in roi)
                and roi[2] > roi[0] and roi[3] > roi[1]):
            errs.append(f"{p}.roi must be [x0, y0, x1, y1] fractions with x1>x0, y1>y0")
        try:
            re.compile(f.get("regex") or "")
            if not f.get("regex"):
                errs.append(f"{p}.regex is required")
        except re.error as e:
            errs.append(f"{p}.regex is invalid: {e}")
        if (f.get("normalise") or "") not in NORMALISERS:
            errs.append(f"{p}.normalise must be one of {sorted(n for n in NORMALISERS if n)}")
    for i, r in enumerate(doc.get("rules") or []):
        p = f"rules[{i}]"
        if not isinstance(r, dict):
            errs.append(f"{p} must be a mapping")
            continue
        if r.get("type") not in RULE_TYPES:
            errs.append(f"{p}.type must be one of {sorted(RULE_TYPES)}")
        if r.get("field") not in names:
            errs.append(f"{p} references unknown field '{r.get('field')}'")
    return errs


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _ts_iso(stem: str) -> str:
    try:
        return datetime.strptime(stem, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return stem


def decode_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_IMAGE_BYTES:
        raise AdminError(413, "image larger than 8 MB")
    if not (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"):
        raise AdminError(415, "only PNG or JPEG images are accepted")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AdminError(415, "image could not be decoded")
    return img


class RegistryAdmin:
    def __init__(self, registry: Registry):
        self.reg = registry
        self.root = registry.root.resolve()
        self._lock = threading.Lock()
        self._sep_cache: tuple[str, dict] | None = None

    # ---- families ------------------------------------------------------------------------------
    def _path(self, fid: str) -> Path:
        return self.root / f"{check_id(fid, 'family id')}.yaml"

    def list_families(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.yaml")):
            fid = p.stem
            fam = self.reg.families.get(fid)
            out.append({"family": fid, "yaml": p.read_text(encoding="utf-8"),
                        "loaded": fam is not None, "error": self.reg.errors.get(fid),
                        "updated_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                        "reference_images": sum(len(v.refs) for v in fam.variants) if fam else 0})
        return out

    def get_family(self, fid: str) -> dict:
        p = self._path(fid)
        if not p.exists():
            raise AdminError(404, f"family '{fid}' not found")
        return {"family": fid, "yaml": p.read_text(encoding="utf-8"),
                "loaded": fid in self.reg.families, "error": self.reg.errors.get(fid),
                "updated_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}

    def put_family(self, fid: str, text: str, create: bool = False, note: str | None = None) -> dict:
        p = self._path(fid)
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise AdminError(422, f"YAML parse error: {e}") from e
        errs = validate_family(doc, fid)
        if errs:
            raise AdminError(422, "; ".join(errs))
        with self._lock:
            if create and p.exists():
                raise AdminError(409, f"family '{fid}' already exists")
            if p.exists():
                if p.read_text(encoding="utf-8") == text:
                    return {"ok": True, "family": fid, "changed": False, "registry_version": self.reg.version}
                self._snapshot(fid, p)
            tmp = p.with_suffix(".yaml.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
            self.reg.reload()
        if fid in self.reg.errors:
            raise AdminError(422, f"saved, but the registry could not load it: {self.reg.errors[fid]}")
        return {"ok": True, "family": fid, "changed": True, "registry_version": self.reg.version}

    def delete_family(self, fid: str, with_refs: bool = True) -> dict:
        p = self._path(fid)
        with self._lock:
            if not p.exists():
                raise AdminError(404, f"family '{fid}' not found")
            self._snapshot(fid, p)
            trash = self.root / ".trash" / _ts()
            trash.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(trash / p.name))
            refs = self.root / "refs" / fid
            if with_refs and refs.is_dir():
                shutil.move(str(refs), str(trash / f"refs_{fid}"))
            self.reg.reload()
        return {"ok": True, "registry_version": self.reg.version}

    def _snapshot(self, fid: str, p: Path) -> None:
        d = self.root / ".history" / fid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / f"{_ts()}.yaml")
        for old in sorted(d.glob("*.yaml"))[:-MAX_HISTORY]:
            old.unlink(missing_ok=True)

    def history(self, fid: str) -> list[dict]:
        check_id(fid, "family id")
        d = self.root / ".history" / fid
        items = [{"version": p.stem, "saved_at": _ts_iso(p.stem), "size": p.stat().st_size}
                 for p in sorted(d.glob("*.yaml"), reverse=True)] if d.is_dir() else []
        return items

    def history_version(self, fid: str, version: str) -> dict:
        check_id(fid, "family id")
        if not re.fullmatch(r"\d{8}T\d{12,}Z", version):
            raise AdminError(400, "invalid version")
        p = self.root / ".history" / fid / f"{version}.yaml"
        if not p.is_file():
            raise AdminError(404, "version not found")
        return {"family": fid, "version": version, "saved_at": _ts_iso(version), "yaml": p.read_text(encoding="utf-8")}

    # ---- reference images ----------------------------------------------------------------------
    def _variant_dir(self, fid: str, vid: str, must_exist_in_registry: bool = True) -> Path:
        check_id(fid, "family id")
        check_id(vid, "variant id")
        if must_exist_in_registry:
            fam = self.reg.families.get(fid)
            if not fam:
                raise AdminError(404, f"family '{fid}' is not loaded")
            if not fam.variant(vid):
                raise AdminError(404, f"variant '{vid}' is not defined in '{fid}'")
        return self.root / "refs" / fid / vid

    def list_refs(self, fid: str | None = None) -> list[dict]:
        out = []
        for fam in self.reg.families.values():
            if fid and fam.family != fid:
                continue
            for v in fam.variants:
                for p in ref_files(self.root, fam.family, v.id):
                    st = p.stat()
                    out.append({"id": f"{fam.family}/{v.id}/{p.name}", "family": fam.family, "variant": v.id,
                                "file": p.name, "size": st.st_size,
                                "added_at": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()})
        return out

    def ref_path(self, fid: str, vid: str, name: str) -> Path:
        if not FILE_RE.match(name) or Path(name).suffix.lower() not in REF_EXTS:
            raise AdminError(400, "invalid file name")
        p = self._variant_dir(fid, vid, must_exist_in_registry=False) / name
        if not p.is_file():
            raise AdminError(404, "reference image not found")
        return p

    def add_ref(self, fid: str, vid: str, img: np.ndarray, warp: bool = True, prefix: str = "ref") -> dict:
        """Store a reference. By default the card is located and warped first, so the reference
        looks exactly like what the pipeline compares against (a 1012×638 upright card)."""
        d = self._variant_dir(fid, vid)
        detected = None
        if warp:
            img, info = normalise.normalise(img)
            detected = info["card_detected"]
        else:
            img = cv2.resize(img, (CARD_W, CARD_H))
        with self._lock:
            d.mkdir(parents=True, exist_ok=True)
            out = d / f"{prefix}_{int(time.time() * 1000)}.png"
            cv2.imwrite(str(out), img)
            self.reg.reload()
        return {"id": f"{fid}/{vid}/{out.name}", "family": fid, "variant": vid, "file": out.name,
                "card_detected": detected, "registry_version": self.reg.version}

    def delete_ref(self, fid: str, vid: str, name: str) -> dict:
        p = self.ref_path(fid, vid, name)
        with self._lock:
            trash = self.root / ".trash" / _ts()
            trash.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(trash / f"{fid}__{vid}__{name}"))
            self.reg.reload()
        return {"ok": True, "registry_version": self.reg.version}

    # ---- diagnostics -----------------------------------------------------------------------------
    def separability(self) -> dict:
        """Mean visual similarity between every pair of variant reference groups.
        High off-diagonal cells are look-alikes that must be separated by text/colour signals."""
        if self._sep_cache and self._sep_cache[0] == self.reg.version:
            return self._sep_cache[1]
        groups = [(f"{f.family}/{v.id}", v.refs) for f in self.reg.families.values() for v in f.variants if v.refs]
        matrix = []
        for _, a in groups:
            row = []
            for _, b in groups:
                sims = [visual_similarity(x, y) for x in a for y in b if x is not y]
                row.append(round(float(np.mean(sims)), 3) if sims else 1.0)
            matrix.append(row)
        res = {"registry_version": self.reg.version, "groups": [g for g, _ in groups], "matrix": matrix}
        self._sep_cache = (self.reg.version, res)
        return res

    def nearest(self, fid: str, vid: str, name: str, k: int = 6) -> list[dict]:
        p = self.ref_path(fid, vid, name)
        img = cv2.imread(str(p))
        if img is None:
            raise AdminError(415, "unreadable image")
        sig = visual_signature(cv2.resize(img, (CARD_W, CARD_H)))
        me = f"{fid}/{vid}/{name}"
        scored = [{"id": r["id"], "family": f.family, "variant": v.id, "similarity": round(visual_similarity(sig, r), 3)}
                  for f in self.reg.families.values() for v in f.variants for r in v.refs if r.get("id") != me]
        return sorted(scored, key=lambda x: -x["similarity"])[:k]
