"""Runtime-tunable decision thresholds.

The business trades automation rate against risk by moving these numbers (ARCHITECTURE §3 [8]).
They used to be module constants; now they live in one `Thresholds` object that is persisted to
`<data_dir>/settings.json`, editable through `GET/PUT /v1/settings`, and passed explicitly into the
pipeline so an evaluation can run with a what-if copy without touching live traffic.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    # [1] quality gate
    min_side: int = 400
    min_blur: float = 60.0
    min_brightness: float = 0.15
    max_brightness: float = 0.92
    max_glare: float = 0.08
    # [4] family: decided only if top1 >= family_min AND top1 - top2 >= family_margin
    family_min: float = 0.50
    family_margin: float = 0.15
    # below this family score the card is rejected as unknown, otherwise it goes to review
    unknown_reject_below: float = 0.30
    # [5] variant
    variant_min: float = 0.45
    variant_margin: float = 0.10
    # [8] extracted fields below this OCR confidence send the card to review
    field_min_conf: float = 0.55

    def validate(self) -> list[str]:
        errs = []
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "min_side":
                if not 16 <= v <= 4000:
                    errs.append("min_side must be between 16 and 4000")
            elif f.name == "min_blur":
                if not 0 <= v <= 5000:
                    errs.append("min_blur must be between 0 and 5000")
            elif not 0 <= v <= 1:
                errs.append(f"{f.name} must be between 0 and 1")
        if self.min_brightness >= self.max_brightness:
            errs.append("min_brightness must be below max_brightness")
        return errs

    def merged(self, patch: dict) -> "Thresholds":
        known = {f.name: f.type for f in fields(self)}
        unknown = sorted(set(patch) - set(known))
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
        cast = {k: (int(v) if k == "min_side" else float(v)) for k, v in patch.items()}
        return replace(self, **cast)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULTS = Thresholds()


class SettingsStore:
    """Thread-safe holder of the live thresholds, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._current = DEFAULTS
        if path.exists():
            try:
                self._current = DEFAULTS.merged(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError, json.JSONDecodeError):
                # a corrupt file must not stop the service; keep defaults and leave the file for inspection
                self._current = DEFAULTS

    @property
    def current(self) -> Thresholds:
        return self._current

    def update(self, patch: dict) -> Thresholds:
        nxt = self._current.merged(patch)
        errs = nxt.validate()
        if errs:
            raise ValueError("; ".join(errs))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(nxt.as_dict(), indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            self._current = nxt
        return nxt

    def reset(self) -> Thresholds:
        with self._lock:
            self.path.unlink(missing_ok=True)
            self._current = DEFAULTS
        return DEFAULTS
