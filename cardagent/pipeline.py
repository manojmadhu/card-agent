"""Orchestrator: runs the agent stages and builds the JSON output contract.

`CardAgent.start()` returns a `PipelineRun` whose `events()` generator yields one event when each
stage starts and one when it ends, with the stage's evidence (quality checks, OCR tokens, fusion
table, variant scores, extracted fields, rule results). The REST layer streams those events to the
console as NDJSON so an operator can watch a real validation step by step. `CardAgent.process()`
drains the same generator and returns only the result envelope, so the contract for the capture
system is unchanged.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from datetime import date

import cv2
import numpy as np

from . import classify, extract, normalise, ocr, quality, validate, vlm
from .config import DEFAULTS, Thresholds
from .registry import Registry

PIPELINE_VERSION = "0.2.0"
SCHEMA_VERSION = "1.0"

STAGES: list[tuple[str, str, str]] = [
    ("ingest", "Ingest", "0"),
    ("quality", "Quality gate", "1"),
    ("normalise", "Normalise", "2"),
    ("ocr", "Full-card OCR", "3"),
    ("family", "Family classification", "4"),
    ("variant", "Variant classification", "5"),
    ("fields", "Field extraction", "6"),
    ("rules", "Validation rules", "7"),
    ("decision", "Decision", "8"),
]

PII_FIELDS = ("card_number", "emirates_id")


def mask_value(name: str, val):
    if val and name in PII_FIELDS and len(val) > 4:
        return "*" * (len(val) - 4) + val[-4:]
    return val


class PipelineRun:
    """One validation. Iterate `events()`; afterwards `result`, `trace`, `image` and `card` are set."""

    def __init__(self, agent: "CardAgent", image_bytes: bytes, request_id: str | None,
                 today: date | None, t: Thresholds):
        self.agent, self.reg, self.t = agent, agent.reg, t
        self.image_bytes, self.today = image_bytes, today
        self.request_id = request_id or str(uuid.uuid4())
        self.trace: list[dict] = [{"id": i, "title": ti, "stage": st, "status": "pending", "summary": "",
                                   "ms": 0, "detail": None} for i, ti, st in STAGES]
        self.result: dict | None = None
        self.image: np.ndarray | None = None   # decoded upload
        self.card: np.ndarray | None = None    # normalised 1012x638 card, upright
        self._t_stage = 0.0

    # ---- trace helpers -------------------------------------------------------------------------
    def _step(self, sid: str) -> dict:
        return next(s for s in self.trace if s["id"] == sid)

    def _begin(self, sid: str) -> dict:
        s = self._step(sid)
        s["status"] = "running"
        self._t_stage = time.perf_counter()
        return {"event": "stage", **s}

    def _end(self, sid: str, status: str, summary: str, detail=None) -> dict:
        s = self._step(sid)
        s.update(status=status, summary=summary, detail=detail,
                 ms=max(1, int((time.perf_counter() - self._t_stage) * 1000)))
        return {"event": "stage", **s}

    def _skip_rest(self, why: str) -> list[dict]:
        out = []
        for s in self.trace:
            if s["status"] == "pending" and s["id"] != "decision":
                s.update(status="skip", summary=why)
                out.append({"event": "stage", **s})
        return out

    # ---- the pipeline --------------------------------------------------------------------------
    def events(self) -> Iterator[dict]:
        t0 = time.perf_counter()
        t = self.t
        out = {"schema_version": SCHEMA_VERSION, "request_id": self.request_id, "status": None,
               "card": None, "fields": {}, "validation": [], "reasons": [],
               "quality": None, "candidates": {},
               "model_versions": {"pipeline": PIPELINE_VERSION, "registry": self.reg.version,
                                  "ocr": ocr.engine_version()}}
        status, reasons = None, []

        # [0] ingest
        yield self._begin("ingest")
        img = cv2.imdecode(np.frombuffer(self.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            yield self._end("ingest", "fail", "not a decodable image → INVALID_IMAGE",
                            {"request_id": self.request_id, "bytes": len(self.image_bytes)})
            status, reasons = "REJECT", ["INVALID_IMAGE"]
            yield from self._skip_rest("skipped — image could not be decoded")
        else:
            self.image = img
            h, w = img.shape[:2]
            yield self._end("ingest", "pass", f"{w}×{h} · {len(self.image_bytes) // 1024} KB",
                            {"request_id": self.request_id, "width": w, "height": h, "bytes": len(self.image_bytes)})

        # [1] quality gate
        if status is None:
            yield self._begin("quality")
            q = quality.assess(img, t)
            out["quality"] = {k: v for k, v in q.items() if k != "checks"}
            if not q["passed"]:
                yield self._end("quality", "fail", f"{', '.join(q['issues'])} → RETAKE_REQUIRED", q)
                status, reasons = "REJECT", ["RETAKE_REQUIRED"] + q["issues"]
                yield from self._skip_rest("skipped — quality gate failed")
            else:
                yield self._end("quality", "pass",
                                f"blur {q['blur']} · brightness {q['brightness']} · glare {q['glare_ratio'] * 100:.1f}%", q)

        # [2] normalise + [3] OCR with orientation search
        full = None
        if status is None:
            yield self._begin("normalise")
            card, ninfo = normalise.normalise(img)
            yield self._end("normalise", "pass" if ninfo["card_detected"] else "warn",
                            f"card found by {ninfo['detector']} → warped to 1012×638" if ninfo["card_detected"]
                            else "no card outline found — using the whole frame",
                            {"size": [card.shape[1], card.shape[0]], **ninfo})

            yield self._begin("ocr")
            card, full, rotation, tried = self.agent.ocr_best_orientation(card)
            self.card = card
            out["quality"].update(card_detected=ninfo["card_detected"], detector=ninfo["detector"],
                                  rotation=rotation, ocr_mean_conf=round(full["mean_conf"], 3))
            yield self._end("ocr", "pass" if full["tokens"] else "warn",
                            f"{len(full['tokens'])} tokens · mean conf {full['mean_conf']:.2f}"
                            + (f" · card was upside down (rotated {rotation}°)" if rotation else "")
                            if full["tokens"] else "no text read — the text signal will be 0",
                            {"engine": ocr.engine_version(), "languages": ocr.full_lang(), "rotation": rotation,
                             "orientations": tried,
                             "tokens": [{"text": tx, "conf": c} for tx, c in zip(full["tokens"], full["confs"])],
                             "mean_conf": round(full["mean_conf"], 3)})

        # [4] family
        fam = fam_res = None
        if status is None:
            yield self._begin("family")
            fam_res = classify.classify_family(self.reg, card, full, t)
            out["candidates"]["family"] = [{"family": c["family"], "score": c["score"], "signals": c["signals"]}
                                           for c in fam_res["candidates"]]
            detail = {"candidates": fam_res["candidates"], "tau": t.family_min, "delta": t.family_margin,
                      "margin": fam_res["margin"], "body_rgb": list(fam_res["body_rgb"]),
                      "unknown_reject_below": t.unknown_reject_below}
            top = fam_res["candidates"][0] if fam_res["candidates"] else None
            if not fam_res["decided"]:
                unknown = fam_res["confidence"] < t.family_min
                reason = "UNKNOWN_CARD_TYPE" if unknown else "AMBIGUOUS_CARD_TYPE"
                status = "REJECT" if fam_res["confidence"] < t.unknown_reject_below else "REVIEW"
                reasons = [reason]
                detail["outcome"] = "unknown" if unknown else "ambiguous"
                yield self._end("family", "fail" if status == "REJECT" else "warn",
                                (f"top-1 {top['name']} {top['score']:.2f} < τ {t.family_min} → unknown card"
                                 if unknown else
                                 f"{top['name']} margin {fam_res['margin']:.2f} < δ {t.family_margin} → ambiguous")
                                if top else "no card families are loaded", detail)
                yield from self._skip_rest("skipped — card family not decided")
            else:
                fam = self.reg.families[fam_res["family"]]
                detail["outcome"] = "decided"
                yield self._end("family", "pass",
                                f"{fam.family_name} · score {fam_res['confidence']:.2f} · margin {fam_res['margin']:.2f}",
                                detail)

        if status is None:
            # [5] variant (only among this family's variants)
            yield self._begin("variant")
            var_res = classify.classify_variant(fam, card, full, fam_res["signature"], fam_res["body_rgb"], t)
            out["candidates"]["variant"] = [{"variant": c["variant"], "score": c["score"], "signals": c["signals"]}
                                            for c in var_res["candidates"]]
            v = fam.variant(var_res["variant"]) if var_res["decided"] else None
            out["card"] = {"family": fam.family, "family_name": fam.family_name, "issuer": fam.issuer,
                           "family_confidence": fam_res["confidence"],
                           "variant": v.id if v else None, "variant_name": v.name if v else None,
                           "variant_confidence": var_res["confidence"]}
            vtop = var_res["candidates"][0]
            yield self._end("variant", "pass" if v else "warn",
                            f"{v.name} · score {var_res['confidence']:.2f} · margin {var_res['margin']:.2f}" if v
                            else f"{vtop['name']} {vtop['score']:.2f} — below τ {t.variant_min} or margin "
                                 f"{var_res['margin']:.2f} < δ {t.variant_margin}",
                            {"candidates": var_res["candidates"], "tau": t.variant_min, "delta": t.variant_margin,
                             "margin": var_res["margin"], "number_seen": mask_value("card_number", var_res["number_seen"]),
                             "weights": classify.VARIANT_WEIGHTS})

            # [6] fields
            yield self._begin("fields")
            fields = extract.extract_fields(fam, card, full)
            ok = [n for n, f in fields.items() if f["valid_format"]]
            missing = [n for n, f in fields.items() if f["required"] and not f["valid_format"]]
            yield self._end("fields", "warn" if missing else "pass",
                            f"{len(ok)}/{len(fields)} extracted" + (f" · unreadable: {', '.join(missing)}" if missing else ""),
                            {"fields": [{"name": n, **f} for n, f in fields.items()],
                             "vlm": "enabled" if vlm.enabled() else "disabled", "min_conf": t.field_min_conf})

            # [7] rules
            yield self._begin("rules")
            rules = validate.run_rules(fam, v.id if v else None, fields, self.today)
            n_pass = sum(r["result"] == "pass" for r in rules)
            crit_fail = [r for r in rules if r["critical"] and r["result"] == "fail"]
            soft_fail = [r for r in rules if not r["critical"] and r["result"] == "fail"]
            yield self._end("rules", "fail" if crit_fail else "warn" if soft_fail else "pass",
                            f"{n_pass} pass · {len(crit_fail) + len(soft_fail)} fail · "
                            f"{len(rules) - n_pass - len(crit_fail) - len(soft_fail)} skip",
                            {"results": rules})

            # [8] decision
            if not v:
                reasons.append("AMBIGUOUS_VARIANT")
            for name, f in fields.items():
                if f["required"] and not f["valid_format"]:
                    reasons.append(f"FIELD_UNREADABLE:{name}")
                elif f["valid_format"] and (f["confidence"] < t.field_min_conf or f["source"] == "vlm"):
                    reasons.append(f"LOW_CONFIDENCE:{name}")
            expired = any(r["rule"] == "not_expired" and r["result"] == "fail" and r["detail"]
                          and r["detail"].startswith("CARD_EXPIRED") for r in rules)
            reasons += [f"RULE_FAILED:{r['rule']}:{r['field']}" for r in crit_fail + soft_fail]
            if expired:
                status = "REJECT"
                reasons.insert(0, "CARD_EXPIRED")
            elif crit_fail or reasons:
                status = "REVIEW"
            else:
                status = "ACCEPT"

            out["fields"] = {k: {"value": f["value"], "confidence": f["confidence"], "source": f["source"],
                                 "valid_format": f["valid_format"]} for k, f in fields.items()}
            exp = out["fields"].get("expiry_date")
            if exp and exp["valid_format"]:
                d = validate.parse_expiry(exp["value"])
                exp["valid_until"] = d.isoformat() if d else None  # ISO date for the storage system
            out["validation"] = rules

        yield self._begin("decision")
        out["status"], out["reasons"] = status, reasons
        out["processing_ms"] = int((time.perf_counter() - t0) * 1000)
        self.result = out
        yield self._end("decision", {"ACCEPT": "pass", "REVIEW": "warn"}.get(status, "fail"),
                        status + (f" · {', '.join(reasons)}" if reasons else ""),
                        {"status": status, "reasons": reasons})


class CardAgent:
    def __init__(self, registry: Registry):
        self.reg = registry

    def ocr_best_orientation(self, card):
        """Try upright then 180° (card held upside down); keep the one with best keyword evidence."""
        best, tried = None, []
        for rot in (None, cv2.ROTATE_180):
            img = card if rot is None else cv2.rotate(card, rot)
            o = ocr.read_full(img)
            kw = max((classify.keyword_score(o["tokens"], o["text"], f.keywords)
                      for f in self.reg.families.values()), default=0.0)
            score = kw + o["mean_conf"] * 0.1
            deg = 0 if rot is None else 180
            tried.append({"rotation": deg, "keyword_score": round(kw, 3), "mean_conf": round(o["mean_conf"], 3)})
            if best is None or score > best[0]:
                best = (score, img, o, deg)
            if kw >= 0.99:
                break
        return best[1], best[2], best[3], tried

    def start(self, image_bytes: bytes, request_id: str | None = None, today: date | None = None,
              thresholds: Thresholds = DEFAULTS) -> PipelineRun:
        return PipelineRun(self, image_bytes, request_id, today, thresholds)

    def process(self, image_bytes: bytes, request_id: str | None = None, today: date | None = None,
                thresholds: Thresholds = DEFAULTS) -> dict:
        run = self.start(image_bytes, request_id, today, thresholds)
        for _ in run.events():
            pass
        return run.result

    @staticmethod
    def masked_log_view(result: dict) -> dict:
        """PII-safe copy for logs."""
        r = dict(result)
        r["fields"] = {k: {**v, "value": mask_value(k, v["value"])} for k, v in result.get("fields", {}).items()}
        return r
