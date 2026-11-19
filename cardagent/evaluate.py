"""Labelled-set evaluation (the service-side twin of tests/run_eval.py).

A labelled set is a directory with images plus `manifest.json`:
    [{"file": "x.jpg", "family": "fazaa"|null, "variant": "gold"|null, "expected_status": "ACCEPT", ...}]
Evaluations never write to the run store, and can run with what-if thresholds without touching live traffic.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from .config import Thresholds
from .pipeline import CardAgent


def load_manifest(eval_dir: Path) -> list[dict]:
    p = eval_dir / "manifest.json"
    if not p.is_file():
        return []
    cases = json.loads(p.read_text(encoding="utf-8"))
    return [c for c in cases if isinstance(c, dict) and "file" in c and (eval_dir / c["file"]).is_file()]


def case_passed(case: dict, res: dict) -> bool:
    """Same scoring as tests/run_eval.py: status must match; family/variant must match when the
    agent didn't reject (a REJECT carries no identity)."""
    card = res.get("card") or {}
    if res["status"] != case.get("expected_status"):
        return False
    if res["status"] == "REJECT":
        return True
    if case.get("family") and card.get("family") != case["family"]:
        return False
    if case.get("variant") and card.get("variant") and card.get("variant") != case["variant"]:
        return False
    return True


def iter_eval(agent: CardAgent, eval_dir: Path, thresholds: Thresholds, today: date | None) -> Iterator[dict]:
    cases = load_manifest(eval_dir)
    yield {"event": "start", "total": len(cases)}
    rows = []
    for i, c in enumerate(cases):
        res = agent.process((eval_dir / c["file"]).read_bytes(), request_id=f"eval:{c['file']}",
                            today=today, thresholds=thresholds)
        card = res.get("card") or {}
        num = (res.get("fields", {}).get("card_number") or {}).get("value")
        row = {"file": c["file"], "expected": c.get("expected_status"), "got": res["status"],
               "family_expected": c.get("family"), "family_got": card.get("family"),
               "variant_expected": c.get("variant"), "variant_got": card.get("variant"),
               "number_ok": (num == c["card_number"]) if c.get("card_number") and res.get("fields") else None,
               "reasons": res["reasons"], "processing_ms": res["processing_ms"],
               "passed": case_passed(c, res)}
        rows.append(row)
        yield {"event": "case", "index": i, **row}
    n = len(rows)
    yield {"event": "summary", "total": n, "passed": sum(r["passed"] for r in rows),
           "wrong_accepts": sum(r["got"] == "ACCEPT" and r["expected"] != "ACCEPT" for r in rows),
           "auto_accept": sum(r["got"] == "ACCEPT" for r in rows),
           "thresholds": thresholds.as_dict(), "rows": rows}
