"""Run store: every validation, its stage trace, its images and the human review outcome.

SQLite (WAL) under the data directory — no extra service to run. Images are kept as files next to
the database so a reviewer can see what the agent saw. Retention is enforced on start-up and then
hourly (UAE PDPL: keep card images only as long as needed). Put the data directory on an encrypted
volume in production.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from .pipeline import PipelineRun

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id                 TEXT PRIMARY KEY,
  request_id         TEXT NOT NULL,
  created_at         TEXT NOT NULL,
  source             TEXT NOT NULL,
  label              TEXT,
  status             TEXT NOT NULL,
  family             TEXT,
  variant            TEXT,
  family_confidence  REAL,
  variant_confidence REAL,
  processing_ms      INTEGER,
  reasons            TEXT NOT NULL,
  exit_stage         TEXT,
  registry_version   TEXT,
  result             TEXT NOT NULL,
  trace              TEXT NOT NULL,
  has_image          INTEGER NOT NULL DEFAULT 0,
  has_card           INTEGER NOT NULL DEFAULT 0,
  review_decision    TEXT,
  review_note        TEXT,
  reviewer           TEXT,
  reviewed_at        TEXT,
  enrolled           TEXT
);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS runs_request ON runs(request_id);
"""

SUMMARY_COLS = ("id, request_id, created_at, source, label, status, family, variant, family_confidence, "
                "variant_confidence, processing_ms, reasons, exit_stage, registry_version, has_image, has_card, "
                "review_decision, review_note, reviewer, reviewed_at, enrolled")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def exit_stage(status: str, trace: list[dict]) -> str:
    """The stage where the decision was effectively made — drives the 'where decisions are made' chart."""
    if status == "ACCEPT":
        return "accepted"
    for s in trace:
        if s["id"] != "decision" and s["status"] in ("fail", "warn"):
            return s["id"]
    return "decision"


def quantile(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


class RunStore:
    def __init__(self, data_dir: Path, store_images: bool = True, retention_days: int | None = 90):
        self.dir = data_dir
        self.img_dir = data_dir / "runs"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.store_images = store_images
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._db = sqlite3.connect(data_dir / "runs.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()
        self._last_purge = 0.0
        self.purge()

    # ---- write -------------------------------------------------------------------------------
    def save(self, run: PipelineRun, source: str = "api", label: str | None = None) -> dict:
        res = run.result
        rid = uuid.uuid4().hex[:16]
        has_image = has_card = 0
        if self.store_images:
            if run.image is not None:
                has_image = int(cv2.imwrite(str(self.img_dir / f"{rid}.jpg"), run.image,
                                            [cv2.IMWRITE_JPEG_QUALITY, 90]))
            if run.card is not None:
                has_card = int(cv2.imwrite(str(self.img_dir / f"{rid}_card.png"), run.card))
        card = res.get("card") or {}
        trace = [{k: v for k, v in s.items()} for s in run.trace]
        row = {
            "id": rid, "request_id": res["request_id"], "created_at": iso(utcnow()), "source": source,
            "label": label, "status": res["status"], "family": card.get("family"), "variant": card.get("variant"),
            "family_confidence": card.get("family_confidence"), "variant_confidence": card.get("variant_confidence"),
            "processing_ms": res.get("processing_ms"), "reasons": json.dumps(res["reasons"]),
            "exit_stage": exit_stage(res["status"], trace), "registry_version": res["model_versions"]["registry"],
            "result": json.dumps(res, ensure_ascii=False), "trace": json.dumps(trace, ensure_ascii=False, default=str),
            "has_image": has_image, "has_card": has_card,
        }
        with self._lock:
            self._db.execute(f"INSERT INTO runs ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                             tuple(row.values()))
            self._db.commit()
        if time.time() - self._last_purge > 3600:
            self.purge()
        return self.get_summary(rid)

    def review(self, rid: str, decision: str, note: str | None, reviewer: str | None) -> dict | None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE runs SET review_decision=?, review_note=?, reviewer=?, reviewed_at=? WHERE id=?",
                (decision, note, reviewer, iso(utcnow()), rid))
            self._db.commit()
        return self.get_summary(rid) if cur.rowcount else None

    def clear_review(self, rid: str) -> dict | None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE runs SET review_decision=NULL, review_note=NULL, reviewer=NULL, reviewed_at=NULL WHERE id=?",
                (rid,))
            self._db.commit()
        return self.get_summary(rid) if cur.rowcount else None

    def mark_enrolled(self, rid: str, ref_id: str) -> None:
        with self._lock:
            row = self._db.execute("SELECT enrolled FROM runs WHERE id=?", (rid,)).fetchone()
            lst = json.loads(row["enrolled"]) if row and row["enrolled"] else []
            lst.append(ref_id)
            self._db.execute("UPDATE runs SET enrolled=? WHERE id=?", (json.dumps(lst), rid))
            self._db.commit()

    def delete(self, rid: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM runs WHERE id=?", (rid,))
            self._db.commit()
        for p in (self.img_dir / f"{rid}.jpg", self.img_dir / f"{rid}_card.png"):
            p.unlink(missing_ok=True)
        return bool(cur.rowcount)

    def purge(self) -> int:
        self._last_purge = time.time()
        if not self.retention_days:
            return 0
        cutoff = iso(utcnow() - timedelta(days=self.retention_days))
        with self._lock:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM runs WHERE created_at < ?", (cutoff,))]
            self._db.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,))
            self._db.commit()
        for rid in ids:
            for p in (self.img_dir / f"{rid}.jpg", self.img_dir / f"{rid}_card.png"):
                p.unlink(missing_ok=True)
        return len(ids)

    # ---- read --------------------------------------------------------------------------------
    @staticmethod
    def _summary(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["reasons"] = json.loads(d["reasons"])
        d["enrolled"] = json.loads(d["enrolled"]) if d.get("enrolled") else []
        d["has_image"], d["has_card"] = bool(d["has_image"]), bool(d["has_card"])
        d["review"] = ({"decision": d.pop("review_decision"), "note": d.pop("review_note"),
                        "reviewer": d.pop("reviewer"), "at": d.pop("reviewed_at")}
                       if d.get("review_decision") else None)
        for k in ("review_decision", "review_note", "reviewer", "reviewed_at"):
            d.pop(k, None)
        return d

    def get_summary(self, rid: str) -> dict | None:
        with self._lock:
            r = self._db.execute(f"SELECT {SUMMARY_COLS} FROM runs WHERE id=?", (rid,)).fetchone()
        return self._summary(r) if r else None

    def get(self, rid: str) -> dict | None:
        with self._lock:
            r = self._db.execute(f"SELECT {SUMMARY_COLS}, result, trace FROM runs WHERE id=?", (rid,)).fetchone()
        if not r:
            return None
        d = self._summary(r)
        d["result"], d["trace"] = json.loads(d["result"]), json.loads(d["trace"])
        return d

    def image_path(self, rid: str, kind: str) -> Path | None:
        p = self.img_dir / (f"{rid}_card.png" if kind == "card" else f"{rid}.jpg")
        return p if p.is_file() else None

    def list(self, *, status: str | None = None, family: str | None = None, queue: bool = False,
             reviewed: bool | None = None, q: str | None = None, since: str | None = None,
             limit: int = 50, offset: int = 0) -> dict:
        where, args = [], []
        if queue:
            where.append("status='REVIEW' AND review_decision IS NULL")
        if status:
            where.append("status=?")
            args.append(status)
        if family == "unknown":
            where.append("family IS NULL")
        elif family:
            where.append("family=?")
            args.append(family)
        if reviewed is True:
            where.append("review_decision IS NOT NULL")
        elif reviewed is False:
            where.append("review_decision IS NULL")
        if since:
            where.append("created_at >= ?")
            args.append(since)
        if q:
            where.append("(id LIKE ? OR request_id LIKE ? OR IFNULL(label,'') LIKE ? OR reasons LIKE ?)")
            args += [f"%{q}%"] * 4
        w = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM runs {w}", args).fetchone()[0]
            rows = self._db.execute(f"SELECT {SUMMARY_COLS} FROM runs {w} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                                    [*args, limit, offset]).fetchall()
            queue_n = self._db.execute(
                "SELECT COUNT(*) FROM runs WHERE status='REVIEW' AND review_decision IS NULL").fetchone()[0]
        return {"items": [self._summary(r) for r in rows], "total": total, "limit": limit, "offset": offset,
                "review_queue": queue_n}

    def stats(self, hours: int | None, tz_minutes: int = 0) -> dict:
        """Aggregates for the dashboard. Time buckets are in the viewer's timezone (`tz_minutes` east of UTC)."""
        utc_now = utcnow()
        since = iso(utc_now - timedelta(hours=hours)) if hours else None
        shift = timedelta(minutes=tz_minutes)
        now = utc_now + shift
        with self._lock:
            rows = self._db.execute(
                "SELECT created_at, status, family, processing_ms, reasons, exit_stage, review_decision FROM runs"
                + (" WHERE created_at >= ?" if since else ""), (since,) if since else ()).fetchall()
            queue_n = self._db.execute(
                "SELECT COUNT(*) FROM runs WHERE status='REVIEW' AND review_decision IS NULL").fetchone()[0]
            first = self._db.execute("SELECT MIN(created_at) FROM runs").fetchone()[0]

        decs = ("ACCEPT", "REVIEW", "REJECT")
        by_status = {d: 0 for d in decs}
        fams: dict[str, dict] = {}
        reasons: dict[str, int] = {}
        exits: dict[str, int] = {}
        lat = []
        # agent decision x reviewer decision — the reviewer's call is the ground truth
        confusion = {a: {"ACCEPT": 0, "REJECT": 0} for a in decs}

        # time buckets: hourly up to 48 h, daily beyond
        hourly = bool(hours and hours <= 48)
        if hours:
            span = hours if hourly else max(1, hours // 24)
        else:
            start = datetime.fromisoformat(first.replace("Z", "+00:00")) + shift if first else now
            span = min(366, max(1, (now.date() - start.date()).days + 1))
        step = timedelta(hours=1) if hourly else timedelta(days=1)
        anchor = now.replace(minute=0, second=0, microsecond=0) if hourly else now.replace(
            hour=0, minute=0, second=0, microsecond=0)
        keys = [iso(anchor - step * i)[: 13 if hourly else 10] for i in range(span - 1, -1, -1)]
        series = {k: {"t": k, **{d: 0 for d in decs}} for k in keys}

        for r in rows:
            st = r["status"]
            by_status[st] = by_status.get(st, 0) + 1
            f = fams.setdefault(r["family"] or "unknown", {"family": r["family"] or "unknown", **{d: 0 for d in decs}})
            f[st] += 1
            for x in json.loads(r["reasons"]):
                k = x.split(":")[0]
                reasons[k] = reasons.get(k, 0) + 1
            exits[r["exit_stage"] or "decision"] = exits.get(r["exit_stage"] or "decision", 0) + 1
            if r["processing_ms"] is not None:
                lat.append(r["processing_ms"])
            if r["review_decision"] in ("ACCEPT", "REJECT"):
                confusion[st][r["review_decision"]] += 1
            local = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) + shift
            k = iso(local)[: 13 if hourly else 10]
            if k in series:
                series[k][st] += 1

        n = len(rows)
        decided = [(a, b) for a in ("ACCEPT", "REJECT") for b in ("ACCEPT", "REJECT")]
        audited = sum(confusion[a][b] for a, b in decided)
        agree = confusion["ACCEPT"]["ACCEPT"] + confusion["REJECT"]["REJECT"]
        reviewed_review = confusion["REVIEW"]["ACCEPT"] + confusion["REVIEW"]["REJECT"]
        return {
            "range_hours": hours, "total": n, "by_status": by_status,
            "auto_accept_rate": by_status["ACCEPT"] / n if n else 0.0,
            "latency_ms": {"p50": quantile(lat, 0.5), "p95": quantile(lat, 0.95), "max": max(lat) if lat else 0},
            "series": {"bucket": "hour" if hourly else "day", "points": list(series.values())},
            "by_family": sorted(fams.values(), key=lambda x: -(x["ACCEPT"] + x["REVIEW"] + x["REJECT"])),
            "top_reasons": sorted(({"reason": k, "count": v} for k, v in reasons.items()), key=lambda x: -x["count"])[:10],
            "exit_stages": exits,
            "review": {"queue": queue_n, "reviewed_review_cases": reviewed_review,
                       "audited_decisions": audited, "agreement": agree / audited if audited else None,
                       "wrong_accepts": confusion["ACCEPT"]["REJECT"], "wrong_rejects": confusion["REJECT"]["ACCEPT"],
                       "confusion": confusion},
        }

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def wipe_images(self) -> None:  # used by tests
        shutil.rmtree(self.img_dir, ignore_errors=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
