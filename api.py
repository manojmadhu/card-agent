"""REST API for the capture system and the Card Agent Console.

Run:  uvicorn api:app --host 0.0.0.0 --port 8080

Configuration (environment):
  CARD_AGENT_REGISTRY_DIR    card-family YAML + reference images        (default ./registry)
  CARD_AGENT_DATA_DIR        run store, images, settings.json          (default ./data)
  CARD_AGENT_EVAL_DIR        labelled set with manifest.json           (default ./samples)
  CARD_AGENT_API_KEY         operator key: validate, runs, review       (unset = no auth, dev only)
  CARD_AGENT_ADMIN_KEY       admin key: registry + threshold changes    (unset = API key is admin)
  CARD_AGENT_CORS_ORIGINS    comma-separated console origins            (default http://localhost:5173)
  CARD_AGENT_STORE_IMAGES    keep card images for review (true/false)   (default true)
  CARD_AGENT_RETENTION_DAYS  delete runs + images older than this       (default 90, 0 = keep)
  CARD_AGENT_CONSOLE_DIR     serve the built console (dist/) at /        (optional)
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cardagent import ocr, vlm
from cardagent.config import DEFAULTS, SettingsStore
from cardagent.evaluate import iter_eval, load_manifest
from cardagent.pipeline import PIPELINE_VERSION, SCHEMA_VERSION, STAGES, CardAgent
from cardagent.registry import Registry
from cardagent.registry_admin import AdminError, RegistryAdmin, check_id, decode_image
from cardagent.store import RunStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("card-agent")

HERE = Path(__file__).parent
API_VERSION = "0.2.0"


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    registry_dir: Path = field(default_factory=lambda: Path(os.environ.get("CARD_AGENT_REGISTRY_DIR", HERE / "registry")))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CARD_AGENT_DATA_DIR", HERE / "data")))
    eval_dir: Path = field(default_factory=lambda: Path(os.environ.get("CARD_AGENT_EVAL_DIR", HERE / "samples")))
    api_key: str | None = field(default_factory=lambda: os.environ.get("CARD_AGENT_API_KEY") or None)
    admin_key: str | None = field(default_factory=lambda: os.environ.get("CARD_AGENT_ADMIN_KEY") or None)
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("CARD_AGENT_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()])
    store_images: bool = field(default_factory=lambda: _env_bool("CARD_AGENT_STORE_IMAGES", True))
    retention_days: int = field(default_factory=lambda: int(os.environ.get("CARD_AGENT_RETENTION_DAYS", "90")))
    console_dir: Path | None = field(default_factory=lambda: Path(p) if (p := os.environ.get("CARD_AGENT_CONSOLE_DIR")) else None)


# ---- request bodies ------------------------------------------------------------------------------
class ReviewIn(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]
    note: str | None = Field(None, max_length=2000)
    reviewer: str | None = Field(None, max_length=200)


class EnrollIn(BaseModel):
    family: str
    variant: str


class EvalIn(BaseModel):
    thresholds: dict | None = None
    today: date | None = None


def _ndjson(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _image_response(path: Path, width: int | None) -> Response:
    if not width:
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(415, "unreadable image")
    h, w = img.shape[:2]
    width = max(32, min(width, w))
    small = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    registry = Registry(cfg.registry_dir)
    agent = CardAgent(registry)
    admin = RegistryAdmin(registry)
    settings = SettingsStore(cfg.data_dir / "settings.json")
    store = RunStore(cfg.data_dir, store_images=cfg.store_images, retention_days=cfg.retention_days or None)

    keys: dict[str, str] = {}
    if cfg.api_key:
        keys[cfg.api_key] = "operator" if cfg.admin_key else "admin"
    if cfg.admin_key:
        keys[cfg.admin_key] = "admin"
    if not keys:
        log.warning("CARD_AGENT_API_KEY is not set — the API is open. Set it (and CARD_AGENT_ADMIN_KEY) in production.")

    def require(role: Literal["operator", "admin"]):
        def dep(request: Request) -> str:
            if not keys:
                return "admin"
            presented = request.headers.get("x-api-key") or ""
            auth = request.headers.get("authorization") or ""
            if not presented and auth.lower().startswith("bearer "):
                presented = auth[7:].strip()
            got = next((r for k, r in keys.items() if presented and secrets.compare_digest(presented, k)), None)
            if got is None:
                raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
            if role == "admin" and got != "admin":
                raise HTTPException(403, "this action needs the admin key")
            return got
        return Depends(dep)

    operator, admin_only = require("operator"), require("admin")

    app = FastAPI(title="UAE Discount Card Validation Agent", version=API_VERSION)
    app.state.registry, app.state.store, app.state.settings = registry, store, settings
    if cfg.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_credentials=False,
                           allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Run-Id"])

    @app.exception_handler(AdminError)
    async def _admin_error(_: Request, e: AdminError):
        return JSONResponse({"detail": e.message}, status_code=e.status)

    @app.get("/health")
    def health():
        return {"status": "ok", "registry": registry.version}

    v1 = APIRouter(prefix="/v1")

    @v1.get("/info")
    def info(role: str = operator):
        return {
            "service": "card-agent", "api_version": API_VERSION, "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION, "registry_version": registry.version,
            "families": len(registry.families), "registry_errors": registry.errors,
            "ocr": {"engine": ocr.engine_version(), "languages": ocr.full_lang(),
                    "installed": ocr.installed_languages()},
            "vlm": {"enabled": vlm.enabled(), "model": vlm.MODEL if vlm.enabled() else None},
            "stages": [{"id": i, "title": t, "stage": s} for i, t, s in STAGES],
            "auth": {"enabled": bool(keys), "role": role},
            "store": {"images": cfg.store_images, "retention_days": cfg.retention_days},
            "evaluation": {"cases": len(load_manifest(cfg.eval_dir))},
        }

    @v1.get("/whoami")
    def whoami(role: str = operator):
        return {"role": role, "auth_enabled": bool(keys)}

    # ---- validation ------------------------------------------------------------------------------
    def _log(result: dict) -> None:
        log.info(json.dumps(CardAgent.masked_log_view(result), ensure_ascii=False))

    @v1.post("/cards/validate", dependencies=[operator])
    async def validate_card(image: UploadFile = File(...), request_id: str | None = Form(None),
                            source: str = Form("api"), label: str | None = Form(None)):
        """Capture-system contract: multipart image → result envelope (unchanged). The run is also stored."""
        data = await image.read()
        run = agent.start(data, request_id=request_id, thresholds=settings.current)
        for _ in run.events():
            pass
        saved = store.save(run, source=source[:32], label=label or image.filename)
        _log(run.result)
        return JSONResponse(run.result, headers={"X-Run-Id": saved["id"]})

    @v1.post("/cards/validate/stream", dependencies=[operator])
    async def validate_stream(image: UploadFile = File(...), request_id: str | None = Form(None),
                              source: str = Form("console"), label: str | None = Form(None)):
        """Same validation, streamed as NDJSON: one `stage` event per stage start/end, then `result`."""
        data = await image.read()
        run = agent.start(data, request_id=request_id, thresholds=settings.current)
        name = label or image.filename

        def gen():
            try:
                yield _ndjson({"event": "start", "request_id": run.request_id,
                               "stages": [{"id": i, "title": t, "stage": s} for i, t, s in STAGES]})
                for ev in run.events():
                    yield _ndjson(ev)
                saved = store.save(run, source=source[:32], label=name)
                _log(run.result)
                yield _ndjson({"event": "result", "run": saved, "result": run.result})
            except Exception as e:  # noqa: BLE001 — report to the client instead of cutting the stream
                log.exception("validation failed")
                yield _ndjson({"event": "error", "message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(gen(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    # ---- runs & review ---------------------------------------------------------------------------
    @v1.get("/runs", dependencies=[operator])
    def list_runs(status: Literal["ACCEPT", "REVIEW", "REJECT"] | None = None, family: str | None = None,
                  queue: bool = False, reviewed: bool | None = None, q: str | None = Query(None, max_length=200),
                  since: str | None = None, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return store.list(status=status, family=family, queue=queue, reviewed=reviewed, q=q, since=since,
                          limit=limit, offset=offset)

    @v1.get("/runs/{run_id}", dependencies=[operator])
    def get_run(run_id: str):
        r = store.get(run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @v1.get("/runs/{run_id}/image", dependencies=[operator])
    def run_image(run_id: str, kind: Literal["original", "card"] = "original", w: int | None = Query(None, ge=32, le=2000)):
        p = store.image_path(run_id, kind)
        if not p:
            raise HTTPException(404, "image not stored")
        return _image_response(p, w)

    @v1.post("/runs/{run_id}/review", dependencies=[operator])
    def review_run(run_id: str, body: ReviewIn, request: Request):
        reviewer = body.reviewer or request.headers.get("x-forwarded-user") or request.headers.get("x-user")
        r = store.review(run_id, body.decision, body.note, reviewer)
        if not r:
            raise HTTPException(404, "run not found")
        log.info(json.dumps({"audit": "review", "run": run_id, "decision": body.decision, "reviewer": reviewer}))
        return r

    @v1.delete("/runs/{run_id}/review", dependencies=[operator])
    def clear_review(run_id: str):
        r = store.clear_review(run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @v1.post("/runs/{run_id}/enroll", dependencies=[admin_only])
    def enroll_run(run_id: str, body: EnrollIn):
        """Governance loop: a confirmed card becomes a reference image of its variant."""
        p = store.image_path(run_id, "card")
        if not p:
            raise HTTPException(409, "this run has no normalised card image to enroll")
        img = cv2.imread(str(p))
        ref = admin.add_ref(body.family, body.variant, img, warp=False, prefix=f"run_{run_id}")
        store.mark_enrolled(run_id, ref["id"])
        log.info(json.dumps({"audit": "enroll", "run": run_id, "ref": ref["id"]}))
        return ref

    @v1.delete("/runs/{run_id}", dependencies=[admin_only])
    def delete_run(run_id: str):
        if not store.delete(run_id):
            raise HTTPException(404, "run not found")
        return {"ok": True}

    @v1.get("/stats", dependencies=[operator])
    def stats(hours: int | None = Query(168, ge=1, le=24 * 366),
              tz: int = Query(0, ge=-840, le=840, description="viewer's UTC offset in minutes, for day/hour buckets")):
        return store.stats(hours, tz)

    @v1.get("/stats/all", dependencies=[operator])
    def stats_all(tz: int = Query(0, ge=-840, le=840)):
        return store.stats(None, tz)

    # ---- thresholds ------------------------------------------------------------------------------
    @v1.get("/settings", dependencies=[operator])
    def get_settings():
        return {"current": settings.current.as_dict(), "defaults": DEFAULTS.as_dict()}

    @v1.put("/settings", dependencies=[admin_only])
    def put_settings(patch: dict = Body(...)):
        try:
            cur = settings.update(patch)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        log.info(json.dumps({"audit": "settings", "patch": patch}))
        return {"current": cur.as_dict(), "defaults": DEFAULTS.as_dict()}

    @v1.delete("/settings", dependencies=[admin_only])
    def reset_settings():
        return {"current": settings.reset().as_dict(), "defaults": DEFAULTS.as_dict()}

    # ---- evaluation ------------------------------------------------------------------------------
    @v1.get("/eval/cases", dependencies=[operator])
    def eval_cases():
        return load_manifest(cfg.eval_dir)

    @v1.get("/eval/cases/{name}/image", dependencies=[operator])
    def eval_image(name: str, w: int | None = Query(None, ge=32, le=2000)):
        if name not in {c["file"] for c in load_manifest(cfg.eval_dir)}:
            raise HTTPException(404, "case not found")
        return _image_response(cfg.eval_dir / name, w)

    @v1.post("/eval", dependencies=[operator])
    def run_eval(body: EvalIn = Body(EvalIn())):
        try:
            t = settings.current.merged(body.thresholds or {})
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if errs := t.validate():
            raise HTTPException(422, "; ".join(errs))

        def gen():
            try:
                for ev in iter_eval(agent, cfg.eval_dir, t, body.today):
                    yield _ndjson(ev)
            except Exception as e:  # noqa: BLE001
                log.exception("evaluation failed")
                yield _ndjson({"event": "error", "message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    # ---- registry --------------------------------------------------------------------------------
    @v1.get("/registry", dependencies=[operator])
    def get_registry():
        return registry.summary()

    @v1.post("/registry/reload", dependencies=[admin_only])
    def reload_registry():
        registry.reload()
        return registry.summary()

    @v1.get("/registry/families", dependencies=[operator])
    def list_families():
        return admin.list_families()

    @v1.get("/registry/families/{family_id}", dependencies=[operator])
    def get_family(family_id: str):
        return admin.get_family(family_id)

    @v1.put("/registry/families/{family_id}", dependencies=[admin_only])
    async def put_family(family_id: str, request: Request, create: bool = False):
        text = (await request.body()).decode("utf-8")
        res = admin.put_family(family_id, text, create=create)
        log.info(json.dumps({"audit": "family.put", "family": family_id, "changed": res["changed"]}))
        return res

    @v1.delete("/registry/families/{family_id}", dependencies=[admin_only])
    def delete_family(family_id: str, with_refs: bool = True):
        res = admin.delete_family(family_id, with_refs=with_refs)
        log.info(json.dumps({"audit": "family.delete", "family": family_id}))
        return res

    @v1.get("/registry/families/{family_id}/history", dependencies=[operator])
    def family_history(family_id: str):
        return admin.history(family_id)

    @v1.get("/registry/families/{family_id}/history/{version}", dependencies=[operator])
    def family_history_version(family_id: str, version: str):
        return admin.history_version(family_id, version)

    @v1.get("/registry/refs", dependencies=[operator])
    def list_refs(family: str | None = None):
        if family:
            check_id(family, "family id")
        return admin.list_refs(family)

    @v1.get("/registry/refs/{family_id}/{variant_id}/{name}", dependencies=[operator])
    def ref_image(family_id: str, variant_id: str, name: str, w: int | None = Query(None, ge=32, le=2000)):
        return _image_response(admin.ref_path(family_id, variant_id, name), w)

    @v1.get("/registry/refs/{family_id}/{variant_id}/{name}/nearest", dependencies=[operator])
    def ref_nearest(family_id: str, variant_id: str, name: str):
        return admin.nearest(family_id, variant_id, name)

    @v1.post("/registry/refs/{family_id}/{variant_id}", dependencies=[admin_only])
    async def upload_refs(family_id: str, variant_id: str, images: list[UploadFile] = File(...), warp: bool = True):
        out = []
        for f in images:
            out.append(admin.add_ref(family_id, variant_id, decode_image(await f.read()), warp=warp))
        log.info(json.dumps({"audit": "refs.add", "family": family_id, "variant": variant_id, "count": len(out)}))
        return {"added": out, "registry_version": registry.version}

    @v1.delete("/registry/refs/{family_id}/{variant_id}/{name}", dependencies=[admin_only])
    def delete_ref(family_id: str, variant_id: str, name: str):
        res = admin.delete_ref(family_id, variant_id, name)
        log.info(json.dumps({"audit": "refs.delete", "ref": f"{family_id}/{variant_id}/{name}"}))
        return res

    @v1.get("/registry/separability", dependencies=[operator])
    def separability():
        return admin.separability()

    app.include_router(v1)

    if cfg.console_dir and cfg.console_dir.is_dir():
        app.mount("/", StaticFiles(directory=cfg.console_dir, html=True), name="console")

    return app


app = create_app()
