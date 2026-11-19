"""API tests. Run:  python -m pytest -q tests/test_api.py   (needs Tesseract, like the service)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api import Config, create_app  # noqa: E402

SAMPLES = ROOT / "samples"


def _cfg(tmp: Path, **kw) -> Config:
    reg = tmp / "registry"
    shutil.copytree(ROOT / "registry", reg)
    ev = tmp / "eval"
    ev.mkdir()
    cases = [c for c in json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
             if c["file"] in ("fazaa_gold.jpg", "fazaa_blurry.jpg")]
    for c in cases:
        shutil.copy(SAMPLES / c["file"], ev / c["file"])
    (ev / "manifest.json").write_text(json.dumps(cases), encoding="utf-8")
    return Config(registry_dir=reg, data_dir=tmp / "data", eval_dir=ev, api_key=kw.get("api_key"),
                  admin_key=kw.get("admin_key"), cors_origins=[], store_images=True, retention_days=90,
                  console_dir=None)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(_cfg(tmp_path_factory.mktemp("ca")))) as c:
        yield c


def _validate(client, name="fazaa_gold.jpg", **form):
    with open(SAMPLES / name, "rb") as f:
        return client.post("/v1/cards/validate", files={"image": (name, f, "image/jpeg")}, data=form)


def _ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_health_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/v1/info").json()
    assert info["families"] == 2 and [s["id"] for s in info["stages"]][0] == "ingest"


def test_validate_contract_and_run_stored(client):
    r = _validate(client, request_id="cap-1")
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "ACCEPT" and res["card"]["family"] == "fazaa" and res["card"]["variant"] == "gold"
    assert res["request_id"] == "cap-1" and res["schema_version"] == "1.0"
    run_id = r.headers["x-run-id"]
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["status"] == "ACCEPT" and run["exit_stage"] == "accepted"
    assert [s["id"] for s in run["trace"]][-1] == "decision" and all(s["status"] != "pending" for s in run["trace"])
    assert client.get(f"/v1/runs/{run_id}/image").headers["content-type"].startswith("image/")
    thumb = client.get(f"/v1/runs/{run_id}/image", params={"kind": "card", "w": 200})
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_stream_events(client):
    with open(SAMPLES / "fazaa_blurry.jpg", "rb") as f:
        r = client.post("/v1/cards/validate/stream", files={"image": ("b.jpg", f, "image/jpeg")})
    evs = _ndjson(r)
    assert evs[0]["event"] == "start" and evs[-1]["event"] == "result"
    q = [e for e in evs if e.get("id") == "quality"]
    assert q[0]["status"] == "running" and q[-1]["status"] == "fail"
    assert evs[-1]["result"]["status"] == "REJECT" and "RETAKE_REQUIRED" in evs[-1]["result"]["reasons"]
    assert any(e.get("id") == "family" and e["status"] == "skip" for e in evs)


def test_invalid_image(client):
    r = client.post("/v1/cards/validate", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.json()["status"] == "REJECT" and r.json()["reasons"] == ["INVALID_IMAGE"]


def test_review_and_stats(client):
    run_id = _validate(client).headers["x-run-id"]
    r = client.post(f"/v1/runs/{run_id}/review", json={"decision": "ACCEPT", "note": "checked", "reviewer": "qa"})
    assert r.json()["review"]["decision"] == "ACCEPT"
    s = client.get("/v1/stats", params={"hours": 24}).json()
    assert s["total"] >= 2 and s["review"]["confusion"]["ACCEPT"]["ACCEPT"] >= 1 and s["review"]["agreement"] == 1.0
    for tz in (-600, 0, 240, 840):  # every run lands in exactly one bucket whatever the viewer's timezone
        st = client.get("/v1/stats", params={"hours": 168, "tz": tz}).json()
        assert sum(p[d] for p in st["series"]["points"] for d in ("ACCEPT", "REVIEW", "REJECT")) == st["total"]
    lst = client.get("/v1/runs", params={"status": "ACCEPT", "limit": 5}).json()
    assert lst["total"] >= 2 and all(i["status"] == "ACCEPT" for i in lst["items"])
    assert client.post(f"/v1/runs/{run_id}/review", json={"decision": "MAYBE"}).status_code == 422


def test_settings(client):
    assert client.get("/v1/settings").json()["current"]["family_min"] == 0.5
    assert client.put("/v1/settings", json={"family_min": 2}).status_code == 422
    assert client.put("/v1/settings", json={"nope": 1}).status_code == 422
    assert client.put("/v1/settings", json={"min_blur": 9999}).status_code == 422
    assert client.put("/v1/settings", json={"family_min": 0.55}).json()["current"]["family_min"] == 0.55
    assert client.delete("/v1/settings").json()["current"]["family_min"] == 0.5


def test_family_crud_and_history(client):
    fam = client.get("/v1/registry/families/fazaa").json()
    assert fam["loaded"] and "family: fazaa" in fam["yaml"]
    bad = fam["yaml"].replace("weights: {text: 0.55", "weights: {text: 0.95")
    r = client.put("/v1/registry/families/fazaa", content=bad)
    assert r.status_code == 422 and "sum to 1" in r.json()["detail"]
    good = fam["yaml"].replace("issuer: Fazaa (UAE)", "issuer: Fazaa UAE")
    assert client.put("/v1/registry/families/fazaa", content=good).json()["changed"] is True
    hist = client.get("/v1/registry/families/fazaa/history").json()
    assert len(hist) == 1
    old = client.get(f"/v1/registry/families/fazaa/history/{hist[0]['version']}").json()
    assert "issuer: Fazaa (UAE)" in old["yaml"]
    assert client.put("/v1/registry/families/fazaa?create=true", content=good).status_code == 409
    new = good.replace("family: fazaa", "family: noor")
    assert client.put("/v1/registry/families/noor?create=true", content=new).status_code == 200
    assert {f["family"] for f in client.get("/v1/registry").json()["families"]} == {"fazaa", "esaad", "noor"}
    assert client.delete("/v1/registry/families/noor").json()["ok"]
    assert client.get("/v1/registry/families/noor").status_code == 404
    assert client.get("/v1/registry/families/..%2Fapi").status_code in (400, 404)


def test_refs_and_enroll(client):
    before = len(client.get("/v1/registry/refs", params={"family": "fazaa"}).json())
    with open(SAMPLES / "fazaa_silver.jpg", "rb") as f:
        r = client.post("/v1/registry/refs/fazaa/silver", files=[("images", ("s.jpg", f, "image/jpeg"))])
    added = r.json()["added"][0]
    assert added["card_detected"] is True
    refs = client.get("/v1/registry/refs", params={"family": "fazaa"}).json()
    assert len(refs) == before + 1
    fam, var, name = added["id"].split("/")
    assert client.get(f"/v1/registry/refs/{fam}/{var}/{name}", params={"w": 120}).status_code == 200
    assert len(client.get(f"/v1/registry/refs/{fam}/{var}/{name}/nearest").json()) > 0
    assert client.delete(f"/v1/registry/refs/{fam}/{var}/{name}").json()["ok"]
    with open(SAMPLES / "unknown_card.jpg", "rb") as f:
        bad = client.post("/v1/registry/refs/fazaa/nope", files=[("images", ("u.jpg", f, "image/jpeg"))])
    assert bad.status_code == 404
    run_id = _validate(client).headers["x-run-id"]
    e = client.post(f"/v1/runs/{run_id}/enroll", json={"family": "fazaa", "variant": "gold"}).json()
    assert e["id"].startswith("fazaa/gold/run_")
    assert client.get(f"/v1/runs/{run_id}").json()["enrolled"] == [e["id"]]
    sep = client.get("/v1/registry/separability").json()
    assert len(sep["matrix"]) == len(sep["groups"]) >= 7


def test_eval_stream(client):
    evs = _ndjson(client.post("/v1/eval", json={}))
    summary = evs[-1]
    assert summary["event"] == "summary" and summary["total"] == 2 and summary["passed"] == 2
    assert client.post("/v1/eval", json={"thresholds": {"family_min": 5}}).status_code == 422


def test_auth(tmp_path):
    with TestClient(create_app(_cfg(tmp_path, api_key="op-key", admin_key="admin-key"))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/registry").status_code == 401
        assert c.get("/v1/registry", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/registry", headers={"X-API-Key": "op-key"}).status_code == 200
        assert c.get("/v1/info", headers={"Authorization": "Bearer op-key"}).json()["auth"]["role"] == "operator"
        assert c.put("/v1/settings", json={"family_min": 0.6}, headers={"X-API-Key": "op-key"}).status_code == 403
        assert c.put("/v1/settings", json={"family_min": 0.6}, headers={"X-API-Key": "admin-key"}).status_code == 200
