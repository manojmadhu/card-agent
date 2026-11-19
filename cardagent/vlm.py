"""Optional fallback: a SELF-HOSTED vision-language model (e.g. Qwen2.5-VL via Ollama/vLLM).

Disabled unless CARD_AGENT_VLM_URL points to an internal endpoint. Never a public API.
Example: CARD_AGENT_VLM_URL=http://10.0.0.5:11434  CARD_AGENT_VLM_MODEL=qwen2.5vl:7b
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

import cv2

URL = os.getenv("CARD_AGENT_VLM_URL", "")
MODEL = os.getenv("CARD_AGENT_VLM_MODEL", "qwen2.5vl:7b")


def enabled() -> bool:
    return bool(URL)


def extract_field(img_bgr, field_name: str, regex: str) -> str | None:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return None
    prompt = (f"Read the '{field_name}' printed on this membership card. "
              f"It must match the regex {regex}. Reply only with JSON: "
              f'{{"value": "<text or null>"}}. If not visible, use null. Do not guess.')
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0},
                       "images": [base64.b64encode(buf.tobytes()).decode()]}).encode()
    try:
        req = urllib.request.Request(f"{URL}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(json.loads(r.read())["response"]).get("value")
    except Exception:
        return None
