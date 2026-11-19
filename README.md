# UAE Discount Card Validation Agent

Fully offline (no public AI APIs). See **ARCHITECTURE.md** for the design. The operator console
(**card-agent-web**) is a separate React app that talks to this service's REST API.

## Run
```bash
pip install -r requirements.txt          # + system package: tesseract-ocr tesseract-ocr-ara
python tools/make_samples.py             # synthetic reference images + labelled test photos
python tests/run_eval.py                 # scorecard; JSON outputs in samples/results/
uvicorn api:app --port 8080              # REST API
curl -F image=@samples/fazaa_gold.jpg -F request_id=abc http://localhost:8080/v1/cards/validate
```
Or `docker build -t card-agent . && docker run -p 8080:8080 -v card-agent-data:/data card-agent`.

Tests: `pip install -r requirements-dev.txt && python -m pytest -q tests/test_api.py`.

### Windows
Tesseract is a separate program (pip only installs the Python wrapper):
```powershell
winget install UB-Mannheim.TesseractOCR      # installs to C:\Program Files\Tesseract-OCR
# Arabic data is needed for Arabic keywords. Either tick "Arabic" in the installer's language list,
# or copy https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata into
# C:\Program Files\Tesseract-OCR\tessdata\  (needs an admin shell)
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs   # should show ara and eng
```
`cardagent/ocr.py` finds the binary in the standard install folder automatically. Set `TESSERACT_CMD` to
point at a different `tesseract.exe`. If a language in `CARD_AGENT_OCR_LANGS` (default `eng+ara`) is not
installed, the service logs a warning and runs with the languages it has (the console shows the warning too).
Use `curl.exe` (not the PowerShell `curl` alias). If uvicorn logs `[Errno 10048]`, the port is taken — use
another one such as `--port 8090`.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `CARD_AGENT_API_KEY` | unset | Operator key: validate, runs, review. **Unset = API open (dev only).** |
| `CARD_AGENT_ADMIN_KEY` | unset | Admin key: registry edits, references, enroll, thresholds. Unset = the API key is admin. |
| `CARD_AGENT_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated console origins allowed to call the API. |
| `CARD_AGENT_REGISTRY_DIR` | `./registry` | Card-family YAML + reference images. |
| `CARD_AGENT_DATA_DIR` | `./data` | Run store (`runs.db`), stored images, `settings.json`. Put it on an encrypted volume. |
| `CARD_AGENT_STORE_IMAGES` | `true` | Keep the uploaded photo + normalised card for review and enrolment. |
| `CARD_AGENT_RETENTION_DAYS` | `90` | Runs and images older than this are deleted (0 = keep). |
| `CARD_AGENT_EVAL_DIR` | `./samples` | Labelled set (`manifest.json` + images) for `/v1/eval`. |
| `CARD_AGENT_CONSOLE_DIR` | unset | Serve the built console (`dist/`) at `/` from this service. |
| `CARD_AGENT_OCR_LANGS` | `eng+ara` | Tesseract languages for full-card OCR. |
| `CARD_AGENT_VLM_URL` / `_MODEL` | unset | Optional self-hosted VLM fallback (see `cardagent/vlm.py`). |

Keys are sent as `X-API-Key: <key>` or `Authorization: Bearer <key>`. Registry, threshold, review and
enrolment changes are written to the log as `audit` lines.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | /v1/cards/validate | **Capture-system contract.** Multipart `image` (+ `request_id`, `label`, `source`) → result JSON. Run id in `X-Run-Id`. |
| POST | /v1/cards/validate/stream | Same validation streamed as NDJSON: a `stage` event as each stage starts and ends, then `result`. |
| GET | /v1/runs | Stored runs. Filters: `status`, `family` (`unknown`), `queue` (unreviewed REVIEW), `reviewed`, `q`, `since`, `limit`, `offset`. |
| GET | /v1/runs/{id} | Run with result envelope and full stage trace. |
| GET | /v1/runs/{id}/image | `kind=original\|card`, optional `w` thumbnail width. |
| POST/DELETE | /v1/runs/{id}/review | Reviewer decision `{decision: ACCEPT\|REJECT, note}` (ground truth for accuracy) / reopen. |
| POST | /v1/runs/{id}/enroll | Add the run's normalised card to `{family, variant}` references (admin). |
| DELETE | /v1/runs/{id} | Delete a run and its images (admin). |
| GET | /v1/stats | Dashboard aggregates over `hours` (or `/v1/stats/all`); `tz` = viewer's UTC offset in minutes. |
| GET/PUT/DELETE | /v1/settings | Live decision thresholds (PUT a partial patch; DELETE resets). Writes need admin. |
| GET | /v1/eval/cases | Labelled test set. `/v1/eval/cases/{file}/image` for its photos. |
| POST | /v1/eval | Run the labelled set (optionally with what-if `thresholds`), streamed as NDJSON. Not stored. |
| GET | /v1/registry | Loaded families/variants, reference counts, load errors. |
| POST | /v1/registry/reload | Hot-reload YAML + reference images (admin). |
| GET | /v1/registry/families | Every family file with its YAML and load status. |
| GET/PUT/DELETE | /v1/registry/families/{id} | Raw YAML. PUT validates, keeps the old version in `.history/`, writes atomically, reloads; `?create=true` fails if it exists. DELETE moves to `.trash/`. |
| GET | /v1/registry/families/{id}/history[/{version}] | Previous versions of the file. |
| GET | /v1/registry/refs | Reference images (`?family=`). |
| GET/DELETE | /v1/registry/refs/{family}/{variant}/{file} | Image (`?w=` thumbnail) / move to `.trash/`. `/nearest` lists its most similar references. |
| POST | /v1/registry/refs/{family}/{variant} | Upload `images` (PNG/JPEG ≤ 8 MB). Cards are located and warped to 1012×638 first (`warp=false` to skip). |
| GET | /v1/registry/separability | Mean visual similarity between every pair of variant reference groups. |
| GET | /v1/info, /health | Versions, OCR languages, auth role, store config / liveness. |

## Add a new card type or variant (no code change)
Use the console (**Card families → Add card family**), or:
1. Create/edit `registry/<family>.yaml` (keywords, variants with colour + keywords + number prefix, field ROIs + regex, rules).
2. Put 5–20 clean images in `registry/refs/<family>/<variant>/` (PNG or JPEG).
3. `POST /v1/registry/reload`.

## Layout
```
api.py          REST API (auth, CORS, streaming, runs, registry admin, settings, evaluation)
cardagent/
  config.py     runtime thresholds (persisted to data/settings.json)
  quality.py    [1] blur / exposure / glare gate
  normalise.py  [2] card detection + perspective warp to 1012x638
  ocr.py        [3] local Tesseract (eng+ara), both polarities
  classify.py   [4] family (text+visual+colour fusion, score+margin) → [5] variant
  extract.py    [6] ROI OCR → full-text regex → optional local VLM
  vlm.py            self-hosted VLM adapter (Ollama/vLLM), off unless CARD_AGENT_VLM_URL set
  validate.py   [7] declarative rules (regex, not_expired, emirates_id_format, luhn, prefix)
  pipeline.py   [8] decision ACCEPT/REVIEW/REJECT + [9] JSON envelope, stage trace, PII-masked log view
  store.py      run store (SQLite) + images + review + stats + retention
  registry.py   loads YAML + reference images, atomic hot reload
  registry_admin.py  validated YAML writes, history, reference images, separability
  evaluate.py   labelled-set evaluation
registry/       card definitions + reference images
tests/          run_eval.py (scorecard), test_api.py (API tests)
```

## Important
Layouts, colours and number formats in `registry/*.yaml` and all sample images are **synthetic placeholders**.
Calibrate with real, consented card images, then tune the thresholds (console → Thresholds & settings, with
the what-if evaluation) against a labelled test set before production.
