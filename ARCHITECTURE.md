# UAE Discount Card Validation Agent — Architecture

**Constraint:** no public AI APIs. Every model (OCR, vision embeddings, optional VLM) runs on-premise / in your private cloud. Card images never leave your network.

---

## 1. Context

```
┌──────────────┐  image + meta   ┌─────────────────────────┐  JSON result  ┌──────────────┐
│ Capture      │ ──────────────▶ │  Card Validation Agent  │ ────────────▶ │ Consumer /   │
│ system       │  (REST / queue) │  (this service)         │               │ storage sys  │
└──────────────┘                 └───────────┬─────────────┘               └──────────────┘
                                             │ low-confidence cases
                                             ▼
                                   ┌─────────────────────┐
                                   │ Human review queue  │ ── corrections ──▶ training/reference set
                                   └─────────────────────┘
```

## 2. Core design principle: config-driven Card Registry

The agent never hard-codes card knowledge. Each **card family** (Fazaa, Esaad, …) is a YAML file with:

| Section | Purpose |
|---|---|
| `family` / `issuer` | ID and display name |
| `keywords` | Text that proves the family (issuer name, English + Arabic) |
| `negative_keywords` | Text that proves it's *not* this family (avoids look-alikes) |
| `reference_images/` | 5–20 clean samples per variant, used for visual matching |
| `variants[]` | e.g. silver / gold / platinum — each with colour profile, variant keywords, number prefix rules |
| `fields[]` | What to extract: name, card number, expiry, Emirates ID…, with regex, region-of-interest (ROI), required flag |
| `rules[]` | Validation rules (expiry in future, number format, checksum, cross-field checks) |

**Adding a new card or variant = add YAML + reference images. No code change, no redeploy** (registry hot-reloads).

## 3. Agent pipeline (hierarchical: Family → Variant → Fields → Rules)

```
 [0] Ingest ──▶ [1] Quality gate ──▶ [2] Normalise ──▶ [3] Full-card OCR
                     │ fail → REJECT (RETAKE)                     │
                                                                  ▼
 [4] FAMILY classification (coarse)  ◀── evidence fusion ──  OCR text + visual embedding + colour
        │ top-1 score < τ_family or margin < δ  → REVIEW (unknown card)
        ▼
 [5] VARIANT classification (fine, only among that family's variants)
        │ ambiguous → REVIEW
        ▼
 [6] FIELD extraction (variant's template ROIs → regex)  ──fallback──▶ [6b] local VLM, schema-constrained
        ▼
 [7] VALIDATION rules engine (format, checksum, expiry, cross-field, optional issuer lookup)
        ▼
 [8] DECISION: ACCEPT / REVIEW / REJECT  +  confidence  +  reasons
        ▼
 [9] JSON output (fixed envelope) + audit log
```

### Stage details

**[1] Quality gate** — blur (Laplacian variance), resolution, brightness/glare ratio. Fails fast with `RETAKE_REQUIRED` so the capture system can prompt the user again instead of wasting compute.

**[2] Normalise** — detect card quadrilateral (contour + perspective warp) → fixed 1012×638 (ISO ID-1 ratio 85.6×54 mm); fix rotation (try 0/90/180/270, keep the orientation with best OCR confidence).

**[3] OCR** — local engine (Tesseract `eng+ara`, or PaddleOCR for better accuracy). Produces tokens with boxes and confidences.

**[4] Family classification — why multi-signal fusion:** one signal alone fails on real photos. Text can be glared out; colours shift under lighting; layouts are shared by different issuers. So we score every family on three independent signals and fuse:

| Signal | How | Robust to |
|---|---|---|
| S_text | fuzzy keyword match of OCR text vs `keywords` (minus `negative_keywords`) | colour/lighting changes |
| S_visual | cosine similarity of image embedding vs reference images (prototype: ORB keypoints + colour histogram; production: local CLIP/DINOv2 + vector index e.g. Qdrant/FAISS) | glare on text |
| S_layout | colour/hue profile of the card body | partial OCR failure |

`score_family = w_t·S_text + w_v·S_visual + w_c·S_colour` (weights per registry, tuned on validation set).
Decision requires **both** `top1 ≥ τ_family` **and** `top1 − top2 ≥ δ` (margin) — this is what prevents confident wrong answers between look-alike cards.

**[5] Variant classification** — runs *only* within the chosen family (smaller, easier problem). Signals: dominant body hue/saturation/brightness vs variant colour profile (silver = low saturation/high value, gold = hue ≈ 40–55°, platinum = dark/low-sat), variant keywords printed on card ("GOLD", "PLATINUM", "ذهبية"), card-number prefix rule. Same threshold + margin logic.

**[6] Field extraction** — each variant points to a layout template. Crop ROI → OCR → regex → normalise (dates to ISO-8601, numbers stripped of spaces). If a required field fails regex, **fallback** to a locally-hosted VLM (Qwen2.5-VL-7B / InternVL2 via vLLM or Ollama) prompted with the field schema and constrained to JSON. VLM output is *still* validated by the same regex/rules — the VLM never gets the final word.

**[7] Validation rules engine** — declarative rules from YAML:
`regex`, `not_expired`, `date_order`, `luhn`, `emirates_id_format` (784-YYYY-NNNNNNN-C), `prefix_matches_variant`, `lookup` (optional hook to issuer/membership DB if you have one). Each rule returns pass/fail/skip + reason code.

**[8] Decision policy**

| Outcome | Condition |
|---|---|
| `ACCEPT` | family & variant above thresholds, all required fields extracted, all `critical` rules pass |
| `REVIEW` | any confidence in grey zone, a non-critical rule failed, or VLM fallback was used for a critical field |
| `REJECT` | quality gate failed, unknown card, card expired, critical rule failed with high confidence |

Thresholds are config, so the business can trade automation rate vs risk.

## 4. Output contract (JSON)

```json
{
  "request_id": "…",
  "status": "ACCEPT | REVIEW | REJECT",
  "card": {
    "family": "fazaa", "family_name": "Fazaa", "family_confidence": 0.93,
    "variant": "gold", "variant_confidence": 0.88
  },
  "fields": {
    "holder_name": {"value": "AHMED ALI", "confidence": 0.91, "source": "ocr"},
    "card_number": {"value": "…", "confidence": 0.95, "source": "ocr"},
    "expiry_date": {"value": "2027-05-31", "confidence": 0.9, "source": "ocr"}
  },
  "validation": [{"rule": "not_expired", "field": "expiry_date", "result": "pass"}],
  "reasons": [],
  "quality": {"blur": 412.3, "brightness": 0.55, "glare_ratio": 0.01},
  "candidates": [{"family": "fazaa", "score": 0.93}, {"family": "esaad", "score": 0.31}],
  "model_versions": {"registry": "…", "ocr": "tesseract-5.3.4", "pipeline": "0.1.0"},
  "processing_ms": 640
}
```
Stable envelope, versioned (`schema_version`), JSON-Schema published so the storage system can validate it.

## 5. Deployment (all private)

| Component | Prototype | Production |
|---|---|---|
| API | FastAPI | FastAPI behind internal gateway, mTLS |
| Queue | sync REST | RabbitMQ/Kafka for bursts, idempotent by `request_id` |
| OCR | Tesseract 5 | PaddleOCR (GPU) or Tesseract; Arabic+English |
| Visual match | ORB + colour histograms | CLIP / DINOv2 embeddings in Qdrant / FAISS |
| VLM fallback | adapter stub (Ollama URL) | Qwen2.5-VL-7B on vLLM, 1× L4/A10 GPU, on-prem |
| Storage | — | Result JSON → your system; images encrypted at rest, retention policy (UAE PDPL) |
| Observability | logs | Prometheus metrics: auto-accept rate, review rate, per-family confusion, latency |

## 6. Accuracy & governance loop

1. Collect 50–200 real images per variant (consented), split train/val/test.
2. Tune thresholds τ/δ to hit target precision (e.g. ≥ 99% on ACCEPT).
3. Every human-review correction is stored → added to reference set → re-index embeddings weekly.
4. Track a confusion matrix per family/variant; alert on drift.
5. Security: PII masking in logs (card numbers truncated), RBAC on review UI, audit trail of every decision.

## 7. Why not a single "LLM reads the card" call?

- Non-deterministic, hard to audit, hallucinates plausible numbers.
- Heavy GPU cost per call.
- The hierarchical pipeline gives explainable reasons per stage, deterministic results, and uses the VLM only where it adds value (hard fields), with its output re-validated.
