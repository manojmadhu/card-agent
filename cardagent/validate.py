"""Stage 7 — declarative rules engine (rules come from the family YAML)."""
from __future__ import annotations

import calendar
import re
from datetime import date

from .registry import Family


def parse_expiry(mm_yy: str) -> date | None:
    m = re.match(r"^(\d{2})/(\d{2}|\d{4})$", mm_yy or "")
    if not m:
        return None
    month, year = int(m.group(1)), int(m.group(2))
    year = year + 2000 if year < 100 else year
    return date(year, month, calendar.monthrange(year, month)[1])  # valid through month end


def luhn_ok(num: str) -> bool:
    digits = [int(c) for c in num][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def run_rules(fam: Family, variant_id: str | None, fields: dict, today: date | None = None) -> list[dict]:
    today = today or date.today()
    results = []
    field_cfg = {f["name"]: f for f in fam.fields}
    for rule in fam.rules:
        t, fname, crit = rule["type"], rule.get("field"), bool(rule.get("critical"))
        val = (fields.get(fname) or {}).get("value")
        res, detail = "skip", None
        if val is None:
            res = "fail" if field_cfg.get(fname, {}).get("required") else "skip"
            detail = "FIELD_MISSING"
        elif t == "regex":
            res = "pass" if re.match(field_cfg[fname]["regex"], val) else "fail"
        elif t == "not_expired":
            exp = parse_expiry(val)
            if exp is None:
                res, detail = "fail", "UNPARSEABLE_DATE"
            else:
                res = "pass" if exp >= today else "fail"
                detail = f"valid_until={exp.isoformat()}"
                if res == "fail":
                    detail = "CARD_EXPIRED " + detail
        elif t == "emirates_id_format":
            res = "pass" if re.match(r"^784-(19|20)\d{2}-\d{7}-\d$", val) else "fail"
        elif t == "luhn":
            res = "pass" if luhn_ok(re.sub(r"\D", "", val)) else "fail"
        elif t == "prefix_matches_variant":
            v = fam.variant(variant_id) if variant_id else None
            if v and v.number_prefix:
                res = "pass" if val.startswith(v.number_prefix) else "fail"
                if res == "fail":
                    detail = "NUMBER_PREFIX_VARIANT_MISMATCH"
        results.append({"rule": t, "field": fname, "critical": crit, "result": res, "detail": detail})
    return results
