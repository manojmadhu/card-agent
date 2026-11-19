"""Evaluate the agent against samples/manifest.json and print a scorecard."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cardagent.pipeline import CardAgent  # noqa: E402
from cardagent.registry import Registry  # noqa: E402

TODAY = date(2026, 11, 16)


def main():
    # Windows consoles default to cp1252, which can't print ✓/✅/❌.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agent = CardAgent(Registry(ROOT / "registry"))
    manifest = json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "samples" / "results"
    out_dir.mkdir(exist_ok=True)
    ok = 0
    print(f"{'file':34} {'expected':8} {'got':8} {'family':7} {'variant':10} {'num✓':5} reasons")
    for case in manifest:
        res = agent.process((ROOT / "samples" / case["file"]).read_bytes(), request_id=case["file"], today=TODAY)
        (out_dir / (Path(case["file"]).stem + ".json")).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        card = res["card"] or {}
        num = (res["fields"].get("card_number") or {}).get("value")
        fam_ok = card.get("family") == case["family"] if case["family"] and res["status"] != "REJECT" else True
        var_ok = card.get("variant") == case["variant"] if case["variant"] and res["status"] != "REJECT" or card.get("variant") else True
        num_ok = (num == case.get("card_number")) if case.get("card_number") and res["fields"] else None
        passed = res["status"] == case["expected_status"] and fam_ok and (var_ok or res["status"] == "REJECT")
        ok += passed
        print(f"{case['file']:34} {case['expected_status']:8} {res['status']:8} {str(card.get('family')):7} "
              f"{str(card.get('variant')):10} {str(num_ok):5} {', '.join(res['reasons'])[:70]}  {'✅' if passed else '❌'}")
    print(f"\n{ok}/{len(manifest)} cases matched expected outcome")


if __name__ == "__main__":
    main()
