#!/usr/bin/env python
"""Reduce a generated taxonomy from measured trace support. FREE: offline.

Reads the full taxonomy and the judgements produced by ``judge_corpus.py``,
keeps the codes the corpus actually supports, and writes:

* ``taxonomy.full.json``   -- the taxonomy exactly as generated (preserved)
* ``taxonomy.json``        -- the reduced taxonomy downstream runs load
* ``reduction_report.json``-- every generated code: retained / ungrounded /
                              over_cap, with its support count

Deterministic and re-runnable: changing the threshold is a re-run over stored
artifacts, not another billed generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from failure_taxonomy.reduce import reduce_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True, help="taxonomy.json as generated")
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="drop codes cited in fewer distinct traces than this. 1 removes only "
        "codes the corpus never exhibits; the default 2 additionally filters a "
        "code resting on a single annotation.",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=25,
        help="safety net, not the active filter: if this binds before "
        "--min-support does, the taxonomy is being shaped by a count rather "
        "than by evidence (the report says when that happened).",
    )
    args = parser.parse_args()

    document = json.loads(args.taxonomy.read_text(encoding="utf-8-sig"))
    if "reduction" in document:
        raise SystemExit(
            f"REFUSING: {args.taxonomy} is already reduced. Re-reduce from "
            "taxonomy.full.json instead, so thresholds always apply to the full draft."
        )
    judgements = {}
    for line in args.judgements.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            judgements[str(row["trace_id"])] = row.get("failure_modes") or []

    result = reduce_taxonomy(document, judgements, min_support=args.min_support, max_codes=args.max_codes)

    out_dir = args.taxonomy.parent
    full = out_dir / "taxonomy.full.json"
    if not full.exists():
        full.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    args.taxonomy.write_text(json.dumps(result.document, indent=1) + "\n", encoding="utf-8")
    report_path = out_dir / "reduction_report.json"
    report_path.write_text(json.dumps(result.report, indent=1) + "\n", encoding="utf-8")

    t = result.report["tallies"]
    print(
        f"reduced: {t['retained']} retained, {t['ungrounded']} ungrounded, "
        f"{t['over_cap']} over cap (of {result.report['tallies']['retained'] + t['ungrounded'] + t['over_cap']} generated)"
    )
    if result.document["reduction"]["cap_bound"]:
        print(
            "  [!] --max-codes bound before --min-support did: the count, not the "
            "evidence, shaped this taxonomy. Consider raising --max-codes."
        )
    print(f"full draft : {full}")
    print(f"reduced    : {args.taxonomy}")
    print(f"report     : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
