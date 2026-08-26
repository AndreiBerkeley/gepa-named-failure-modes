#!/usr/bin/env python
"""Judge a generated taxonomy over its own generation corpus. **Spends money.**

The output feeds evidence-based reduction (``scripts/reduce_taxonomy.py``):
per-trace failure-mode citations, from which support is measured. The gate
outcome of generation is deliberately irrelevant here -- a ``review_required``
taxonomy is judged all the same, because grounding is measured regardless of
whether the agreement gate passed.

``max_trace_chars`` is sized from the corpus, not copied from another trial:
when it is too low the judge inserts its own truncation marker and then codes
the marker as a failure mode, manufacturing a spurious code.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gepa_taxonomy.adamast_trace import AdamastRecord
from gepa_taxonomy.cost import CostMeter
from gepa_taxonomy.taxonomy_judge import MAX_OUTPUT_TOKENS, TaxonomyJudge

#: Room for the judge's own framing on top of the longest trace.
TRACE_CHAR_MARGIN = 8_192


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="judgements JSONL to write")
    parser.add_argument("--provider", default="bedrock")
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=10, help="traces per judge call")
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--adamast-python", type=Path, default=None)
    args = parser.parse_args()

    if not args.traces.exists():
        raise SystemExit(f"traces not found: {args.traces}")
    if args.out.exists():
        raise SystemExit(
            f"REFUSING: {args.out} already exists. Judgements key a reduction; "
            "delete it deliberately or write elsewhere."
        )

    rows = [json.loads(line) for line in args.traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no traces in {args.traces}")

    longest = max(len(r.get("raw_trajectory") or "") for r in rows)
    max_trace_chars = longest + TRACE_CHAR_MARGIN
    print(f"corpus: {len(rows)} traces, longest {longest} chars -> max_trace_chars {max_trace_chars}", flush=True)

    meter = CostMeter()
    judge = TaxonomyJudge(
        taxonomy_path=args.taxonomy,
        meter=meter,
        model=args.model,
        provider=args.provider,
        max_trace_chars=max_trace_chars,
        max_output_tokens=args.max_output_tokens,
        phase="generation",
        allow_review_required=True,
        python=args.adamast_python,
    )
    judge.preflight()

    chunks = [rows[i : i + args.chunk] for i in range(0, len(rows), args.chunk)]
    results: dict[str, list[dict]] = {}
    lock = threading.Lock()
    done = 0

    def judge_chunk(chunk: list[dict]) -> None:
        nonlocal done
        records = [
            AdamastRecord(
                problem_id=str(r["problem_id"]),
                task=str(r.get("task") or ""),
                raw_trajectory=str(r.get("raw_trajectory") or ""),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in chunk
        ]
        response = judge._run(records)
        with lock:
            judge._meter_usage(response.get("usage") or {})
            by_trace = {str(d.get("trace_id", "")): d for d in response.get("diagnoses") or []}
            for record in records:
                diagnosis = by_trace.get(record.problem_id)
                results[record.problem_id] = list((diagnosis or {}).get("failure_modes") or [])
            done += len(records)
            print(f"judged {done}/{len(rows)} traces", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge_chunk, chunk) for chunk in chunks]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                failures += 1
                print(f"  [!] judge chunk failed: {exc}", file=sys.stderr, flush=True)

    if not results:
        raise SystemExit("every judge call failed; no judgements to write")
    coverage = len(results) / len(rows)
    if coverage < 0.9:
        raise SystemExit(
            f"REFUSING: only {len(results)}/{len(rows)} traces were judged "
            f"({coverage:.0%}). Support measured from a heavily partial judging "
            "pass would silently bias the reduction; fix the failures and rerun."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for trace_id in sorted(results):
            fh.write(json.dumps({"trace_id": trace_id, "failure_modes": results[trace_id]}) + "\n")

    meta = {
        "taxonomy": str(args.taxonomy),
        "taxonomy_fingerprint": judge.fingerprint,
        "provider": args.provider,
        "model": args.model,
        "traces": len(rows),
        "judged": len(results),
        "failed_chunks": failures,
        "longest_trace_chars": longest,
        "max_trace_chars": max_trace_chars,
        "spend_usd": round(meter.budgeted_usd + meter.excluded_usd, 6),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"judgements: {args.out}  ({len(results)}/{len(rows)} traces, ${meta['spend_usd']})", flush=True)
    if len(results) < len(rows):
        print(f"  [!] {len(rows) - len(results)} traces have no judgement (failed chunks)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
