#!/usr/bin/env python
"""Evaluate finished HoVer runs on the paper_v100 test split. **Spends money.**

    PYTHONUTF8=1 uv run python scripts/eval_hover_paper_split.py --all

Why a second HoVer test set
---------------------------
Our own HoVer split (D042-style) draws train/val/test from the 4,000-claim dev
release, stratified by ``num_hops`` (2/3/4). This one is a different population
entirely: ``vincentkoc/hover-parquet`` filtered to claims with **exactly 3
unique supporting titles**, 6,084 eligible, shuffled at seed 0, split 40/40/20
test/val/train, sampled at seed 1.

So it is not a re-run of the same measurement -- it is an out-of-distribution
check. Our candidates were selected on a val set with a 28/46/26 hop mix; here
every claim needs exactly three distinct articles. A candidate tuned to the
easier 2-hop cases has nowhere to hide.

Provenance is verified, not assumed: the filter reproduces the manifest's 6,084
eligible examples exactly, and the first five sampled test uids match the
supplied file in order. All 815 distinct gold titles resolve in the 2017
Wikipedia corpus, so no instance is unscoreable.

The metric is the same strict all-or-nothing retrieval used everywhere else in
this project: a claim scores 1 only if every one of its gold titles is
retrieved. Per-instance vectors are written so a paired test can run afterwards
without re-spending.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "data" / "hover" / "splits_150_100_300.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=None, help="run dir; repeatable")
    ap.add_argument("--all", action="store_true", help="every finished hover-{baseline,taxonomy}-seed*")
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--candidate-index", type=int, default=None, help="0 evaluates the SEED candidate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.hover.adapter import HoverAdapter, instances_by_id
    from gepa_taxonomy.hover.program import HoverMultiHopProgram
    from gepa_taxonomy.hover.tasks import instance_from_record

    require_credentials()

    runs: list[Path] = []
    if args.all:
        for arm in ("baseline", "taxonomy"):
            for s in (1, 2, 3):
                d = REPO / "results" / "runs" / f"hover-{arm}-seed{s}"
                if (d / "summary.json").exists():
                    runs.append(d)
    runs += [Path(r) for r in (args.run or [])]
    if not runs:
        raise SystemExit("no runs selected; pass --all or --run")

    doc = json.loads(args.split.read_text(encoding="utf-8"))
    test = [instance_from_record(r) for r in doc["test"]]
    print(f"paper_v100 test: {len(test)} claims, every one needing 3 distinct titles")
    print(f"evaluating {len(runs)} candidate(s)\n", flush=True)

    retriever = WikiRetriever(k=args.k).load()
    results = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8-sig"))
        index = args.candidate_index if args.candidate_index is not None else summary.get("best_candidate_index")
        candidates = json.loads((run / "candidates.json").read_text(encoding="utf-8-sig"))
        if index is None or index >= len(candidates):
            print(f"  {run.name}: no usable candidate index; skipping", flush=True)
            continue
        candidate = candidates[index]

        meter = CostMeter()
        program = HoverMultiHopProgram(
            retriever=retriever,
            lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
            meter=meter, model=args.solver_model, k=args.k,
        )
        # reflection_gold_ids left at its default: nothing on a TEST split may
        # ever see gold titles, and evaluate() takes Instances, not Tasks.
        adapter = HoverAdapter(program=program, instances=instances_by_id(test), max_workers=args.workers)
        started = time.time()
        batch = adapter.evaluate(test, candidate, capture_traces=False)
        scores = list(batch.scores)
        per_instance = {i.task.example_id: float(s) for i, s in zip(test, scores)}
        strict = sum(scores) / len(scores) if scores else 0.0
        spend = meter.snapshot()["budgeted_usd"]
        print(f"  {run.name:<26} candidate {index:>3}  strict {strict:.4f}  "
              f"${spend:.2f}  {(time.time()-started)/60:.1f} min", flush=True)
        results.append({
            "run": run.name, "candidate_index": index,
            "val_score": summary.get("best_val_score"),
            "test_strict": strict, "per_instance": per_instance,
            "spend_usd": round(spend, 4),
            "elapsed_min": round((time.time() - started) / 60, 2),
        })

    if not results:
        raise SystemExit("nothing evaluated")
    out = args.out or REPO / "results" / "test_eval" / f"hover_paper_v100_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "hover", "split": "paper_v100", "source": str(args.split),
        "n": len(test), "candidates": results,
        "spend_usd": round(sum(r["spend_usd"] for r in results), 4),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  total ${sum(r['spend_usd'] for r in results):.2f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
