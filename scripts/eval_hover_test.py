#!/usr/bin/env python
"""Evaluate finished HoVer seeds on the held-out test split. **Spends money.**

Test is touched exactly once per candidate, at the end. val drives selection, so
a val score is an optimistically biased estimate of test -- the SWE-Bench round
lost 7.7pp between them (21.7% val -> 14.0% test), and on HotpotQA the gap ran
4.3-7.9pp. That is why this is a separate script and a separate split.

    PYTHONUTF8=1 uv run python scripts/eval_hover_test.py --all
    PYTHONUTF8=1 uv run python scripts/eval_hover_test.py --run results/runs/hover-baseline-seed1
    # the base-on-test reference, without which no other number means anything:
    PYTHONUTF8=1 uv run python scripts/eval_hover_test.py --run results/runs/hover-baseline-seed1 --candidate-index 0

Per-instance scores are written out so a paired test can be run afterwards
without re-spending. Pair per seed -- the three seeds share the same 300 test
claims, so pooling them triples n against non-independent samples and the pooled
p-value is not interpretable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def evaluate_candidate(run: Path, args, test, retriever) -> dict | None:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from gepa_taxonomy.bedrock import BedrockLM
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hover.adapter import HoverAdapter, instances_by_id
    from gepa_taxonomy.hover.program import SEED_CANDIDATE, HoverMultiHopProgram

    summary_path = run / "summary.json"
    if not summary_path.exists():
        print(f"  {run.name}: no summary.json -- run unfinished, skipping")
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))

    candidates_path = run / "candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.exists() else []

    if args.candidate_index is not None:
        index = args.candidate_index
        label = f"{run.name}#{index}" + (" (SEED/base)" if index == 0 else "")
    else:
        index = summary.get("best_candidate_index")
        if index is None:
            print(f"  {run.name}: no best_candidate_index; skipping")
            return None
        label = run.name
    if index >= len(candidates):
        print(f"  {run.name}: candidate {index} not in candidates.json ({len(candidates)} present); skipping")
        return None

    candidate = dict(candidates[index]) if candidates else dict(SEED_CANDIDATE)
    missing = set(SEED_CANDIDATE) - set(candidate)
    if missing:
        print(f"  {run.name}: candidate {index} missing components {sorted(missing)}; skipping")
        return None

    meter = CostMeter()
    program = HoverMultiHopProgram(
        retriever=retriever,
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        k=args.k,
    )
    # reflection_gold_ids stays None: nothing on the TEST split may ever see a
    # gold title, and there is no reflection here anyway.
    adapter = HoverAdapter(program=program, instances=instances_by_id(test), max_workers=args.workers)

    started = time.time()
    batch = adapter.evaluate(test, candidate, capture_traces=True)
    elapsed = time.time() - started

    by_hop: dict[int, list[float]] = {}
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        by_hop.setdefault(adapter.instances[trace["example_id"]].task.num_hops, []).append(score)

    result = {
        "label": label,
        "run": str(run.relative_to(REPO)),
        "candidate_index": index,
        "val_score": summary.get("best_val_score") if args.candidate_index is None else None,
        "test_strict": statistics.mean(batch.scores),
        "test_loose": statistics.mean(t["grading"]["loose_recall"] for t in batch.trajectories),
        "by_num_hops": {h: round(statistics.mean(v), 4) for h, v in sorted(by_hop.items())},
        "per_instance": {t["example_id"]: s for t, s in zip(batch.trajectories, batch.scores, strict=True)},
        "spend_usd": round(meter.budgeted_usd, 4),
        "elapsed_min": round(elapsed / 60, 1),
        "transport_errors": adapter.transport_errors,
        "program_errors": adapter.program_errors,
    }
    print(
        f"  {label:<34} strict {result['test_strict']:.4f}  loose {result['test_loose']:.4f}  "
        f"hops {result['by_num_hops']}  ${result['spend_usd']:.2f}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--all", action="store_true", help="every results/runs/hover-* with a summary.json")
    parser.add_argument("--candidate-index", type=int, default=None, help="0 evaluates the SEED candidate (base reference)")
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hover")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()

    import importlib.util
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from gepa_taxonomy.bedrock import require_credentials
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever

    runs = list(args.run)
    if args.all:
        runs += sorted(p for p in (REPO / "results" / "runs").glob("hover-*") if (p / "summary.json").exists())
    if not runs:
        raise SystemExit("nothing to evaluate: pass --run or --all")

    require_credentials()

    spec = importlib.util.spec_from_file_location("_hover_runner", REPO / "scripts" / "run_hover_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    test = runner.load_instances(args.manifests / "test.json")
    print(f"held-out test: {len(test)} claims, hop mix {dict(sorted(Counter(i.task.num_hops for i in test).items()))}")
    print(f"evaluating {len(runs)} candidate(s)\n")

    retriever = WikiRetriever(k=args.k).load()
    results = [r for r in (evaluate_candidate(Path(p), args, test, retriever) for p in runs) if r]
    if not results:
        raise SystemExit("no candidates evaluated")

    out = args.out or REPO / "results" / "test_eval" / f"hover_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "hover",
        "n": len(test),
        "candidates": results,
        "spend_usd": round(sum(r["spend_usd"] for r in results), 4),
        "transport_errors": sum(r["transport_errors"] for r in results),
        "program_errors": sum(r["program_errors"] for r in results),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  total spend : ${payload['spend_usd']:.2f}")
    print(f"  written     : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
