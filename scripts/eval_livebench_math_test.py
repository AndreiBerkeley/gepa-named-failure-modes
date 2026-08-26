#!/usr/bin/env python
"""Evaluate a finished seed's BEST candidate on the held-out test split.

**This spends money (~$1.10 per candidate at the measured rate).**

Test is touched exactly once per candidate, at the end. val drives selection, so
a val score is an optimistically biased estimate of it -- the SWE-Bench round
lost 7.7pp between the two (21.7% val -> 14.0% test), which is the whole reason
this is a separate script and a separate split.

    PYTHONUTF8=1 uv run python scripts/eval_livebench_math_test.py \
        --run results/runs/livebench-math-baseline-seed1

    # every finished run
    PYTHONUTF8=1 uv run python scripts/eval_livebench_math_test.py --all
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def evaluate_run(run: Path, args) -> dict | None:
    from gepa_taxonomy.bedrock import BedrockLM
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.livebench_math.adapter import LiveBenchMathAdapter, instances_by_id
    from gepa_taxonomy.livebench_math.program import SEED_CANDIDATE, SolveReviewProgram

    summary_path = run / "summary.json"
    if not summary_path.exists():
        print(f"  {run.name}: no summary.json -- run unfinished, skipping")
        return None

    out_path = run / "test_eval.json"
    if out_path.exists() and not args.force:
        print(f"  {run.name}: already evaluated ({out_path.name}); --force to redo")
        return json.loads(out_path.read_text(encoding="utf-8"))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = json.loads((run / "candidates.json").read_text(encoding="utf-8"))
    index = summary.get("best_candidate_index")
    if index is None:
        print(f"  {run.name}: no best_candidate_index, skipping")
        return None
    candidate = candidates[index]

    import importlib.util

    spec = importlib.util.spec_from_file_location("_lbm_runner", REPO / "scripts" / "run_livebench_math_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    test = runner.load_instances(args.manifests / "test.json")

    meter = CostMeter()
    program = SolveReviewProgram(
        lm=BedrockLM(model=args.solver_model, max_retries=8),
        meter=meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )
    adapter = LiveBenchMathAdapter(program=program, instances=instances_by_id(test), max_workers=args.workers)

    started = time.time()
    batch = adapter.evaluate(test, candidate, capture_traces=True)
    elapsed = time.time() - started

    if adapter.transport_errors:
        print(f"  {run.name}: {adapter.transport_errors} transport errors -- REFUSING to write a corrupted test score")
        return None

    by_scorer: dict[str, list[float]] = collections.defaultdict(list)
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        by_scorer[trace["grading"]["scorer"]].append(score)

    payload = {
        "run": run.name,
        "arm": summary.get("arm"),
        "seed": summary.get("seed"),
        "best_candidate_index": index,
        "n": len(test),
        "test_score": statistics.mean(batch.scores),
        "val_score": summary.get("best_val_score"),
        "by_scorer": {k: {"n": len(v), "mean": statistics.mean(v)} for k, v in sorted(by_scorer.items())},
        # The per-instance vector, so a paired test across arms is possible later
        # without re-spending. Wilcoxon needs the pairs, not the means.
        "per_instance": {t["example_id"]: s for t, s in zip(batch.trajectories, batch.scores, strict=True)},
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "is_seed_candidate": candidate == dict(SEED_CANDIDATE),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    drop = (payload["val_score"] or 0) - payload["test_score"]
    print(
        f"  {run.name}: test {payload['test_score']:.4f}  val {payload['val_score']:.4f}  "
        f"(val-test {drop:+.4f})  ${meter.total_usd:.2f}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="every livebench-math run with a summary.json")
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "livebench_math")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from gepa_taxonomy.bedrock import require_credentials

    require_credentials()

    if args.all:
        runs = sorted((REPO / "results" / "runs").glob("livebench-math-*"))
    elif args.run:
        runs = [args.run]
    else:
        raise SystemExit("pass --run PATH or --all")

    results = [r for r in (evaluate_run(run, args) for run in runs) if r]
    if len(results) > 1:
        print("\n  arm/seed summary:")
        for r in sorted(results, key=lambda x: (x["arm"] or "", x["seed"] or 0)):
            print(f"    {r['arm']:<9} seed {r['seed']}: test {r['test_score']:.4f}  val {r['val_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
