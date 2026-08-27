#!/usr/bin/env python
"""Evaluate the base candidate on val ONCE. **This spends money (~$1.10 est).**

Two artifacts from one pass, which is why this is not wasted spend:

1. ``base_val_cache.json`` -- the replay store every seed and both arms read, so
   all six runs start from **byte-identical** state. Without it each run
   re-samples the base candidate, and the two arms at one seed would differ by an
   independent 90-rollout draw as well as by the treatment -- noise injected into
   precisely the comparison the experiment exists to make.

2. ``base_val.traces.jsonl`` -- the segmented traces the taxonomy is generated
   from. The base candidate's val evaluation IS the taxonomy-generation
   trace source, so this pass had to happen regardless; doing it here means it
   happens once instead of six times.

Replayed rollouts issue no LM call, so they contribute no spend -- satisfying the
budget exclusion for the shared seed evaluation by construction rather
than by special-casing the stopper.

    PYTHONUTF8=1 uv run python scripts/build_livebench_math_base_val.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "livebench_math_base_val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "livebench_math")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args()

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.livebench_math.adapter import LiveBenchMathAdapter, instances_by_id
    from gepa_taxonomy.livebench_math.program import SEED_CANDIDATE, SolveReviewProgram
    from gepa_taxonomy.progress import report_rollouts
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    cache_path = args.out / "base_val_cache.json"
    if cache_path.exists() and not args.force:
        cached = SeedEvaluationCache.load(cache_path)
        print(f"base val already built: {cache_path} ({len(cached.entries)} instances)")
        print("nothing to do. Pass --force to rebuild (this re-spends and changes every run's start state).")
        return 0

    require_credentials()

    # Imported from the runner so there is ONE definition of how a manifest
    # becomes instances -- two loaders would be two chances to disagree about
    # ordering, which gepa keys on positionally.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_lbm_runner", REPO / "scripts" / "run_livebench_math_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    val = runner.load_instances(args.manifests / "val.json")
    print(f"base candidate on {len(val)} val instances, {args.workers} workers")

    # Created BEFORE the evaluation purely so the run is observable: everything
    # else writes on completion, so there was previously no on-disk signal at all
    # while it ran and chain_status.py reported nothing.
    args.out.mkdir(parents=True, exist_ok=True)

    meter = CostMeter()
    program = SolveReviewProgram(
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )
    adapter = LiveBenchMathAdapter(program=program, instances=instances_by_id(val), max_workers=args.workers)

    started = time.time()
    stop_progress = report_rollouts(adapter, len(val))
    batch = adapter.evaluate(val, dict(SEED_CANDIDATE), capture_traces=True)
    stop_progress.set()
    elapsed = time.time() - started

    if adapter.transport_errors:
        raise SystemExit(
            f"REFUSING TO WRITE: {adapter.transport_errors} rollouts failed to reach the model.\n"
            "Those score 0.0, and freezing them into the shared starting state would\n"
            "corrupt every seed and both arms. Rerun with fewer --workers."
        )

    entries = {
        trace["example_id"]: {"score": score, "trace": trace}
        for trace, score in zip(batch.trajectories, batch.scores, strict=True)
    }
    cache = SeedEvaluationCache.build(dict(SEED_CANDIDATE), entries)
    args.out.mkdir(parents=True, exist_ok=True)
    cache.save(cache_path)

    # The taxonomy-generation trace source, segmented by component name so
    # a generator is handed the real structure rather than recovering it from
    # prose -- which is how role discovery previously invented agents.
    from failure_taxonomy import harvest_traces, trace_report, write_generation_traces

    traces = harvest_traces(batch, instance_ids=[t["example_id"] for t in batch.trajectories])
    write_generation_traces(traces, args.out / "base_val.traces.jsonl")
    report = trace_report(traces)

    mean = statistics.mean(batch.scores)
    # Per-scorer means, because the three are not interchangeable: olympiad is
    # the only fractional one, and a headline mean hides which of them moved.
    by_scorer: dict[str, list[float]] = collections.defaultdict(list)
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        by_scorer[trace["grading"]["scorer"]].append(score)

    # How often the review stage changed the draft's answer. If this is ~0 the
    # second module is decorative and the two-module design is not earning its
    # cost; if it is ~1 review is rewriting everything. Either extreme is worth
    # knowing before six runs are launched on top of it.
    changed = sum(
        1
        for t in batch.trajectories
        if (t.get("draft_answer") or "").strip().splitlines()[-1:] != (t.get("answer") or "").strip().splitlines()[-1:]
    )

    summary = {
        "n": len(val),
        "mean_score": mean,
        "by_scorer": {k: {"n": len(v), "mean": statistics.mean(v)} for k, v in sorted(by_scorer.items())},
        "review_changed_answer": changed,
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "usd_per_rollout": round(meter.total_usd / max(1, len(val)), 5),
        "candidate_fingerprint": cache.candidate_fingerprint,
        "transport_errors": adapter.transport_errors,
        "program_errors": adapter.program_errors,
        "trace_report": report,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n  base val score   : {mean:.1%}")
    for scorer, stats in summary["by_scorer"].items():
        print(f"    {scorer:<16} n={stats['n']:<4} mean={stats['mean']:.1%}")
    print(f"  review changed   : {changed}/{len(val)} answers")
    print(f"  elapsed          : {elapsed / 60:.1f} min")
    print(f"  spend            : ${meter.total_usd:.4f}  (${summary['usd_per_rollout']:.5f}/rollout)")
    print(
        f"  traces           : {report['segmented']}/{report['traces']} segmented, components {list(report['components'])}"
    )
    print(f"\n  cache  : {cache_path}")
    print(f"  traces : {args.out / 'base_val.traces.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
