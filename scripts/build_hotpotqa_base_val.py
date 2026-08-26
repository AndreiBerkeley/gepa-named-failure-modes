#!/usr/bin/env python
"""Evaluate the base candidate on val ONCE. **This spends money (~$1.75).**

Two artifacts from one pass, which is why this is not wasted spend:

1. ``base_val_cache.json`` -- the replay store every seed and both arms read, so
   all six runs start from **byte-identical** state (D009). Without it each run
   re-samples the base candidate: two runs of the identical candidate on the
   identical val set measured 56.0% and 56.5%, so the baseline and taxonomy arms
   at the same seed would differ by an independent 300-rollout draw as well as by
   the treatment -- noise injected into precisely the comparison the experiment
   exists to make.

2. ``base_val.traces.jsonl`` -- the segmented traces the taxonomy is generated
   from (D025). The base candidate's val evaluation IS the taxonomy-generation
   trace source, so this run had to happen regardless; doing it here means it
   happens once instead of six times.

Replayed rollouts issue no LM call, so they contribute no spend -- which is how
the budget exclusion for the shared seed evaluation (D013a) is satisfied by
construction rather than by special-casing the stopper.

    PYTHONUTF8=1 uv run python scripts/build_hotpotqa_base_val.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "base_val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hotpotqa")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args()

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hotpotqa.adapter import HotpotQAAdapter, instances_by_id
    from gepa_taxonomy.hotpotqa.program import SEED_CANDIDATE, MultiHopProgram
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    cache_path = args.out / "base_val_cache.json"
    if cache_path.exists() and not args.force:
        cached = SeedEvaluationCache.load(cache_path)
        print(f"base val already built: {cache_path} ({len(cached.entries)} instances)")
        print("nothing to do. Pass --force to rebuild (this re-spends and changes every run's start state).")
        return 0

    require_credentials()

    # Imported here so the run script's loader is the single definition of how a
    # manifest becomes instances -- two loaders would be two chances to disagree
    # about ordering, which gepa keys on positionally (F014).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_runner", REPO / "scripts" / "run_hotpotqa_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    val = runner.load_instances(args.manifests / "val.json")
    print(f"base candidate on {len(val)} val instances, {args.workers} workers")

    meter = CostMeter()
    program = MultiHopProgram(
        retriever=WikiRetriever(k=args.k).load(),
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        k=args.k,
    )
    adapter = HotpotQAAdapter(program=program, instances=instances_by_id(val), max_workers=args.workers)

    started = time.time()
    batch = adapter.evaluate(val, dict(SEED_CANDIDATE), capture_traces=True)
    elapsed = time.time() - started

    if adapter.transport_errors:
        raise SystemExit(
            f"REFUSING TO WRITE: {adapter.transport_errors} rollouts failed to reach the model.\n"
            "Those score 0.0, and freezing them into the shared starting state would "
            "corrupt every seed and both arms. Rerun with fewer --workers."
        )

    entries = {
        trace["example_id"]: {"score": score, "trace": trace}
        for trace, score in zip(batch.trajectories, batch.scores, strict=True)
    }
    cache = SeedEvaluationCache.build(dict(SEED_CANDIDATE), entries)
    args.out.mkdir(parents=True, exist_ok=True)
    cache.save(cache_path)

    # The taxonomy-generation trace source (D025), segmented by component name so
    # a generator is handed the real structure rather than recovering it from
    # prose -- which is how role discovery previously invented agents (F018).
    from failure_taxonomy import harvest_traces, trace_report, write_generation_traces

    traces = harvest_traces(batch, instance_ids=[t["example_id"] for t in batch.trajectories])
    write_generation_traces(traces, args.out / "base_val.traces.jsonl")
    report = trace_report(traces)

    mean = statistics.mean(batch.scores)
    recall = statistics.mean(t["grading"]["retrieval_recall"] for t in batch.trajectories)
    summary = {
        "n": len(val),
        "mean_answer_f1": mean,
        "mean_retrieval_recall": recall,
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "candidate_fingerprint": cache.candidate_fingerprint,
        "transport_errors": adapter.transport_errors,
        "program_errors": adapter.program_errors,
        "trace_report": report,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n  base val F1      : {mean:.1%}")
    print(f"  retrieval recall : {recall:.1%}")
    print(f"  elapsed          : {elapsed / 60:.1f} min")
    print(f"  spend            : ${meter.total_usd:.4f}")
    print(
        f"  traces           : {report['segmented']}/{report['traces']} segmented, "
        f"components {list(report['components'])}"
    )
    print(f"\n  cache  : {cache_path}")
    print(f"  traces : {args.out / 'base_val.traces.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
