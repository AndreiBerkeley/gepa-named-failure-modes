#!/usr/bin/env python
"""Evaluate the HoVer base candidate on val ONCE. **This spends money.**

Two artifacts from one pass, which is why this is not wasted spend:

1. ``base_val_cache.json`` -- the replay store every seed and both arms read, so
   all six runs start from **byte-identical** state (D009). Without it each run
   re-samples the base candidate and the arms would differ by an independent
   300-rollout draw as well as by the treatment -- noise injected into precisely
   the comparison the experiment exists to make.

2. ``base_val.traces.jsonl`` -- the segmented traces the taxonomy is generated
   from (D025). The base candidate's val evaluation IS the trace source, so this
   pass had to happen regardless; doing it here means once, not six times.

Replayed rollouts issue no LM call, so they contribute no spend -- satisfying
the budget exclusion for the shared seed evaluation (D013a) by construction
rather than by special-casing the stopper.

    PYTHONUTF8=1 uv run python scripts/build_hover_base_val.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "hover_base_val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hover")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "src"))

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.hover.adapter import HoverAdapter, instances_by_id
    from gepa_taxonomy.hover.program import SEED_CANDIDATE, HoverMultiHopProgram
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    cache_path = args.out / "base_val_cache.json"
    if cache_path.exists() and not args.force:
        cached = SeedEvaluationCache.load(cache_path)
        print(f"base val already built: {cache_path} ({len(cached.entries)} instances)")
        print("nothing to do. --force re-spends and changes every run's start state.")
        return 0

    require_credentials()

    # Imported so the run script's loader is the SINGLE definition of how a
    # manifest becomes instances -- two loaders are two chances to disagree
    # about ordering, which gepa keys on positionally (F014).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_hover_runner", REPO / "scripts" / "run_hover_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    val = runner.load_instances(args.manifests / "val.json")
    hops = Counter(i.task.num_hops for i in val)
    print(f"base candidate on {len(val)} val claims, {args.workers} workers")
    print(f"hop mix: {dict(sorted(hops.items()))}")

    meter = CostMeter()
    program = HoverMultiHopProgram(
        retriever=WikiRetriever(k=args.k).load(),
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        k=args.k,
    )
    adapter = HoverAdapter(program=program, instances=instances_by_id(val), max_workers=args.workers)

    started = time.time()
    batch = adapter.evaluate(val, dict(SEED_CANDIDATE), capture_traces=True)
    elapsed = time.time() - started

    if adapter.transport_errors:
        raise SystemExit(
            f"REFUSING TO WRITE: {adapter.transport_errors} rollouts failed to reach the model.\n"
            "Those score 0.0, and freezing them into the shared starting state would "
            "corrupt every seed and both arms. Rerun with fewer --workers.\n"
            f"samples: {adapter.failures.summary().get('error_samples')}"
        )

    entries = {
        trace["example_id"]: {"score": score, "trace": trace}
        for trace, score in zip(batch.trajectories, batch.scores, strict=True)
    }
    cache = SeedEvaluationCache.build(dict(SEED_CANDIDATE), entries)
    args.out.mkdir(parents=True, exist_ok=True)
    cache.save(cache_path)

    from failure_taxonomy import harvest_traces, trace_report, write_generation_traces

    traces = harvest_traces(batch, instance_ids=[t["example_id"] for t in batch.trajectories])
    write_generation_traces(traces, args.out / "base_val.traces.jsonl")
    report = trace_report(traces)

    strict = statistics.mean(batch.scores)
    loose = statistics.mean(t["grading"]["loose_recall"] for t in batch.trajectories)
    by_hop: dict[int, list[float]] = {}
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        by_hop.setdefault(adapter.instances[trace["example_id"]].task.num_hops, []).append(score)

    summary = {
        "benchmark": "hover",
        "n": len(val),
        "strict_retrieval": strict,
        "loose_recall": loose,
        "by_num_hops": {h: round(statistics.mean(v), 4) for h, v in sorted(by_hop.items())},
        "elapsed_hours": round(elapsed / 3600, 3),
        "spend": meter.snapshot(),
        "usd_per_rollout": round(meter.budgeted_usd / max(1, len(val)), 6),
        "traces": report,
        "adapter": adapter.summary(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n  strict retrieval : {strict:.4f}   (all gold documents found)")
    print(f"  loose recall     : {loose:.4f}   reported, never selected on")
    print(f"  by hop count     : {summary['by_num_hops']}")
    print(f"  traces           : {report['segmented']}/{report['traces']} segmented")
    print(f"  spend            : ${meter.budgeted_usd:.2f}  (${summary['usd_per_rollout']:.5f}/rollout)")
    print(f"  elapsed          : {elapsed / 60:.1f} min")
    print(f"\n  cache  : {cache_path}")
    print(f"  traces : {args.out / 'base_val.traces.jsonl'}")
    print(f"\n  projected per seed at $60: ~{60 / max(1e-9, summary['usd_per_rollout'] * 2 * 6 + 0.05):.0f} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
