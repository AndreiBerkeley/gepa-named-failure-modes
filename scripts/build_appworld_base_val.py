#!/usr/bin/env python
"""Evaluate the AppWorld base candidate on val ONCE. **This spends money.**

Same purpose as the HotpotQA equivalent, for the same reason: without it
every run re-samples the starting state, so the baseline and taxonomy arms at
the same seed would differ by an independent draw as well as by the treatment.
A multi-step ReAct agent is *more* stochastic than a fixed four-call chain, not
less, so the argument is stronger here.

Two artifacts from one pass:

1. ``base_val_cache.json`` -- replayed by every seed and both arms, so all runs
   start byte-identically. A replayed rollout starts no environment and makes no
   LM call, so it costs nothing.
2. ``base_val.traces.jsonl`` -- the segmented traces the taxonomy is generated
   from, with the real component name attached rather than recovered from
   prose.

    PYTHONUTF8=1 uv run python scripts/build_appworld_base_val.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "appworld_base_val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "appworld")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--base-url", default="http://localhost:8123")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import importlib.util

    from gepa_taxonomy.appworld.adapter import AppWorldAdapter, client_factory
    from gepa_taxonomy.appworld.program import ReActProgram
    from gepa_taxonomy.appworld.prompts import SEED_CANDIDATE
    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.progress import report_rollouts
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    cache_path = args.out / "base_val_cache.json"
    if cache_path.exists() and not args.force:
        cached = SeedEvaluationCache.load(cache_path)
        print(f"already built: {cache_path} ({len(cached.entries)} tasks). --force to rebuild.")
        return 0

    require_credentials()

    # The run script owns server startup and manifest loading; reusing it keeps
    # one definition of each rather than two that can drift apart.
    spec = importlib.util.spec_from_file_location("_runner", REPO / "scripts" / "run_appworld_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ensure_servers(args.port, args.workers)

    val = runner.load_task_ids(args.manifests / "val.json")
    print(f"base candidate on {len(val)} val tasks, {args.workers} workers")

    meter = CostMeter()
    adapter = AppWorldAdapter(
        program=ReActProgram(
            client=None,
            lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
            meter=meter,
            model=args.solver_model,
            max_steps=args.max_steps,
        ),
        client_factory=client_factory(args.port, args.workers, prefix="baseval"),
        max_workers=args.workers,
    )

    started = time.time()
    stop_progress = report_rollouts(adapter, len(val))
    batch = adapter.evaluate(val, dict(SEED_CANDIDATE), capture_traces=True)
    stop_progress.set()
    elapsed = time.time() - started

    if adapter.transport_errors:
        raise SystemExit(
            f"REFUSING TO WRITE: {adapter.transport_errors} rollouts failed to reach the model "
            "or the server. Those score 0.0, and freezing them into the shared starting state "
            "would corrupt every seed and both arms. Rerun with fewer --workers."
        )

    entries = {
        trace["task_id"]: {
            "score": score,
            "trace": trace,
            "grading": trace.get("grading") or {},
        }
        for trace, score in zip(batch.trajectories, batch.scores, strict=True)
    }
    cache = SeedEvaluationCache.build(dict(SEED_CANDIDATE), entries)
    args.out.mkdir(parents=True, exist_ok=True)
    cache.save(cache_path)

    from failure_taxonomy import harvest_traces, trace_report, write_generation_traces

    traces = harvest_traces(batch, instance_ids=[t["task_id"] for t in batch.trajectories])
    write_generation_traces(traces, args.out / "base_val.traces.jsonl")
    report = trace_report(traces)

    mean = statistics.mean(batch.scores)
    tgc = statistics.mean(1.0 if t["grading"]["success"] else 0.0 for t in batch.trajectories)
    summary = {
        "n": len(val),
        "mean_score": mean,
        "task_goal_completion": tgc,
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "candidate_fingerprint": cache.candidate_fingerprint,
        "adapter": adapter.summary(),
        "trace_report": report,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n  mean score (selection) : {mean:.1%}")
    print(f"  TGC (reported)         : {tgc:.1%}")
    print(f"  mean steps             : {adapter.summary()['mean_steps']}")
    print(f"  step exhaustions       : {adapter.summary()['step_exhaustions']}")
    print(f"  elapsed                : {elapsed / 60:.1f} min")
    print(f"  spend                  : ${meter.total_usd:.4f}")
    print(f"  traces                 : {report['segmented']}/{report['traces']} segmented")
    print(f"\n  cache  : {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
