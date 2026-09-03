#!/usr/bin/env python
"""Evaluate the base candidate on val ONCE. **This spends money.**

Two artifacts from one pass, which is why this is not wasted spend:

1. ``base_val_cache.json`` -- the replay store every seed and both arms read, so
   every run starts from **byte-identical** state. Without it each run
   re-samples the base candidate, and the two arms at one seed would differ by an
   independent 300-rollout draw as well as by the treatment.

2. ``base_val.traces.jsonl`` -- the segmented traces the taxonomy is generated
   from. The base candidate's val evaluation IS the taxonomy-generation
   trace source, so this pass had to happen regardless.

Replayed rollouts issue no LM call, so they contribute no spend -- satisfying the
budget exclusion for the shared seed evaluation by construction.

    PYTHONUTF8=1 uv run python demo/build_ifbench_base_val.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "ifbench_base_val"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifests", type=Path, default=REPO / "demo" / "manifests")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--solver-model", default="gpt-5-mini")
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="MODEL=IN,OUT",
        help="price for a model litellm's table does not know, in USD per million input,output tokens; repeatable",
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args()
    from gepa_taxonomy.cost import assert_priced, load_price_overrides

    load_price_overrides(args.price)
    assert_priced(args.solver_model)

    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.ifbench.adapter import IFBenchAdapter, instances_by_id
    from gepa_taxonomy.ifbench.grading import Grade, report
    from gepa_taxonomy.ifbench.program import SEED_CANDIDATE, GenerateEnsureProgram
    from gepa_taxonomy.lm import MeteredLM
    from gepa_taxonomy.progress import report_rollouts
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    cache_path = args.out / "base_val_cache.json"
    if cache_path.exists() and not args.force:
        cached = SeedEvaluationCache.load(cache_path)
        print(f"base val already built: {cache_path} ({len(cached.entries)} instances)")
        print("nothing to do. Pass --force to rebuild (this re-spends and changes every run's start state).")
        return 0

    # Imported from the runner so there is ONE definition of how a manifest
    # becomes instances -- two loaders would be two chances to disagree about
    # ordering, which gepa keys on positionally.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ifb_runner", REPO / "demo" / "run_ifbench_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    val = runner.load_instances(args.manifests / "val.json")
    print(f"base candidate on {len(val)} val instances, {args.workers} workers")

    # Created BEFORE the evaluation so the run directory exists, and the pass is
    # observable, while it is in progress; everything else writes on completion.
    args.out.mkdir(parents=True, exist_ok=True)

    meter = CostMeter()
    program = GenerateEnsureProgram(
        lm=MeteredLM(model=args.solver_model, max_retries=args.max_retries),
        meter=meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )
    adapter = IFBenchAdapter(program=program, instances=instances_by_id(val), max_workers=args.workers)

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
    if adapter.verifier_errors:
        raise SystemExit(
            f"REFUSING TO WRITE: {adapter.verifier_errors} constraint verifiers raised.\n"
            "A verifier that raises is scored as NOT FOLLOWED, so freezing it in would\n"
            "silently depress one constraint class in both arms. The vendored verifiers\n"
            "were checked to run clean, so this means something\n"
            "changed upstream -- refresh with demo/vendor_ifbench.py and inspect."
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
    trace_summary = trace_report(traces)

    grades = [
        Grade(
            score=t["grading"]["score"],
            all_followed=t["grading"]["all_followed"],
            followed=tuple(t["grading"]["followed"]),
            failed_ids=tuple(t["grading"]["failed_ids"]),
            loose_score=t["grading"]["loose_score"],
        )
        for t in batch.trajectories
    ]
    metrics = report(grades, [i.gold for i in val])

    # How often the ensure stage rewrote the draft. If ~0 the second module is
    # decorative and the two-module design is not earning its cost; if ~1 it is
    # rewriting everything. Either extreme is worth knowing before six runs.
    changed = sum(1 for t in batch.trajectories if (t.get("draft") or "").strip() != (t.get("response") or "").strip())

    summary = {
        "n": len(val),
        "mean_score": metrics["instruction_level_strict"],
        "prompt_level_strict": metrics["prompt_level_strict"],
        "instruction_level_loose": metrics["instruction_level_loose"],
        "by_family": metrics["by_family"],
        "ensure_changed_draft": changed,
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "usd_per_rollout": round(meter.total_usd / max(1, len(val)), 5),
        "candidate_fingerprint": cache.candidate_fingerprint,
        "transport_errors": adapter.transport_errors,
        "program_errors": adapter.program_errors,
        "verifier_errors": adapter.verifier_errors,
        "trace_report": trace_summary,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n  instruction-level strict : {metrics['instruction_level_strict']:.1%}   <- what GEPA selects on")
    print(f"  prompt-level strict      : {metrics['prompt_level_strict']:.1%}")
    print(f"  instruction-level loose  : {metrics['instruction_level_loose']:.1%}   (gap = formatting, not compliance)")
    print("  by constraint family:")
    for family, stats in metrics["by_family"].items():
        print(f"    {family:<12} n={stats['n']:<4} {stats['mean']:.1%}")
    print(f"  ensure changed draft     : {changed}/{len(val)}")
    print(f"  elapsed                  : {elapsed / 60:.1f} min")
    print(f"  spend                    : ${meter.total_usd:.4f}  (${summary['usd_per_rollout']:.5f}/rollout)")
    print(f"  traces                   : {trace_summary['segmented']}/{trace_summary['traces']} segmented")
    print(f"\n  cache  : {cache_path}")
    print(f"  traces : {args.out / 'base_val.traces.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
