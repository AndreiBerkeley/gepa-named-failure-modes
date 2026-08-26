#!/usr/bin/env python
"""CloudCast broadcast routing through ``optimize_anything``, one arm at a time.

**This spends money.** Reflection is an LLM call; the broadcast simulator runs
locally and costs nothing.

Replicates ``examples/adrs/cloudcast/main.py`` from the pinned gepa clone,
changing exactly two things:

* the reflection LM is ours (Bedrock, metered) instead of ``gemini-3-pro-preview``;
* the evaluator is wrapped by :class:`ArmedEvaluator`, which decides what
  ``side_info`` reaches reflection.

Everything else -- the seed program, objective, background, dataset,
``minibatch_size``, ``skip_perfect_score=False`` -- is theirs. W&B tracking is
dropped: it is orthogonal to the comparison and would need credentials.

Why this benchmark, alongside circle packing
--------------------------------------------
Deliberately the *unlike* half of the pair. Circle packing is Single-Task with
no dataset and mechanical failures (overlap, out-of-bounds, exception) that a
hand-written diagnostic can enumerate exhaustively. CloudCast is Multi-Task over
cloud configurations, and its failures are semantic -- provider-blind routing,
bad partitioning, cost/time trade-offs mispriced. A taxonomy that matches
hand-authored ASI on *both* has been tested where enumeration is easy and where
it is not.

It also produces many traces per candidate (one per configuration) rather than
one, which exercises a second trace-harvest shape.

Order of operations is the same as circle packing: ``--arm stock`` first (it is
the only arm that writes traces), then generate the taxonomy from those traces,
then ``--arm taxonomy`` and ``--arm score``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEPA_ROOT = REPO.parent / "gepa-v0.1.4"
CLOUDCAST = GEPA_ROOT / "examples" / "adrs" / "cloudcast"


def _load_cloudcast_main():
    """Import their ``main.py`` by path.

    It is written to be run as a script: ``from utils.lm import ...`` resolves
    only because Python puts the script's own directory on ``sys.path[0]``.
    Importing it as ``examples.adrs.cloudcast.main`` would therefore fail on
    that bare ``utils``, so the directory goes on the path first and the module
    is loaded by location.
    """
    sys.path.insert(0, str(CLOUDCAST))
    spec = importlib.util.spec_from_file_location("_cloudcast_main", CLOUDCAST / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["stock", "taxonomy", "score"])
    parser.add_argument("--budget", type=float, required=True, help="dollar ceiling for this run")
    parser.add_argument("--taxonomy", type=Path, help="taxonomy.json; required for --arm taxonomy")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=100, help="their CLI default")
    parser.add_argument(
        "--max-candidate-proposals",
        type=int,
        default=None,
        help="native depth cap (optimize_anything.py:1416). Set on the taxonomy and score "
        "arms to the STOCK arm's candidate count so all three compare at equal depth.",
    )
    parser.add_argument("--minibatch-size", type=int, default=3, help="their CLI default")
    # See run_circle_packing.py: Opus 4.6 is the strongest invocable model on
    # this key (probe, 2026-08-15). Judge stays on Sonnet 4.6.
    parser.add_argument("--reflection-model", default="us.anthropic.claude-opus-4-6-v1")
    parser.add_argument("--judge-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=120.0,
        help="seconds before an evaluation is abandoned and scored 0. CloudCast imports "
        "the evolved program and calls it IN-PROCESS, guarded only by `except Exception` -- "
        "and a non-terminating loop is not an exception. Without this a candidate that "
        "fails to terminate hangs the run silently: no error, no output, and no spend, so "
        "the dollar-budget stopper can never fire either. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="see run_circle_packing.py; resume is not cost-continuous (F064)")
    args = parser.parse_args()

    if args.arm == "taxonomy" and not args.taxonomy:
        raise SystemExit("--arm taxonomy requires --taxonomy; without it the run would silently be score-only")
    if not CLOUDCAST.exists():
        raise SystemExit(f"cloudcast example not found: {CLOUDCAST}")

    sys.path.insert(0, str(GEPA_ROOT))
    sys.path.insert(0, str(REPO / "src"))

    out = args.out or REPO / "results" / "runs" / f"cloudcast-{args.arm}"
    state = out / "gepa_state.bin"
    if state.exists() and not args.resume:
        raise SystemExit(
            f"REFUSING: {state} already exists.\n"
            f"gepa would silently RESUME from it, inheriting that run's scores and spend.\n"
            f"  start clean:            move or delete {out}\n"
            f"  or resume deliberately: --resume"
        )
    out.mkdir(parents=True, exist_ok=True)

    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, require_credentials
    from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
    from gepa_taxonomy.oa import ArmedEvaluator, TraceSink

    cc = _load_cloudcast_main()
    require_credentials()

    dataset = cc.load_config_dataset(config_dir=cc.DATASET_ROOT)
    if not dataset:
        raise SystemExit(f"no configuration files found in {cc.DATASET_ROOT}")

    reflection_meter = CostMeter(spend_log=out / "spend.reflection.json")
    judge_meter = CostMeter(spend_log=out / "spend.judge.json")
    meters = [reflection_meter]

    judge = None
    taxonomy = None
    if args.arm == "taxonomy":
        from failure_taxonomy import JudgeCache, LLMFailureJudge, load_taxonomy

        taxonomy = load_taxonomy(args.taxonomy)
        judge_lm = BedrockLM(model=args.judge_model, max_retries=args.max_retries)

        def judge_call(prompt: str) -> str:
            text, tin, tout = judge_lm.complete(prompt, max_tokens=4096)
            judge_meter.record(model=args.judge_model, input_tokens=tin, output_tokens=tout, phase="optimization")
            return text

        judge = LLMFailureJudge(
            taxonomy=taxonomy,
            lm=judge_call,
            cache=JudgeCache.open(out / "judge_cache.jsonl"),
            max_workers=args.judge_workers,
        )
        meters.append(judge_meter)

    armed = ArmedEvaluator(
        inner=cc.evaluate,
        arm=args.arm,
        judge=judge,
        sink=TraceSink(path=out / "traces.jsonl") if args.arm == "stock" else None,
        task_text=cc.OPTIMIZATION_OBJECTIVE,
        timeout_s=args.eval_timeout or None,
    )

    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=args.reflection_model, max_retries=args.max_retries),
        meter=reflection_meter,
        model=args.reflection_model,
        spend_log=out / "reflection_spend.jsonl",
    )

    stopper = MaxTotalCostStopper(budget_usd=args.budget, meters=meters)

    print(f"arm        : {args.arm}")
    print(f"out        : {out}")
    print(f"dataset    : {len(dataset)} configurations (train = val, their setup)")
    print(f"budget     : ${args.budget:.2f}   max_metric_calls={args.max_metric_calls}  minibatch={args.minibatch_size}")
    print(f"reflection : {args.reflection_model}")
    if taxonomy is not None:
        print(f"taxonomy   : {args.taxonomy} ({len(taxonomy)} codes, fingerprint {taxonomy.fingerprint})")
    if args.arm == "stock":
        print(f"traces     : {out / 'traces.jsonl'}  (input to taxonomy generation)")
    print("", flush=True)

    started = time.time()
    result = optimize_anything(
        seed_candidate={"program": cc.INITIAL_PROGRAM},
        evaluator=armed,
        dataset=dataset,
        valset=dataset,
        objective=cc.OPTIMIZATION_OBJECTIVE,
        background=cc.OPTIMIZATION_BACKGROUND,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(out),
                seed=args.seed,
                max_metric_calls=args.max_metric_calls,
                max_candidate_proposals=args.max_candidate_proposals,
                track_best_outputs=True,
                use_cloudpickle=True,
                display_progress_bar=True,
            ),
            reflection=ReflectionConfig(
                reflection_minibatch_size=args.minibatch_size,
                reflection_lm=reflection_lm,
                skip_perfect_score=False,
            ),
            stop_callbacks=[stopper],
        ),
    )
    elapsed = time.time() - started
    # Final exact snapshot: the periodic flush can lag by up to flush_every calls.
    for _m in (m for m in (locals().get('solver_meter'), reflection_meter, judge_meter) if m is not None):
        _m.flush()

    scores = list(getattr(result, "val_aggregate_scores", []) or [])
    summary = {
        "benchmark": "cloudcast",
        "arm": args.arm,
        "seed": args.seed,
        "budget_usd": args.budget,
        "max_metric_calls": args.max_metric_calls,
        "minibatch_size": args.minibatch_size,
        "dataset_size": len(dataset),
        "reflection_model": args.reflection_model,
        "taxonomy": str(args.taxonomy) if args.taxonomy else None,
        "taxonomy_fingerprint": taxonomy.fingerprint if taxonomy else None,
        "elapsed_hours": round(elapsed / 3600, 3),
        "best_score": max(scores) if scores else None,
        "base_score": scores[0] if scores else None,
        "candidates": len(getattr(result, "candidates", []) or []),
        "val_aggregate_scores": scores,
        "evaluator": armed.summary(),
        "spend": {
            "reflection": reflection_meter.snapshot(),
            "judge": judge_meter.snapshot() if args.arm == "taxonomy" else None,
            "realised_usd": round(stopper.realised_usd, 6),
            "budget_fired_at_usd": stopper.fired_at_usd,
        },
        "reference": {
            "published_gepa_gemini3pro": "40.2% cost savings vs Dijkstra baseline",
            "note": "different model and provider; anchor only, not the comparison",
        },
    }
    if judge is not None:
        summary["judge"] = judge.summary()

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(
        json.dumps(getattr(result, "candidates", []), indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\n  base score : {summary['base_score']}")
    print(f"  best score : {summary['best_score']}")
    print(f"  candidates : {summary['candidates']}")
    print(f"  spend      : ${stopper.realised_usd:.2f} of ${args.budget:.2f}")
    print(f"  elapsed    : {summary['elapsed_hours']}h")
    print(f"  evaluator  : {armed.summary()}")
    print(f"  summary    : {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONUTF8") != "1":
        raise SystemExit("set PYTHONUTF8=1 (F031: a non-cp1252 character in a proposal kills the run on Windows).")
    raise SystemExit(main())
