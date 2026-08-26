#!/usr/bin/env python
"""Circle packing (n=26) through ``optimize_anything``, one arm at a time.

**This spends money.** Reflection and the refiner are LLM calls; the packing
code itself runs locally and costs nothing.

Replicates ``examples/circle_packing/main.py`` from the pinned gepa clone,
changing exactly two things:

* the reflection LM is ours (Bedrock, metered) instead of ``openai/gpt-5``;
* the evaluator is wrapped by :class:`ArmedEvaluator`, which decides what
  ``side_info`` reaches reflection.

Everything else -- seed code, objective, background, ``max_metric_calls=150``,
``frontier_type="objective"``, ``cache_evaluation``, the refiner -- is theirs.

Why their published 2.63598 is not the comparison
-------------------------------------------------
They ran ``openai/gpt-5``. Comparing our taxonomy arm to that number would
confound feedback content with model and provider. The comparison is
``--arm stock`` against ``--arm taxonomy`` on this machine with this model;
their figure is a loose anchor for whether our stock arm is in the right
neighbourhood at all.

Order of operations::

    # 1. baseline + trace harvest
    run_circle_packing.py --arm stock --budget 15
    # 2. taxonomy from those traces (scripts/generate_taxonomy.py)
    # 3. the arm under test
    run_circle_packing.py --arm taxonomy --budget 15 --taxonomy <path>
    # 4. the floor
    run_circle_packing.py --arm score --budget 15

Must run under the gepa clone's own interpreter, which has the example's
dependencies; see ``--help`` output for the exact invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEPA_ROOT = REPO.parent / "gepa-v0.1.4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["stock", "taxonomy", "score"])
    parser.add_argument("--budget", type=float, required=True, help="dollar ceiling for this run")
    parser.add_argument("--taxonomy", type=Path, help="taxonomy.json; required for --arm taxonomy")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=150, help="theirs is 150; keep it to stay comparable")
    parser.add_argument(
        "--max-candidate-proposals",
        type=int,
        default=None,
        help="native depth cap (optimize_anything.py:1416). Set this on the taxonomy and "
        "score arms to the candidate count the STOCK arm reached, so all three are compared "
        "at equal depth. Unlike the HotpotQA/IFBench arms this needs no external stop-file "
        "watcher -- the cap is built into the engine.",
    )
    # Model ids MUST use the `us.` inference-profile prefix. An AWS Organizations
    # SCP (p-uiee3zlq) denies bedrock:InvokeModel on every `global.*`
    # foundation-model ARN outright -- it is an explicit deny, so no key change
    # can lift it. `us.*` reaches the full lineup, Opus 5 included, at a ~10%
    # cross-region premium that cost.py already encodes.
    #
    # Reflection carries this benchmark: their published run used gpt-5 and beat
    # AlphaEvolve by 0.00018, so headroom in the proposer is what matters. The
    # judge stays on Sonnet 4.6 -- classifying a trace against a fixed taxonomy
    # is a far easier job than evolving a 900-line optimiser.
    parser.add_argument("--reflection-model", default="us.anthropic.claude-opus-4-6-v1")
    parser.add_argument("--judge-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="deliberately resume from an existing gepa_state.bin. Without this the run "
        "REFUSES to start when state exists, because gepa resumes silently and would "
        "inherit stale results (F029). Note resume is NOT cost-continuous (F064).",
    )
    args = parser.parse_args()

    if args.arm == "taxonomy" and not args.taxonomy:
        raise SystemExit("--arm taxonomy requires --taxonomy; without it the run would silently be score-only")
    if not GEPA_ROOT.exists():
        raise SystemExit(f"pinned gepa clone not found: {GEPA_ROOT}")

    # The example imports as `examples.circle_packing.*`, so the gepa root must
    # be importable. Our own src too, for failure_taxonomy / gepa_taxonomy.
    sys.path.insert(0, str(GEPA_ROOT))
    sys.path.insert(0, str(REPO / "src"))

    out = args.out or REPO / "results" / "runs" / f"circlepack-{args.arm}"
    state = out / "gepa_state.bin"
    if state.exists() and not args.resume:
        raise SystemExit(
            f"REFUSING: {state} already exists.\n"
            f"gepa would silently RESUME from it, inheriting that run's scores and\n"
            f"spend, and the result would be neither a fresh run nor a clean resume.\n"
            f"  start clean:            move or delete {out}\n"
            f"  or resume deliberately: --resume"
        )
    out.mkdir(parents=True, exist_ok=True)

    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        RefinerConfig,
        ReflectionConfig,
        optimize_anything,
    )

    from examples.circle_packing.main import SEED_CODE, evaluate  # noqa: E402
    from examples.circle_packing.utils import CIRCLE_PACKING_BACKGROUND  # noqa: E402
    from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, require_credentials
    from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
    from gepa_taxonomy.oa import ArmedEvaluator, TraceSink

    require_credentials()

    OBJECTIVE = (
        "Optimize circle packing code to maximize sum of circle radii within a unit square for N=26 circles."
    )

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
            judge_meter.record(
                model=args.judge_model, input_tokens=tin, output_tokens=tout, phase="optimization"
            )
            return text

        judge = LLMFailureJudge(
            taxonomy=taxonomy,
            lm=judge_call,
            cache=JudgeCache.open(out / "judge_cache.jsonl"),
        )
        # Judge spend competes for the SAME dollar budget as reflection, so the
        # taxonomy arm buys fewer iterations for the same money. That trade is
        # part of what the comparison measures, not a confound (D032).
        meters.append(judge_meter)

    armed = ArmedEvaluator(
        inner=evaluate,
        arm=args.arm,
        judge=judge,
        sink=TraceSink(path=out / "traces.jsonl") if args.arm == "stock" else None,
        task_text=OBJECTIVE,
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
    print(f"budget     : ${args.budget:.2f}   max_metric_calls={args.max_metric_calls}")
    print(f"reflection : {args.reflection_model}")
    if taxonomy is not None:
        print(f"taxonomy   : {args.taxonomy} ({len(taxonomy)} codes, fingerprint {taxonomy.fingerprint})")
    if args.arm == "stock":
        print(f"traces     : {out / 'traces.jsonl'}  (input to taxonomy generation)")
    print("", flush=True)

    started = time.time()
    result = optimize_anything(
        seed_candidate=SEED_CODE,
        evaluator=armed,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(out),
                seed=args.seed,
                max_metric_calls=args.max_metric_calls,
                max_candidate_proposals=args.max_candidate_proposals,
                track_best_outputs=True,
                cache_evaluation=True,
                frontier_type="objective",
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
            refiner=RefinerConfig(),
            stop_callbacks=[stopper],
        ),
        objective=OBJECTIVE,
        background=CIRCLE_PACKING_BACKGROUND,
    )
    elapsed = time.time() - started
    # Final exact snapshot: the periodic flush can lag by up to flush_every calls.
    for _m in (m for m in (locals().get('solver_meter'), reflection_meter, judge_meter) if m is not None):
        _m.flush()

    best = max(result.val_aggregate_scores) if getattr(result, "val_aggregate_scores", None) else None
    summary = {
        "benchmark": "circle_packing_n26",
        "arm": args.arm,
        "seed": args.seed,
        "budget_usd": args.budget,
        "max_metric_calls": args.max_metric_calls,
        "reflection_model": args.reflection_model,
        "taxonomy": str(args.taxonomy) if args.taxonomy else None,
        "taxonomy_fingerprint": taxonomy.fingerprint if taxonomy else None,
        "elapsed_hours": round(elapsed / 3600, 3),
        "best_score": best,
        "candidates": len(getattr(result, "candidates", []) or []),
        "val_aggregate_scores": list(getattr(result, "val_aggregate_scores", []) or []),
        "evaluator": armed.summary(),
        "spend": {
            "reflection": reflection_meter.snapshot(),
            "judge": judge_meter.snapshot() if args.arm == "taxonomy" else None,
            "realised_usd": round(stopper.realised_usd, 6),
            "budget_fired_at_usd": stopper.fired_at_usd,
        },
        "reference": {
            "published_gepa_gpt5": 2.63598,
            "alphaevolve": 2.6358,
            "note": "different model and provider; anchor only, not the comparison",
        },
    }
    if judge is not None:
        summary["judge"] = judge.summary()

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(
        json.dumps(getattr(result, "candidates", []), indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\n  best score : {best}")
    print(f"  candidates : {summary['candidates']}")
    print(f"  spend      : ${stopper.realised_usd:.2f} of ${args.budget:.2f}")
    print(f"  elapsed    : {summary['elapsed_hours']}h")
    print(f"  evaluator  : {armed.summary()}")
    print(f"  summary    : {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONUTF8") != "1":
        raise SystemExit(
            "set PYTHONUTF8=1. gepa writes its run log with the platform default "
            "encoding, and a proposed candidate containing a non-cp1252 character "
            "kills the run on Windows (F031)."
        )
    raise SystemExit(main())
