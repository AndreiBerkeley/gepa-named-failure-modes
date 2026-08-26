#!/usr/bin/env python
"""Run one GEPA seed on HoVer. **This spends money.**

    # baseline arm
    PYTHONUTF8=1 uv run python scripts/run_hover_seed.py --seed 1 --budget 60 --workers 8
    # taxonomy arm
    PYTHONUTF8=1 uv run python scripts/run_hover_seed.py --seed 1 --budget 60 --workers 8 \
        --taxonomy results/taxonomy/hover_v1/taxonomy.json

The baseline arm runs the pinned gepa release UNMODIFIED. The only addition is
the dollar-budget stopper, which observes spend and touches nothing else -- no
influence on candidate selection, reflection, sampling or scheduling.

The taxonomy arm leaves the benchmark adapter unchanged. An optimizer-side
hook reviews the current minibatch trajectories after the adapter has built its
normal reflection records and before GEPA asks for a proposal.

Minibatch
---------
Default 6. On an all-or-nothing metric the subsample score is a count out of the
minibatch, so its resolution IS the minibatch size: at 3 the acceptance gate can
only see 0/3, 1/3, 2/3, 3/3 and ties constantly. 6 doubles that resolution for
6 extra rollouts per iteration, which is cheap next to the 300-instance val
evaluation that dominates each accepted iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_instances(manifest_path: Path, pool_path: Path | None = None):
    """Manifest ids -> Instances, in MANIFEST ORDER.

    Order is load-bearing and this is the single definition of it: gepa keys val
    subscores and the Pareto frontier by POSITION, so two loaders that disagreed
    about ordering would silently mis-attach scores (F014).
    """
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from gepa_taxonomy.hover.tasks import instance_from_record

    pool_path = pool_path or REPO / "data" / "hover" / "hover_dev_release_v1.1.json"
    records = {str(r["uid"]): r for r in json.loads(pool_path.read_text(encoding="utf-8"))}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing = [i for i in manifest["example_ids"] if i not in records]
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids are not in the pool, e.g. {missing[:3]}")
    return [instance_from_record(records[i]) for i in manifest["example_ids"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=float, required=True, help="dollar ceiling for this seed")
    parser.add_argument("--taxonomy", type=Path, help="taxonomy.json; presence selects the treatment arm")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hover")
    parser.add_argument("--base-val-cache", type=Path, default=REPO / "results" / "hover_base_val" / "base_val_cache.json")
    parser.add_argument("--minibatch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--reflection-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-transport-errors", type=int, default=25)
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=20000,
        help="hard cap on rollouts, independent of spend. NOT redundant with the dollar "
        "budget: when calls fail to reach the model they cost nothing, so the spend "
        "stopper can never fire and the run loops forever (F063).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="deliberately resume from an existing gepa_state.bin. Without this the run "
        "REFUSES to start when state exists, because gepa resumes silently and would "
        "inherit stale results (F029). Resume is NOT cost-continuous (F064).",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "src"))

    import gepa

    from gepa_taxonomy.bedrock import BedrockLM, MeteredReflectionLM, require_credentials
    from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.hover.adapter import HoverAdapter, instances_by_id
    from gepa_taxonomy.hover.program import COMPONENTS, SEED_CANDIDATE, HoverMultiHopProgram
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    arm = "taxonomy" if args.taxonomy else "baseline"
    out = args.out or REPO / "results" / "runs" / f"hover-{arm}-seed{args.seed}"
    state = out / "gepa_state.bin"
    if state.exists() and not args.resume:
        raise SystemExit(
            f"REFUSING: {state} already exists.\n"
            f"gepa would silently RESUME from it, inheriting that run's scores and spend.\n"
            f"  start clean:            move or delete {out}\n"
            f"  or resume deliberately: --resume"
        )
    out.mkdir(parents=True, exist_ok=True)
    require_credentials()

    train = load_instances(args.manifests / "train.json")
    val = load_instances(args.manifests / "val.json")
    print(f"arm       : {arm}   seed {args.seed}   budget ${args.budget:.2f}")
    print(f"splits    : train {len(train)} | val {len(val)}   minibatch {args.minibatch_size}")

    solver_meter = CostMeter(spend_log=out / "spend.solver.json")
    reflection_meter = CostMeter(spend_log=out / "spend.reflection.json")
    judge_meter = CostMeter(spend_log=out / "spend.judge.json")

    program = HoverMultiHopProgram(
        retriever=WikiRetriever(k=args.k).load(),
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=solver_meter,
        model=args.solver_model,
        k=args.k,
    )

    seed_cache = None
    if args.base_val_cache.exists():
        seed_cache = SeedEvaluationCache.load(args.base_val_cache)
        if not seed_cache.matches(dict(SEED_CANDIDATE)):
            raise SystemExit(
                f"base-val cache at {args.base_val_cache} was built for a DIFFERENT seed candidate.\n"
                "Replaying it would start this run from someone else's state. Rebuild it."
            )
        seed_cache.assert_covers(i.task.example_id for i in val)
        print(f"base-val  : {len(seed_cache.entries)} val instances replayed (no spend)")
    else:
        print(f"base-val  : NONE at {args.base_val_cache} -- this seed will re-evaluate the base candidate")
        print("            and will NOT share a starting state with other seeds (D009).")

    adapter = HoverAdapter(
        program=program,
        instances=instances_by_id([*train, *val]),
        # Gold titles may be named in feedback for TRAIN only. Naming them on a
        # val rollout would hand the retriever the documents it is scored on.
        reflection_gold_ids=frozenset(i.task.example_id for i in train),
        max_workers=args.workers,
        max_transport_errors=args.max_transport_errors,
        seed_cache=seed_cache,
    )

    meters = [solver_meter, reflection_meter]
    taxonomy = None
    taxonomy_feedback = None
    if args.taxonomy:
        import inspect

        from failure_taxonomy import JudgeCache, LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

        if "reflective_dataset_enricher" not in inspect.signature(gepa.optimize).parameters:
            raise SystemExit(
                "taxonomy runs require GEPA's optimizer-side reflective_dataset_enricher hook. "
                "Apply patches/gepa-reflective-dataset-enricher.patch to the pinned GEPA checkout."
            )

        taxonomy = load_taxonomy(args.taxonomy)
        judge_lm = BedrockLM(model=args.reflection_model, max_retries=args.max_retries)

        judge_prompt_log = out / "judge_prompts.jsonl"

        def judge_call(prompt: str) -> str:
            text, tin, tout = judge_lm.complete(prompt, max_tokens=4096)
            judge_meter.record(
                model=args.reflection_model, input_tokens=tin, output_tokens=tout, phase="optimization"
            )
            try:  # archive; loss must never cost the run
                with judge_prompt_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"prompt": prompt, "response": text}) + "\n")
            except OSError:
                pass
            return text

        taxonomy_feedback = TaxonomyFeedbackEnricher(
            judge=LLMFailureJudge(
                taxonomy=taxonomy,
                lm=judge_call,
                cache=JudgeCache.open(out / "judge_cache.jsonl"),
                max_workers=args.judge_workers,
            ),
        )
        # Judge spend competes for the SAME budget as rollouts and reflection, so
        # the treatment arm buys fewer iterations for the same money. That trade
        # is what the comparison measures, not a confound (D032).
        meters.append(judge_meter)
        print(f"taxonomy  : {args.taxonomy} ({len(taxonomy)} codes, fingerprint {taxonomy.fingerprint})")

    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=args.reflection_model, max_retries=args.max_retries),
        meter=reflection_meter,
        model=args.reflection_model,
        spend_log=out / "reflection_spend.jsonl",
        # Raw prompt/response archive: nothing else persists reflection bodies,
        # and auditing injection without them takes triangulation (D074).
        prompt_log=out / "reflection_prompts.jsonl",
    )

    stopper = MaxTotalCostStopper(budget_usd=args.budget, meters=meters)
    print(f"models    : solver {args.solver_model}\n            reflect {args.reflection_model}\n", flush=True)

    started = time.time()
    optimize_kwargs: dict[str, object] = {}
    if taxonomy_feedback is not None:
        optimize_kwargs["reflective_dataset_enricher"] = taxonomy_feedback

    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.minibatch_size,
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=[stopper],
        run_dir=str(out),
        seed=args.seed,
        display_progress_bar=True,
        track_best_outputs=True,
        **optimize_kwargs,
    )
    elapsed = time.time() - started
    # Final exact snapshot: the periodic flush can lag by up to flush_every calls.
    for _m in (m for m in (locals().get('solver_meter'), reflection_meter, judge_meter) if m is not None):
        _m.flush()

    scores = list(getattr(result, "val_aggregate_scores", []) or [])
    summary = {
        "benchmark": "hover",
        "arm": arm,
        "seed": args.seed,
        "budget_usd": args.budget,
        "minibatch_size": args.minibatch_size,
        "components": list(COMPONENTS),
        "elapsed_hours": round(elapsed / 3600, 3),
        "best_val_score": max(scores) if scores else None,
        "base_val_score": scores[0] if scores else None,
        "best_candidate_index": scores.index(max(scores)) if scores else None,
        "candidates": len(getattr(result, "candidates", []) or []),
        "val_aggregate_scores": scores,
        "taxonomy": str(args.taxonomy) if args.taxonomy else None,
        "taxonomy_fingerprint": taxonomy.fingerprint if taxonomy else None,
        "spend": {
            "solver": solver_meter.snapshot(),
            "reflection": reflection_meter.snapshot(),
            "judge": judge_meter.snapshot() if args.taxonomy else None,
            "realised_usd": round(stopper.realised_usd, 6),
            "budget_fired_at_usd": stopper.fired_at_usd,
        },
        "adapter": adapter.summary(),
        "taxonomy_feedback": taxonomy_feedback.summary() if taxonomy_feedback is not None else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(
        json.dumps(getattr(result, "candidates", []), indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\n  base val   : {summary['base_val_score']}")
    print(f"  best val   : {summary['best_val_score']}  (candidate {summary['best_candidate_index']})")
    print(f"  candidates : {summary['candidates']}")
    print(f"  spend      : ${stopper.realised_usd:.2f} of ${args.budget:.2f}")
    print(f"  elapsed    : {summary['elapsed_hours']}h")
    print(f"  summary    : {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONUTF8") != "1":
        raise SystemExit(
            "set PYTHONUTF8=1. gepa writes its run log with the platform default encoding, "
            "and a proposed prompt containing a non-cp1252 character kills the run on Windows (F031)."
        )
    raise SystemExit(main())
