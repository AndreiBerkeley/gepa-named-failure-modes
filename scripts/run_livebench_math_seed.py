#!/usr/bin/env python
"""Run one LiveBench-Math seed. **This spends money.** Andrei launches it.

Baseline arm (no ``--taxonomy``) is unmodified gepa v0.1.4 driving our adapter.
The only addition is the dollar-budget stopper, which observes spend and nothing
else (CLAUDE.md hard rule 2).

Treatment arm (``--taxonomy PATH``) adds an optimizer-side trace review between
the adapter's normal reflective dataset and GEPA's proposal call. The adapter is
identical across arms. With the flag absent, no enricher or judge is constructed.

    # baseline
    PYTHONUTF8=1 uv run python scripts/run_livebench_math_seed.py --seed 1 --budget 30

    # treatment, same budget, same seed
    PYTHONUTF8=1 uv run python scripts/run_livebench_math_seed.py --seed 1 --budget 30 \
        --taxonomy results/taxonomy/livebench_math_v1/taxonomy.pruned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_instances(manifest_path: Path):
    from datasets import load_dataset

    from gepa_taxonomy.livebench_math.tasks import instance_from_record

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = set(manifest["example_ids"])
    ds = load_dataset(manifest["dataset"], split=manifest["dataset_split"])
    by_id = {str(r["question_id"]): instance_from_record(r) for r in ds if str(r["question_id"]) in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids missing from the dataset, e.g. {sorted(missing)[:3]}")
    # Manifest order, which is sorted: gepa keys val subscores and the Pareto
    # frontier POSITIONALLY (F014), so this ordering is load-bearing.
    return [by_id[i] for i in sorted(wanted)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="run seed (1, 2, 3)")
    parser.add_argument(
        "--log-reflection-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append every post-enrichment reflective dataset to "
        "reflection_datasets.jsonl in the run dir, exactly as reflection "
        "consumed it (observability; disable with --no-log-reflection-datasets)",
    )
    parser.add_argument("--budget", type=float, required=True, help="dollar budget for this seed")
    parser.add_argument("--taxonomy", type=Path, default=None, help="enable the treatment arm")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=5,
        help="instances per reflective minibatch. 5 rather than gepa's default 3: "
        "math_comp and aime score 0/1, so a small minibatch ties often, and a tie "
        "is a wasted iteration (gepa accepts on STRICT improvement only).",
    )
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "livebench_math")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--reflection-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="hard cap on rollouts, independent of spend. The dollar budget is the "
        "primary control; this is a backstop for when the per-rollout cost estimate "
        "is wrong -- which it was by 87%% on AppWorld. ~58 metric calls per iteration.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="ceiling per call; billed on tokens produced")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="litellm retries per call. Raised above the usual 3 because an exhausted "
        "retry scores 0.0 -- indistinguishable to the optimizer from a bad candidate.",
    )
    parser.add_argument("--max-transport-errors", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="instances evaluated concurrently. Rollouts are network-bound, so this "
        "is near-linear in this range. Held at 8 to stay clear of the Bedrock rate "
        "limit; raise only if transport_errors stays at 0.",
    )
    parser.add_argument(
        "--base-val-cache",
        type=Path,
        default=REPO / "results" / "livebench_math_base_val" / "base_val_cache.json",
        help="shared base-candidate val evaluation, replayed so every seed and both "
        "arms start from byte-identical state (D009). Pass 'none' to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="deliberately resume from an existing gepa_state.bin. Without this the "
        "run REFUSES to start when state exists, because gepa resumes silently and "
        "would inherit stale results (F029).",
    )
    args = parser.parse_args()

    import gepa

    from gepa_taxonomy.bedrock import (
        BedrockLM,
        MeteredReflectionLM,
        require_credentials,
        verify_reflection_lm,
    )
    from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
    from gepa_taxonomy.livebench_math.adapter import LiveBenchMathAdapter, instances_by_id
    from gepa_taxonomy.livebench_math.program import COMPONENTS, SEED_CANDIDATE, SolveReviewProgram
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    arm = "taxonomy" if args.taxonomy else "baseline"
    out = args.out or REPO / "results" / "runs" / f"livebench-math-{arm}-seed{args.seed}"

    # Checked BEFORE credentials: the cheapest check that can abort a run should
    # be first. gepa RESUMES silently from an existing run_dir, so a run
    # relaunched after a code fix would inherit the OLD code's val scores,
    # candidate pool and Pareto frontier -- invisibly (F029).
    state = out / "gepa_state.bin"
    if state.exists() and not args.resume:
        raise SystemExit(
            f"REFUSING TO START: {state} already exists.\n"
            f"gepa would silently RESUME from it, inheriting that run's scores and\n"
            f"candidate pool. If the code changed since, every result would be\n"
            f"contaminated and would look normal.\n\n"
            f"  archive it:  mv {out} {out}.$(date +%Y%m%d_%H%M%S)\n"
            f"  or resume deliberately:  --resume"
        )
    out.mkdir(parents=True, exist_ok=True)

    # gepa's logger opens the run log with no encoding argument, so on Windows it
    # uses cp1252. Reflection routinely proposes prompts containing emoji or
    # typographic punctuation, and writing one raises UnicodeEncodeError -- which
    # killed a run at iteration 2 (F031).
    import locale

    if (locale.getpreferredencoding(False) or "").lower() not in {"utf-8", "utf8"}:
        raise SystemExit(
            f"REFUSING TO START: default encoding is "
            f"{locale.getpreferredencoding(False)!r}, not UTF-8.\n"
            f"gepa writes its run log with the platform default, and a proposed\n"
            f"prompt containing any non-cp1252 character will kill the run.\n\n"
            f"  relaunch with:  PYTHONUTF8=1 uv run python {Path(__file__).name} ..."
        )

    require_credentials()

    train = load_instances(args.manifests / "train.json")
    val = load_instances(args.manifests / "val.json")
    print(f"arm={arm} seed={args.seed} budget=${args.budget:.2f} train={len(train)} val={len(val)}")

    solver_meter, reflection_meter, judge_meter = CostMeter(), CostMeter(), CostMeter()

    program = SolveReviewProgram(
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=solver_meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )

    seed_cache = None
    if str(args.base_val_cache).lower() != "none":
        if not args.base_val_cache.exists():
            raise SystemExit(
                f"base-val cache not found: {args.base_val_cache}\n"
                "Every seed and both arms must start from the SAME base-candidate\n"
                "evaluation, or the paired comparison carries an extra 90-rollout\n"
                "draw of noise on top of the treatment (D009).\n\n"
                "  build it:  PYTHONUTF8=1 uv run python scripts/build_livebench_math_base_val.py\n"
                "  or opt out deliberately:  --base-val-cache none"
            )
        seed_cache = SeedEvaluationCache.load(args.base_val_cache)
        if not seed_cache.matches(dict(SEED_CANDIDATE)):
            raise SystemExit(
                "base-val cache was built for a DIFFERENT seed candidate.\n"
                "Replaying it would start this run from another program's results.\n"
                "Rebuild: scripts/build_livebench_math_base_val.py --force"
            )
        # D009's completeness guarantee, checked ONCE against the val manifest --
        # not inferred from a per-lookup miss, which is legitimate for train
        # instances and crashed a run when treated as an error (F016).
        seed_cache.assert_covers(i.task.example_id for i in val)
        print(f"base-val cache: {len(seed_cache.entries)} val instances will be replayed (no spend)")

    adapter = LiveBenchMathAdapter(
        program=program,
        instances=instances_by_id([*train, *val]),
        # Ground truth reaches reflection for TRAIN ids only (D028).
        reflection_gold_ids=frozenset(i.task.example_id for i in train),
        max_transport_errors=args.max_transport_errors,
        max_workers=args.workers,
        seed_cache=seed_cache,
    )

    meters = [solver_meter, reflection_meter]
    taxonomy_feedback = None
    if args.taxonomy:
        import inspect

        from failure_taxonomy import JudgeCache, LLMFailureJudge, TaxonomyFeedbackEnricher, load_taxonomy

        if "reflective_dataset_enricher" not in inspect.signature(gepa.optimize).parameters:
            raise SystemExit(
                "taxonomy runs require GEPA's optimizer-side reflective_dataset_enricher hook. "
                "Apply patches/gepa-reflective-dataset-enricher.patch to the pinned GEPA checkout."
            )

        judge_lm = BedrockLM(model=args.reflection_model, max_retries=args.max_retries)
        taxonomy = load_taxonomy(args.taxonomy)

        def judge_call(prompt: str) -> str:
            text, tin, tout = judge_lm.complete(prompt, max_tokens=4096)
            judge_meter.record(
                model=args.reflection_model, input_tokens=tin, output_tokens=tout, phase="optimization"
            )
            return text

        taxonomy_feedback = TaxonomyFeedbackEnricher(
            judge=LLMFailureJudge(
                taxonomy=taxonomy,
                lm=judge_call,
                cache=JudgeCache.open(out / "judge_cache.jsonl"),
            ),
        )
        # Judge spend competes for the SAME budget as rollouts and reflection.
        # The treatment arm may buy fewer rollouts for the same money; that trade
        # is what the comparison measures, not a confound (D032).
        meters.append(judge_meter)
        print(f"taxonomy: {args.taxonomy} ({len(taxonomy)} codes, fingerprint {taxonomy.fingerprint})")

    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=args.reflection_model, max_retries=args.max_retries),
        meter=reflection_meter,
        model=args.reflection_model,
        spend_log=out / "reflection_spend.jsonl",
    )
    # Free preflight. gepa SWALLOWS a non-conformant reflection LM and logs "did
    # not propose a new candidate", so the run would burn its budget while never
    # leaving the seed candidate. Fail here instead (F025).
    print(f"reflection LM preflight: {verify_reflection_lm(reflection_lm)}")

    stopper = MaxTotalCostStopper(args.budget, meters)

    optimize_kwargs: dict[str, object] = {}
    if args.log_reflection_datasets:
        from gepa_taxonomy.reflection_log import ReflectionDatasetLogger

        optimize_kwargs["callbacks"] = [ReflectionDatasetLogger(out / "reflection_datasets.jsonl")]
    if args.max_metric_calls is not None:
        optimize_kwargs["max_metric_calls"] = args.max_metric_calls
    if taxonomy_feedback is not None:
        optimize_kwargs["reflective_dataset_enricher"] = taxonomy_feedback

    started = time.time()
    result = gepa.optimize(
        seed_candidate=dict(SEED_CANDIDATE),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.minibatch_size,
        stop_callbacks=[stopper],
        seed=args.seed,
        display_progress_bar=True,
        run_dir=str(out),
        **optimize_kwargs,
    )
    elapsed = time.time() - started

    summary = {
        "arm": arm,
        "benchmark": "livebench_math",
        "seed": args.seed,
        "budget_usd": args.budget,
        "minibatch_size": args.minibatch_size,
        "components": list(COMPONENTS),
        "elapsed_hours": round(elapsed / 3600, 3),
        "best_val_score": max(result.val_aggregate_scores) if result.val_aggregate_scores else None,
        # The INDEX matters as much as the score: gepa orders candidates by
        # discovery, not by quality, so test evaluation cannot recover "the best
        # one" from the score alone without re-deriving it.
        "best_candidate_index": (
            max(range(len(result.val_aggregate_scores)), key=result.val_aggregate_scores.__getitem__)
            if result.val_aggregate_scores
            else None
        ),
        "val_aggregate_scores": list(result.val_aggregate_scores or []),
        "candidates": len(result.candidates),
        "spend": {
            "solver": solver_meter.snapshot(),
            "reflection": reflection_meter.snapshot(),
            "judge": judge_meter.snapshot(),
        },
        "adapter": adapter.summary(),
        "taxonomy_feedback": taxonomy_feedback.summary() if taxonomy_feedback is not None else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps(result.candidates, indent=2) + "\n", encoding="utf-8")

    print(f"\nbest val: {summary['best_val_score']}  candidates: {summary['candidates']}")
    print(f"realised: ${solver_meter.budgeted_usd + reflection_meter.budgeted_usd + judge_meter.budgeted_usd:.2f}")
    print(f"written : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
