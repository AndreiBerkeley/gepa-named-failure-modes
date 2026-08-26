#!/usr/bin/env python
"""Run one HotpotQA seed. **This spends money.** Andrei launches it, not Claude.

Baseline arm (no ``--taxonomy``) is unmodified gepa v0.1.4 driving our adapter.
The only addition is the dollar-budget stopper, which observes spend and
nothing else (CLAUDE.md hard rule 2).

Treatment arm (``--taxonomy PATH``) adds an optimizer-side trace review between
the adapter's normal reflective dataset and GEPA's proposal call. The adapter is
identical across arms. With the flag absent, no enricher or judge is constructed.

    # baseline
    uv run python scripts/run_hotpotqa_seed.py --seed 1 --budget 25

    # treatment, same budget, same seed
    uv run python scripts/run_hotpotqa_seed.py --seed 1 --budget 25 \
        --taxonomy results/taxonomy/hotpotqa_v1/taxonomy.pruned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_instances(manifest_path: Path):
    from datasets import load_dataset

    from gepa_taxonomy.hotpotqa.tasks import instance_from_record

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = set(manifest["example_ids"])
    ds = load_dataset(manifest["dataset"], manifest["dataset_config"], split=manifest["dataset_split"])
    by_id = {r["id"]: instance_from_record(r) for r in ds if r["id"] in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids missing from the dataset, e.g. {sorted(missing)[:3]}")
    # Manifest order, which is sorted: gepa keys val subscores and the Pareto
    # frontier POSITIONALLY (F014), so this ordering is load-bearing.
    return [by_id[i] for i in sorted(wanted)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="run seed (1, 2, 3)")
    parser.add_argument("--budget", type=float, required=True, help="dollar budget for this seed")
    parser.add_argument("--taxonomy", type=Path, default=None, help="enable the treatment arm")
    parser.add_argument("--minibatch-size", type=int, default=15)
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "hotpotqa")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--reflection-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--k", type=int, default=10, help="passages retrieved per hop")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="litellm retries per call. Default is raised above the usual 3 because "
        "concurrent seeds share one Bedrock quota, and an exhausted retry scores 0.0 "
        "-- indistinguishable to the optimizer from a genuinely bad candidate.",
    )
    parser.add_argument(
        "--max-transport-errors",
        type=int,
        default=25,
        help="abort after this many rollouts fail to reach the model",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="instances evaluated concurrently. Rollouts are network-bound "
        "(measured 0.0 CPU-seconds per 20s elapsed), so this is near-linear in "
        "this range. Held at 8 rather than higher to stay clear of the Bedrock "
        "rate limit: an exhausted retry scores 0.0, which the optimizer cannot "
        "distinguish from a genuinely bad candidate. Raise only if "
        "transport_errors stays at 0.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=20000,
        help="hard cap on rollouts, independent of spend. NOT redundant with the "
        "dollar budget: on 2026-08-14 the network failed mid-run, every rollout "
        "errored before reaching the model, spend froze near $0 -- so the budget "
        "stopper could never fire -- and the run spun to iteration 7,490 over 8.5 "
        "hours (expected ~57). A run that cannot spend cannot be stopped by a "
        "spend limit. 20000 is ~2x the ~11,300 a healthy $100 seed uses.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=8,
        help="traces judged concurrently in the taxonomy arm. Judging is "
        "network-bound and per-trace independent, so this is wall-clock only -- "
        "identical occurrences either way. Ignored without --taxonomy.",
    )
    parser.add_argument(
        "--base-val-cache",
        type=Path,
        default=REPO / "results" / "base_val" / "base_val_cache.json",
        help="shared base-candidate val evaluation, replayed so every seed and both "
        "arms start from byte-identical state (D009). Build it with "
        "scripts/build_hotpotqa_base_val.py. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="deliberately resume from an existing gepa_state.bin in the output "
        "directory. Without this the run REFUSES to start when state exists, "
        "because gepa resumes silently and would inherit stale results (F029).",
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
    from gepa_taxonomy.hotpotqa.adapter import HotpotQAAdapter, instances_by_id
    from gepa_taxonomy.hotpotqa.program import COMPONENTS, SEED_CANDIDATE, MultiHopProgram
    from gepa_taxonomy.hotpotqa.retrieval import WikiRetriever
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    arm = "taxonomy" if args.taxonomy else "baseline"
    out = args.out or REPO / "results" / "runs" / f"hotpotqa-{arm}-seed{args.seed}"

    # Checked BEFORE credentials: the cheapest check that can abort a run should
    # be the first one, so a stale-state mistake surfaces instantly.
    #
    # gepa RESUMES silently from an existing run_dir: "If the directory already
    # exists, GEPA will read the state from this directory and resume the
    # optimization from the last saved state" (api.py:176). A run relaunched
    # after a code fix would therefore inherit the OLD code's val scores,
    # candidate pool, Pareto frontier and evaluation cache -- and the base
    # candidate is never re-evaluated, so the corruption is invisible in the
    # output. This exact thing happened on 2026-08-12 (F029). Refuse rather than
    # resume by accident; --resume is how you ask for it on purpose.
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

    # gepa's logger opens the run log with no encoding argument
    # (``logging/logger.py:48``), so on Windows it uses cp1252. Reflection
    # routinely proposes prompts containing emoji or typographic punctuation --
    # a real proposal here contained U+274C -- and writing one raises
    # UnicodeEncodeError, which killed a run at iteration 2 (F031). Refuse to
    # start rather than die seven minutes in.
    import locale

    if (locale.getpreferredencoding(False) or "").lower() not in {"utf-8", "utf8"}:
        raise SystemExit(
            f"REFUSING TO START: default encoding is "
            f"{locale.getpreferredencoding(False)!r}, not UTF-8.\n"
            f"gepa writes its run log with the platform default, and a proposed\n"
            f"prompt containing any non-cp1252 character (emoji, curly quotes,\n"
            f"dashes) will raise UnicodeEncodeError and kill the run mid-flight.\n\n"
            f"  relaunch with:  PYTHONUTF8=1 uv run python {Path(__file__).name} ..."
        )

    require_credentials()

    train = load_instances(args.manifests / "train.json")
    val = load_instances(args.manifests / "val.json")
    print(f"arm={arm} seed={args.seed} budget=${args.budget:.2f} train={len(train)} val={len(val)}")

    # One live snapshot per stream: solver, reflection and judge are budgeted
    # together but diagnosed separately, and only the reflection one used to
    # reach disk during a run.
    solver_meter = CostMeter(spend_log=out / "spend.solver.json")
    reflection_meter = CostMeter(spend_log=out / "spend.reflection.json")
    judge_meter = CostMeter(spend_log=out / "spend.judge.json")
    retriever = WikiRetriever(k=args.k).load()
    instances = instances_by_id([*train, *val])

    program = MultiHopProgram(
        retriever=retriever,
        lm=BedrockLM(model=args.solver_model, max_retries=args.max_retries),
        meter=solver_meter,
        model=args.solver_model,
        k=args.k,
    )
    seed_cache = None
    if str(args.base_val_cache).lower() != "none":
        if not args.base_val_cache.exists():
            raise SystemExit(
                f"base-val cache not found: {args.base_val_cache}\n"
                "Every seed and both arms must start from the SAME base-candidate\n"
                "evaluation, or the paired comparison carries an extra 300-rollout\n"
                "draw of noise on top of the treatment (D009).\n\n"
                "  build it:  PYTHONUTF8=1 uv run python scripts/build_hotpotqa_base_val.py\n"
                "  or opt out deliberately:  --base-val-cache none"
            )
        seed_cache = SeedEvaluationCache.load(args.base_val_cache)
        if not seed_cache.matches(dict(SEED_CANDIDATE)):
            raise SystemExit(
                "base-val cache was built for a DIFFERENT seed candidate.\n"
                "Replaying it would start this run from another program's results.\n"
                "Rebuild: scripts/build_hotpotqa_base_val.py --force"
            )
        # D009's completeness guarantee, checked ONCE against the val manifest --
        # not inferred from a per-lookup miss, which is legitimate for train
        # instances and crashed a run when treated as an error (F016).
        seed_cache.assert_covers(i.task.example_id for i in val)
        print(f"base-val cache: {len(seed_cache.entries)} val instances will be replayed (no spend)")

    adapter = HotpotQAAdapter(
        program=program,
        instances=instances,
        # Gold answers reach reflection for TRAIN ids only (D028).
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
                # The judge runs on EVERY trace in a reflective minibatch, and
                # this arm's minibatch is 15 (matched to the baseline seeds), so
                # serial judging would add fifteen sequential Sonnet calls to
                # every iteration and roughly double the run. Concurrency changes
                # wall-clock only -- same prompts, same model, same occurrences,
                # asserted by tests/test_judge_parallel.py.
                max_workers=args.judge_workers,
            ),
        )
        # Judge spend competes for the SAME budget as rollouts and reflection.
        # The treatment arm may therefore buy fewer rollouts for the same money;
        # that trade is what the comparison measures, not a confound (D032).
        meters.append(judge_meter)
        print(f"taxonomy: {args.taxonomy} ({len(taxonomy)} codes, fingerprint {taxonomy.fingerprint})")

    reflection_lm = MeteredReflectionLM(
        lm=BedrockLM(model=args.reflection_model, max_retries=args.max_retries),
        meter=reflection_meter,
        model=args.reflection_model,
        # The out-of-process watchdog can only enforce a ceiling on what is on
        # disk; without this, reflection spend is invisible to it (D030).
        spend_log=out / "reflection_spend.jsonl",
    )
    # Free preflight. gepa SWALLOWS a non-conformant reflection LM and logs
    # "did not propose a new candidate", so the run would burn its whole budget
    # while never leaving the seed candidate. Fail here instead.
    print(f"reflection LM preflight: {verify_reflection_lm(reflection_lm)}")

    stopper = MaxTotalCostStopper(args.budget, meters)

    optimize_kwargs: dict[str, object] = {}
    if args.max_metric_calls:
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
