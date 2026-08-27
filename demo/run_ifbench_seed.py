#!/usr/bin/env python
"""Run one IFBench seed. **This spends money.** Launch it deliberately.

Baseline arm (no ``--taxonomy``) is unmodified gepa v0.1.4 driving our adapter.
The only addition is the dollar-budget stopper, which observes spend and nothing
else (baseline purity).

Treatment arm (``--taxonomy PATH``) adds an optimizer-side trace review between
the adapter's normal reflective dataset and GEPA's proposal call. The adapter is
identical across arms. With the flag absent, no enricher or judge is constructed.

    # baseline
    PYTHONUTF8=1 uv run python scripts/run_ifbench_seed.py --seed 1 --budget 60

    # treatment, same budget, same seed
    PYTHONUTF8=1 uv run python scripts/run_ifbench_seed.py --seed 1 --budget 60 \
        --taxonomy results/taxonomy/<benchmark>-auto-v1/taxonomy.json
"""

from __future__ import annotations

import argparse
import atexit
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_instances(manifest_path: Path):
    """Load one split, choosing the record parser from the manifest's dataset.

    train/val come from IF-RLVR Train and test from IFBench, and the two
    have different row shapes AND disjoint constraint vocabularies. The manifest
    records which dataset it was built from, so the parser is looked up rather
    than guessed -- a mismatch here would attach the wrong verifier registry to
    every instance and score the whole split against constraints it never had.
    """
    from datasets import load_dataset

    from gepa_taxonomy.ifbench.splits import TEST_DATASET, TRAIN_DATASET, _sort_key
    from gepa_taxonomy.ifbench.tasks import instance_from_ifbench, instance_from_ifrlvr

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    parsers = {TRAIN_DATASET: instance_from_ifrlvr, TEST_DATASET: instance_from_ifbench}
    if dataset not in parsers:
        raise SystemExit(f"manifest {manifest_path} names an unknown dataset {dataset!r}")
    parse = parsers[dataset]

    wanted = set(manifest["example_ids"])
    ds = load_dataset(dataset, split=manifest["dataset_split"])
    by_id = {str(r["key"]): parse(r) for r in ds if str(r["key"]) in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids missing from {dataset}, e.g. {sorted(missing)[:3]}")
    # Manifest order, which gepa keys val subscores against POSITIONALLY.
    return [by_id[i] for i in sorted(wanted, key=_sort_key)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="run seed (1, 2, 3)")
    parser.add_argument("--budget", type=float, required=True, help="dollar budget for this seed")
    parser.add_argument("--taxonomy", type=Path, default=None, help="enable the treatment arm")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=5,
        help="instances per reflective minibatch. 5 rather than gepa's default 3: "
        "256 of 300 IFBench instances carry a single constraint, so scoring is "
        "mostly binary and a small minibatch ties often -- and a tie is a wasted "
        "iteration, since gepa accepts on STRICT improvement only.",
    )
    parser.add_argument("--manifests", type=Path, default=REPO / "demo" / "manifests")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver-model", default="gpt-5-mini")
    parser.add_argument("--reflection-model", default="gpt-5-mini")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="hard cap on rollouts, independent of spend. The dollar budget is the "
        "primary control; this is a backstop for when the per-rollout cost estimate "
        "is wrong -- which it was by 87%% on AppWorld.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="ceiling per call; billed on tokens produced")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--log-reflection-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append every post-enrichment reflective dataset to "
        "reflection_datasets.jsonl in the run dir, exactly as reflection "
        "consumed it (observability; disable with --no-log-reflection-datasets)",
    )
    parser.add_argument("--max-transport-errors", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="instances evaluated concurrently. Rollouts are network-bound, so this "
        "is near-linear in this range. Lower it when another arm is running: an "
        "exhausted retry scores 0.0, which the optimizer cannot distinguish from a "
        "genuinely bad candidate.",
    )
    parser.add_argument(
        "--base-val-cache",
        type=Path,
        default=REPO / "results" / "ifbench_base_val" / "base_val_cache.json",
        help="shared base-candidate val evaluation, replayed so every seed and both "
        "arms start from byte-identical state. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="deliberately resume from an existing gepa_state.bin. Without this the "
        "run REFUSES to start when state exists, because gepa resumes silently and "
        "would inherit stale results.",
    )
    args = parser.parse_args()

    import gepa

    from gepa_taxonomy.cost import CostMeter, MaxTotalCostStopper
    from gepa_taxonomy.ifbench.adapter import IFBenchAdapter, instances_by_id
    from gepa_taxonomy.ifbench.program import COMPONENTS, SEED_CANDIDATE, GenerateEnsureProgram
    from gepa_taxonomy.lm import (
        MeteredLM,
        MeteredReflectionLM,
        verify_reflection_lm,
    )
    from gepa_taxonomy.seed_cache import SeedEvaluationCache

    arm = "taxonomy" if args.taxonomy else "baseline"
    out = args.out or REPO / "results" / "runs" / f"ifbench-{arm}-seed{args.seed}"

    # Checked BEFORE credentials: the cheapest check that can abort a run should
    # be first. gepa RESUMES silently from an existing run_dir, so a relaunch
    # after a code fix would inherit the OLD code's scores, invisibly.
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

    # gepa's logger writes its run log with the platform default encoding, so a
    # proposed prompt containing an emoji kills the run on Windows.
    import locale

    if (locale.getpreferredencoding(False) or "").lower() not in {"utf-8", "utf8"}:
        raise SystemExit(
            f"REFUSING TO START: default encoding is "
            f"{locale.getpreferredencoding(False)!r}, not UTF-8.\n"
            f"  relaunch with:  PYTHONUTF8=1 uv run python {Path(__file__).name} ..."
        )

    train = load_instances(args.manifests / "train.json")
    val = load_instances(args.manifests / "val.json")
    print(f"arm={arm} seed={args.seed} budget=${args.budget:.2f} train={len(train)} val={len(val)}")

    # One live snapshot per stream: solver, reflection and judge are budgeted
    # together but diagnosed separately, and only the reflection one used to
    # reach disk during a run.
    solver_meter = CostMeter(spend_log=out / "spend.solver.json")
    reflection_meter = CostMeter(spend_log=out / "spend.reflection.json")
    judge_meter = CostMeter(spend_log=out / "spend.judge.json")

    # Meters snapshot to disk every 25 records; a short run can end before the
    # first snapshot, leaving spend files missing. Exit-time flush makes the
    # on-disk record exhaustive, and the heartbeat keeps quiet phases visible.
    for _meter in (solver_meter, reflection_meter, judge_meter):
        atexit.register(_meter.flush)
    from gepa_taxonomy.progress import report_spend

    report_spend((solver_meter, reflection_meter, judge_meter))

    program = GenerateEnsureProgram(
        lm=MeteredLM(model=args.solver_model, max_retries=args.max_retries),
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
                "evaluation, or the paired comparison carries an extra 120-rollout\n"
                "draw of noise on top of the treatment.\n\n"
                "  build it:  PYTHONUTF8=1 uv run python scripts/build_ifbench_base_val.py\n"
                "  or opt out deliberately:  --base-val-cache none"
            )
        seed_cache = SeedEvaluationCache.load(args.base_val_cache)
        if not seed_cache.matches(dict(SEED_CANDIDATE)):
            raise SystemExit(
                "base-val cache was built for a DIFFERENT seed candidate.\n"
                "Replaying it would start this run from another program's results.\n"
                "Rebuild: scripts/build_ifbench_base_val.py --force"
            )
        # Checked ONCE against the val manifest, not inferred from a per-lookup
        # miss -- which is legitimate for train instances and crashed a run.
        seed_cache.assert_covers(i.task.example_id for i in val)
        print(f"base-val cache: {len(seed_cache.entries)} val instances will be replayed (no spend)")

    adapter = IFBenchAdapter(
        program=program,
        instances=instances_by_id([*train, *val]),
        # Failed-constraint names reach reflection for TRAIN ids only.
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

        judge_lm = MeteredLM(model=args.reflection_model, max_retries=args.max_retries)
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
        # is what the comparison measures, not a confound.
        meters.append(judge_meter)
        print(f"taxonomy: {args.taxonomy} ({len(taxonomy)} codes, fingerprint {taxonomy.fingerprint})")

    reflection_lm = MeteredReflectionLM(
        lm=MeteredLM(model=args.reflection_model, max_retries=args.max_retries),
        meter=reflection_meter,
        model=args.reflection_model,
        spend_log=out / "reflection_spend.jsonl",
    )
    # Free preflight. gepa SWALLOWS a non-conformant reflection LM and logs "did
    # not propose a new candidate", so the run would burn its budget while never
    # leaving the seed candidate.
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
        "benchmark": "ifbench",
        "seed": args.seed,
        "budget_usd": args.budget,
        "minibatch_size": args.minibatch_size,
        "components": list(COMPONENTS),
        "elapsed_hours": round(elapsed / 3600, 3),
        "best_val_score": max(result.val_aggregate_scores) if result.val_aggregate_scores else None,
        # The INDEX matters as much as the score: gepa orders candidates by
        # discovery, not quality, so test evaluation cannot recover "the best one"
        # from the score alone.
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
