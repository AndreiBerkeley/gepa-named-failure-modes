#!/usr/bin/env python
"""Evaluate a finished seed's BEST candidate on the held-out test split.

**This spends money: one pass over the test split per candidate.**

Test is touched exactly once per candidate, at the end. val drives selection, so
a val score is an optimistically biased estimate of it, which is why this is a
separate script and a separate split.

    PYTHONUTF8=1 uv run python demo/eval_ifbench_test.py --run results/runs/ifbench-baseline-seed1
    PYTHONUTF8=1 uv run python demo/eval_ifbench_test.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def evaluate_run(run: Path, args) -> dict | None:
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.ifbench.adapter import IFBenchAdapter, instances_by_id
    from gepa_taxonomy.ifbench.grading import Grade, report
    from gepa_taxonomy.ifbench.program import SEED_CANDIDATE, GenerateEnsureProgram
    from gepa_taxonomy.lm import MeteredLM

    summary_path = run / "summary.json"
    if not summary_path.exists():
        print(f"  {run.name}: no summary.json -- run unfinished, skipping")
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = json.loads((run / "candidates.json").read_text(encoding="utf-8"))

    # --candidate-index 0 evaluates the SEED candidate, which gepa stores as
    # candidate 0. That reference is not optional: without a base-on-test number
    # we can report "the best candidate scores X" but not "GEPA improved test by
    # Y". The result is written to its own file so it never clobbers the
    # best-candidate evaluation.
    if args.candidate_index is not None:
        index = args.candidate_index
        if not 0 <= index < len(candidates):
            print(f"  {run.name}: candidate index {index} out of range (0..{len(candidates) - 1})")
            return None
        out_path = run / f"test_eval_cand{index}.json"
    else:
        index = summary.get("best_candidate_index")
        if index is None:
            print(f"  {run.name}: no best_candidate_index, skipping")
            return None
        out_path = run / "test_eval.json"

    if out_path.exists() and not args.force:
        print(f"  {run.name}: already evaluated ({out_path.name}); --force to redo")
        return json.loads(out_path.read_text(encoding="utf-8"))

    candidate = candidates[index]

    import importlib.util

    spec = importlib.util.spec_from_file_location("_ifb_runner", REPO / "demo" / "run_ifbench_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    test = runner.load_instances(args.manifests / "test.json")

    meter = CostMeter()
    program = GenerateEnsureProgram(
        lm=MeteredLM(model=args.solver_model, max_retries=8),
        meter=meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )
    adapter = IFBenchAdapter(program=program, instances=instances_by_id(test), max_workers=args.workers)

    started = time.time()
    batch = adapter.evaluate(test, candidate, capture_traces=True)
    elapsed = time.time() - started

    if adapter.transport_errors:
        print(f"  {run.name}: {adapter.transport_errors} transport errors -- REFUSING to write a corrupted score")
        return None

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
    metrics = report(grades, [i.gold for i in test])

    payload = {
        "run": run.name,
        "arm": summary.get("arm"),
        "seed": summary.get("seed"),
        "best_candidate_index": index,
        "n": len(test),
        "test_score": metrics["instruction_level_strict"],
        # Only meaningful for the run's best candidate; None otherwise, rather
        # than silently reporting some other candidate's val score.
        "val_score": summary.get("best_val_score") if args.candidate_index is None else None,
        "prompt_level_strict": metrics["prompt_level_strict"],
        "instruction_level_loose": metrics["instruction_level_loose"],
        "by_family": metrics["by_family"],
        # The per-instance vector, so a paired Wilcoxon across arms is possible
        # later without re-spending. The test needs the pairs, not the means.
        "per_instance": {t["example_id"]: s for t, s in zip(batch.trajectories, batch.scores, strict=True)},
        "elapsed_minutes": round(elapsed / 60, 2),
        "spend_usd": meter.total_usd,
        "verifier_errors": adapter.verifier_errors,
        "is_seed_candidate": candidate == dict(SEED_CANDIDATE),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if payload["val_score"] is None:
        label = "SEED (base reference)" if index == 0 else f"candidate {index}"
        print(f"  {run.name}: {label} test {payload['test_score']:.4f}  ${meter.total_usd:.2f}")
    else:
        drop = payload["val_score"] - payload["test_score"]
        print(
            f"  {run.name}: test {payload['test_score']:.4f}  val {payload['val_score']:.4f}  "
            f"(val-test {drop:+.4f})  ${meter.total_usd:.2f}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="every ifbench run with a summary.json")
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
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help="evaluate this candidate instead of the run's best. Pass 0 for the SEED "
        "candidate -- the base-on-test reference, without which a test score has "
        "nothing to be compared against. Written to test_eval_cand<N>.json.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from gepa_taxonomy.cost import assert_priced, load_price_overrides

    load_price_overrides(args.price)
    assert_priced(args.solver_model)

    if args.all:
        runs = sorted((REPO / "results" / "runs").glob("ifbench-*"))
    elif args.run:
        runs = [args.run]
    else:
        raise SystemExit("pass --run PATH or --all")

    results = [r for r in (evaluate_run(run, args) for run in runs) if r]
    if len(results) > 1:
        print("\n  arm/seed summary:")
        for r in sorted(results, key=lambda x: (x["arm"] or "", x["seed"] or 0)):
            # val_score is None for an explicitly-indexed candidate (e.g. the
            # seed reference), so it cannot be formatted unconditionally.
            val = f"{r['val_score']:.4f}" if r["val_score"] is not None else "n/a"
            print(f"    {r['arm']:<9} seed {r['seed']}: test {r['test_score']:.4f}  val {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
