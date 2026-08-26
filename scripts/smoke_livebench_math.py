#!/usr/bin/env python
"""Live pre-flight for the LiveBench-Math arm. **Spends ~$0.05** (6 rollouts).

Everything the offline tests cannot check: that Bedrock is reachable, that the
real model's output actually parses under each of the three scorers, and what a
rollout really costs. The cost number is the point -- the AppWorld estimate was
87% low because it was extrapolated from three rollouts of the wrong shape, and
that error is what made a $100 seed buy 9 iterations instead of 20.

    PYTHONUTF8=1 uv run python scripts/smoke_livebench_math.py

Reads TRAIN ids only, so it cannot contaminate the shared base-val state.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="rollouts, sampled across all three scorers")
    parser.add_argument("--manifests", type=Path, default=REPO / "manifests" / "livebench_math")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--solver-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    args = parser.parse_args()

    from gepa_taxonomy.bedrock import BedrockLM, require_credentials
    from gepa_taxonomy.cost import CostMeter
    from gepa_taxonomy.livebench_math.adapter import LiveBenchMathAdapter, instances_by_id
    from gepa_taxonomy.livebench_math.grading import scorer_for
    from gepa_taxonomy.livebench_math.program import REVIEW, SEED_CANDIDATE, SOLVE, SolveReviewProgram

    require_credentials()

    import importlib.util

    spec = importlib.util.spec_from_file_location("_lbm_runner", REPO / "scripts" / "run_livebench_math_seed.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    train = runner.load_instances(args.manifests / "train.json")

    # One instance per scorer first, then fill -- a smoke that happens to draw six
    # multiple-choice questions proves nothing about the other two parsers.
    picked, seen = [], set()
    for inst in train:
        which = scorer_for(inst.task.subtask)
        if which not in seen:
            picked.append(inst)
            seen.add(which)
    picked += [i for i in train if i not in picked][: max(0, args.n - len(picked))]
    picked = picked[: args.n]
    print(f"smoke: {len(picked)} rollouts covering {sorted(seen)}")

    meter = CostMeter()
    program = SolveReviewProgram(
        lm=BedrockLM(model=args.solver_model, max_retries=4),
        meter=meter,
        model=args.solver_model,
        max_tokens=args.max_tokens,
    )
    adapter = LiveBenchMathAdapter(
        program=program,
        instances=instances_by_id(picked),
        reflection_gold_ids=frozenset(i.task.example_id for i in picked),
        max_workers=args.workers,
    )

    started = time.time()
    batch = adapter.evaluate(picked, dict(SEED_CANDIDATE), capture_traces=True)
    elapsed = time.time() - started

    print(f"\n{'id':<14} {'scorer':<16} {'score':>6} {'parsed':<12} {'steps':>6} {'$':>8}")
    for trace, score in zip(batch.trajectories, batch.scores, strict=True):
        g = trace["grading"]
        print(
            f"{trace['example_id'][:12]:<14} {g['scorer']:<16} {score:>6.2f} "
            f"{str(g['parsed'])[:10]:<12} {len(trace['module_calls']):>6} {trace['cost_usd']:>8.4f}"
        )

    per = meter.total_usd / max(1, len(picked))
    changed = sum(
        1
        for t in batch.trajectories
        if (t.get("draft_answer") or "").strip().splitlines()[-1:] != (t.get("answer") or "").strip().splitlines()[-1:]
    )

    # The reflective dataset is built here too: a launch-time failure in feedback
    # construction would otherwise surface only after the base val was paid for.
    dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), batch, [SOLVE, REVIEW])

    print(f"\n  mean score       : {statistics.mean(batch.scores):.3f}")
    print(f"  review changed   : {changed}/{len(picked)} answers")
    print(f"  transport errors : {adapter.transport_errors}   program errors: {adapter.program_errors}")
    print(f"  elapsed          : {elapsed:.1f}s  ({elapsed / len(picked):.1f}s/rollout)")
    print(f"  spend            : ${meter.total_usd:.4f}  (${per:.5f}/rollout)")
    print(f"  reflective set   : {[(k, len(v)) for k, v in dataset.items()]}")

    val_n, train_n = 90, 40
    print("\n  --- projected, from THIS measurement ---")
    print(f"  base val (90)    : ${per * val_n:.2f}")
    for mb in (3, 5):
        # 2*mb minibatch rollouts + reflection + accept_rate * full val.
        # 0.54 is the accept rate measured on HotpotQA seeds 1-2, not a guess.
        per_iter = 2 * mb * per + 0.05 + 0.54 * val_n * per
        print(f"  minibatch {mb:>2}     : ${per_iter:.2f}/iteration -> {30 / per_iter:.0f} iterations at $30")
    print(f"  (train pool {train_n}, val {val_n})")

    if adapter.transport_errors or adapter.program_errors:
        print("\n  NOT CLEAN -- do not launch until this is understood.")
        return 1
    print("\n  clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
